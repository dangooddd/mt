from argparse import ArgumentParser
from collections.abc import Callable
from multiprocessing import freeze_support
from pathlib import Path
from typing import Any, cast, no_type_check

import torch
import torch.optim as optim
from datasets import load_from_disk
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, ProgressBar, global_step_from_engine
from sacrebleu.metrics import CHRF
from torch import Tensor
from torch.amp import autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from mt.tokenizers import BaseTokenizer

from ..load import load_from_config
from ..modules import DecoderOnly, EncoderDecoder
from .utils import (
    BilingualEvalCollateFn,
    EvalCollateFn,
    OutputMetric,
    attach_tensorboard_logging,
    compute_bilingual_predictions,
    compute_predictions,
)


def completion_mask(token_ids: Tensor, eos_token_id: int) -> Tensor:
    eos_mask = token_ids.eq(eos_token_id)
    eos_count = eos_mask.cumsum(dim=1)
    return eos_count.eq(0) | (eos_mask & eos_count.eq(1))


def compute_chrf_rewards(
    predictions: list[str],
    references: list[str],
    metric: CHRF,
) -> list[float]:
    return [
        metric.sentence_score(prediction or "", [reference or ""]).score
        for prediction, reference in zip(predictions, references, strict=True)
    ]


def compute_advantages(
    rewards: list[float],
    batch_size: int,
    num_generations: int,
    device: torch.device,
    eps: float,
) -> tuple[Tensor, Tensor]:
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device).view(
        batch_size,
        num_generations,
    )
    mean = rewards_tensor.mean(dim=1, keepdim=True)
    std = rewards_tensor.std(dim=1, keepdim=True, unbiased=False)
    advantages = (rewards_tensor - mean) / (std + eps)
    return advantages.reshape(-1), rewards_tensor


def grpo_loss(token_log_probs: Tensor, mask: Tensor, advantages: Tensor) -> Tensor:
    weights = advantages.detach().view(-1, 1).to(dtype=token_log_probs.dtype)
    mask = mask.to(dtype=token_log_probs.dtype)
    return -((token_log_probs * weights * mask).sum() / mask.sum().clamp_min(1.0))


def encoder_decoder_token_log_probs(
    model: EncoderDecoder,
    src_ids: Tensor,
    src_mask: Tensor,
    generated_ids: Tensor,
) -> tuple[Tensor, Tensor]:
    decoder_input = generated_ids[:, :-1]
    targets = generated_ids[:, 1:]

    logits = model.forward(src_ids, decoder_input, src_mask)
    log_probs = logits.float().log_softmax(dim=-1)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    mask = completion_mask(targets, model.tgt_eos_token_id)
    return token_log_probs, mask


def decoder_only_token_log_probs(
    model: DecoderOnly,
    input_ids: Tensor,
    attention_mask: Tensor,
    type_ids: Tensor,
    generated_ids: Tensor,
) -> tuple[Tensor, Tensor]:
    batch_size, generated_length = generated_ids.shape
    prompt_lengths = attention_mask.long().sum(dim=1).clamp_min(1)
    max_prompt_length = int(prompt_lengths.max().item())
    total_length = max_prompt_length + generated_length
    device = input_ids.device

    full_ids = torch.full(
        (batch_size, total_length),
        model.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    full_mask = torch.zeros((batch_size, total_length), dtype=torch.bool, device=device)
    full_type_ids = torch.zeros((batch_size, total_length), dtype=torch.long, device=device)
    loss_mask = torch.zeros((batch_size, total_length - 1), dtype=torch.bool, device=device)
    generated_mask = completion_mask(generated_ids, model.eos_token_id)

    for i in range(batch_size):
        prompt_length = int(prompt_lengths[i].item())
        generated_start = prompt_length
        generated_end = generated_start + generated_length
        loss_start = prompt_length - 1
        loss_end = loss_start + generated_length

        full_ids[i, :prompt_length] = input_ids[i, :prompt_length]
        full_ids[i, generated_start:generated_end] = generated_ids[i]
        full_mask[i, :generated_end] = True
        full_type_ids[i, :prompt_length] = type_ids[i, :prompt_length]
        full_type_ids[i, generated_start:generated_end] = 1
        loss_mask[i, loss_start:loss_end] = generated_mask[i]

    logits = model.forward(full_ids[:, :-1], full_mask[:, :-1], full_type_ids[:, :-1])
    targets = full_ids[:, 1:]
    log_probs = logits.float().log_softmax(dim=-1)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return token_log_probs, loss_mask


def compute_encoder_decoder_grpo_loss(
    model: EncoderDecoder,
    batch: dict[str, Any],
    device: torch.device,
    tgt_tokenizer: BaseTokenizer,
    metric: CHRF,
    num_generations: int,
    max_length: int,
    temperature: float,
    advantage_eps: float,
) -> tuple[Tensor, dict[str, float]]:
    src_ids = batch["src_ids"].to(device=device)
    src_mask = batch["src_mask"].to(device=device)
    references = cast(list[str], batch["targets"])
    batch_size = src_ids.size(0)

    src_ids = src_ids.repeat_interleave(num_generations, dim=0)
    src_mask = src_mask.repeat_interleave(num_generations, dim=0)
    references = [reference for reference in references for _ in range(num_generations)]

    model.eval()
    generated_ids = model.inference(src_ids, src_mask, max_length, temperature=temperature)
    predictions = tgt_tokenizer.decode_batch(generated_ids.detach().cpu().tolist())
    rewards = compute_chrf_rewards(predictions, references, metric)
    advantages, rewards_tensor = compute_advantages(
        rewards,
        batch_size=batch_size,
        num_generations=num_generations,
        device=device,
        eps=advantage_eps,
    )

    model.train()
    token_log_probs, mask = encoder_decoder_token_log_probs(model, src_ids, src_mask, generated_ids)
    loss = grpo_loss(token_log_probs, mask, advantages)

    return loss, {
        "reward": float(rewards_tensor.mean().detach().cpu()),
        "reward_std": float(rewards_tensor.std(unbiased=False).detach().cpu()),
        "completion_length": float(mask.float().sum(dim=1).mean().detach().cpu()),
    }


def compute_decoder_only_grpo_loss(
    model: DecoderOnly,
    batch: dict[str, Any],
    device: torch.device,
    tokenizer,
    metric: CHRF,
    num_generations: int,
    max_length: int,
    temperature: float,
    advantage_eps: float,
) -> tuple[Tensor, dict[str, float]]:
    input_ids = batch["inference_input_ids"].to(device=device)
    attention_mask = batch["inference_attention_mask"].to(device=device)
    type_ids = batch["inference_type_ids"].to(device=device)
    references = cast(list[str], batch["targets"])
    batch_size = input_ids.size(0)

    input_ids = input_ids.repeat_interleave(num_generations, dim=0)
    attention_mask = attention_mask.repeat_interleave(num_generations, dim=0)
    type_ids = type_ids.repeat_interleave(num_generations, dim=0)
    references = [reference for reference in references for _ in range(num_generations)]

    model.eval()
    generated_ids = model.inference(
        input_ids,
        attention_mask,
        type_ids,
        max_length,
        temperature=temperature,
    )
    predictions = tokenizer.decode_batch(generated_ids.detach().cpu().tolist())
    rewards = compute_chrf_rewards(predictions, references, metric)
    advantages, rewards_tensor = compute_advantages(
        rewards,
        batch_size=batch_size,
        num_generations=num_generations,
        device=device,
        eps=advantage_eps,
    )

    model.train()
    token_log_probs, mask = decoder_only_token_log_probs(
        model,
        input_ids,
        attention_mask,
        type_ids,
        generated_ids,
    )
    loss = grpo_loss(token_log_probs, mask, advantages)

    return loss, {
        "reward": float(rewards_tensor.mean().detach().cpu()),
        "reward_std": float(rewards_tensor.std(unbiased=False).detach().cpu()),
        "completion_length": float(mask.float().sum(dim=1).mean().detach().cpu()),
    }


def create_grpo_trainer(
    model: EncoderDecoder | DecoderOnly,
    optimizer: Optimizer,
    compute_loss: Callable[..., tuple[Tensor, dict[str, float]]],
    compute_loss_kwargs: dict,
    max_grad_norm: float = 1.0,
) -> Engine:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    model.to(device)

    def train_step(engine: Engine, batch: dict[str, Tensor]) -> dict[str, Any]:
        _ = engine
        model.train()
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
            loss, metrics = compute_loss(model, batch, device=device, **compute_loss_kwargs)

        loss.backward()
        grad_norm = clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        return {
            "loss": loss.detach(),
            "lr": optimizer.param_groups[0]["lr"],
            "grad_norm": grad_norm,
            **metrics,
        }

    trainer = Engine(train_step)
    OutputMetric(output_transform=lambda o: o["loss"]).attach(trainer, "loss")
    OutputMetric(output_transform=lambda o: o["lr"]).attach(trainer, "lr")
    OutputMetric(output_transform=lambda o: o["grad_norm"]).attach(trainer, "grad_norm")
    OutputMetric(output_transform=lambda o: o["reward"]).attach(trainer, "reward")
    OutputMetric(output_transform=lambda o: o["reward_std"]).attach(trainer, "reward_std")
    OutputMetric(output_transform=lambda o: o["completion_length"]).attach(
        trainer,
        "completion_length",
    )
    ProgressBar().attach(trainer, metric_names="all")
    return trainer


def create_chrf_evaluator(
    model: EncoderDecoder | DecoderOnly,
    compute_predictions: Callable[..., tuple[list[str], list[str]]],
    compute_predictions_kwargs: dict,
) -> Engine:
    device = next(model.parameters()).device
    use_amp = device.type == "cuda"
    chrf = CHRF()

    @torch.inference_mode()
    def evaluate_step(engine: Engine, batch: dict[str, Tensor]):
        _ = engine
        model.eval()

        with autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
            predictions, references = compute_predictions(
                model,
                batch,
                device,
                **compute_predictions_kwargs,
            )

        return {
            "predictions": predictions,
            "references": references,
        }

    evaluator = Engine(evaluate_step)
    pbar = ProgressBar(persist=True)

    evaluator.state_dict_user_keys.append("predictions")
    evaluator.state_dict_user_keys.append("references")

    @no_type_check
    @evaluator.on(Events.STARTED)
    def _reset(_):
        evaluator.state.predictions = []
        evaluator.state.references = []

    @no_type_check
    @evaluator.on(Events.ITERATION_COMPLETED)
    def _accumulate(_):
        output = evaluator.state.output
        evaluator.state.predictions.extend(output["predictions"])
        evaluator.state.references.extend(output["references"])

    @no_type_check
    @evaluator.on(Events.COMPLETED)
    def _finalize(_):
        score = chrf.corpus_score(evaluator.state.predictions, [evaluator.state.references]).score
        evaluator.state.metrics["chrf"] = float(score)

        if pbar.pbar is not None:
            pbar.pbar.set_postfix({"chrf": f"{score:.4f}"})
            pbar.pbar.refresh()

    pbar.attach(evaluator, metric_names="all", closing_event_name=Events.COMPLETED)
    return evaluator


def main() -> None:
    parser = ArgumentParser("Train MT model with GRPO and chRF reward")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--validation-split", type=str, default="validation")
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("data/checkpoints"))
    parser.add_argument("--runs-dir", type=Path, default=Path("data/runs"))
    parser.add_argument("--save-total-limit", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--src", type=str, default="ru")
    parser.add_argument("--tgt", type=str, default="en")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epoch-steps", type=int, default=100)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--advantage-eps", type=float, default=1e-6)
    parser.add_argument("--load-weights", action="store_true")
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset_path)
    model, tokenizers, _ = load_from_config(args.model_dir, load_weights=args.load_weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    chrf = CHRF()

    if isinstance(tokenizers, tuple):
        src_tokenizer, tgt_tokenizer = cast(tuple[BaseTokenizer, BaseTokenizer], tokenizers)
        train_collate_fn = EvalCollateFn(
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            max_src_length=args.max_length,
            max_tgt_length=args.max_length,
            src_column=args.src,
            tgt_column=args.tgt,
        )
        evaluation_collate_fn = EvalCollateFn(
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            max_src_length=args.max_length,
            max_tgt_length=args.max_length,
            src_column=args.src,
            tgt_column=args.tgt,
        )
        loss_fn = compute_encoder_decoder_grpo_loss
        loss_kwargs = {
            "tgt_tokenizer": tgt_tokenizer,
            "metric": chrf,
            "num_generations": args.num_generations,
            "max_length": args.max_length,
            "temperature": args.temperature,
            "advantage_eps": args.advantage_eps,
        }
        predictions_fn = compute_predictions
        predictions_kwargs = {
            "tgt_tokenizer": tgt_tokenizer,
            "max_length": args.max_length,
        }
    else:
        tokenizer = tokenizers
        train_collate_fn = BilingualEvalCollateFn(
            tokenizer=tokenizer,
            max_length=args.max_length,
            src_column=args.src,
            tgt_column=args.tgt,
        )
        evaluation_collate_fn = BilingualEvalCollateFn(
            tokenizer=tokenizer,
            max_length=args.max_length,
            src_column=args.src,
            tgt_column=args.tgt,
        )
        loss_fn = compute_decoder_only_grpo_loss
        loss_kwargs = {
            "tokenizer": tokenizer,
            "metric": chrf,
            "num_generations": args.num_generations,
            "max_length": args.max_length,
            "temperature": args.temperature,
            "advantage_eps": args.advantage_eps,
        }
        predictions_fn = compute_bilingual_predictions
        predictions_kwargs = {
            "tokenizer": tokenizer,
            "max_length": args.max_length,
        }

    train_loader = DataLoader(
        cast(Dataset, dataset[args.train_split]),
        batch_size=args.batch_size,
        collate_fn=train_collate_fn,
        num_workers=0,
        shuffle=True,
    )

    evaluation_loader = DataLoader(
        cast(Dataset, dataset[args.validation_split]),
        batch_size=args.batch_size,
        collate_fn=evaluation_collate_fn,
        num_workers=0,
    )

    trainer = create_grpo_trainer(
        model=model,
        optimizer=optimizer,
        compute_loss=loss_fn,
        compute_loss_kwargs=loss_kwargs,
        max_grad_norm=args.max_grad_norm,
    )

    evaluator = create_chrf_evaluator(
        model=model,
        compute_predictions=predictions_fn,
        compute_predictions_kwargs=predictions_kwargs,
    )

    experiment = args.experiment or f"{args.model_dir.name}-grpo"
    checkpoints_dir = args.checkpoints_dir / experiment
    runs_dir = args.runs_dir / experiment

    checkpoint_handler = Checkpoint(
        {
            "model": model,
            "optimizer": optimizer,
            "trainer": trainer,
        },
        DiskSaver(checkpoints_dir, create_dir=True, require_empty=False),
        n_saved=args.save_total_limit,
        filename_prefix="grpo",
    )

    best_checkpoint_handler = Checkpoint(
        {"model": model},
        DiskSaver(checkpoints_dir / "best", create_dir=True, require_empty=False),
        n_saved=1,
        filename_prefix="best",
        score_name="chrf",
        score_function=lambda engine: engine.state.metrics["chrf"],
    )

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=args.epoch_steps),
        checkpoint_handler,
    )

    evaluator.add_event_handler(Events.COMPLETED, best_checkpoint_handler)

    train_tb_logger = attach_tensorboard_logging(
        trainer,
        runs_dir,
        tag="train",
        every=args.log_every,
    )

    evaluation_tb_logger = attach_tensorboard_logging(
        evaluator,
        runs_dir,
        tag="evaluation",
        global_step_transform=global_step_from_engine(trainer),
    )

    @trainer.on(Events.COMPLETED)
    def _close_loggers(_):
        train_tb_logger.close()
        evaluation_tb_logger.close()

    @trainer.on(Events.EPOCH_COMPLETED)
    def _run_validation(_):
        evaluator.run(evaluation_loader)

    trainer.run(
        train_loader,
        max_epochs=args.steps // args.epoch_steps,
        epoch_length=args.epoch_steps,
    )


if __name__ == "__main__":
    freeze_support()
    main()
