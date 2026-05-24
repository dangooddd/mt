from argparse import ArgumentParser
from copy import deepcopy
from multiprocessing import freeze_support
from pathlib import Path
from typing import cast

import torch.optim as optim
from datasets import load_from_disk
from ignite.engine import Events
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from torch.utils.data import DataLoader, Dataset

from mt.models.modules import DecoderOnly, EncoderDecoder, EncoderDecoderBilingual
from mt.tokenizers import BaseTokenizer, BilingualBaseTokenizer, DecoderBaseTokenizer

from ..load import load_from_config
from .utils.dapo import compute_decoder_only_dapo_loss, compute_encoder_decoder_dapo_loss
from .utils.pretrain import (
    BilingualCollateFn,
    BilingualEvalCollateFn,
    DecoderEvalCollateFn,
    EvalCollateFn,
    attach_tensorboard_logging,
    compute_bilingual_loss,
    compute_decoder_loss,
    compute_decoder_predictions,
    compute_predictions,
    create_evaluator,
    create_trainer,
    split_decay_params,
)
from .utils.pretrain import compute_loss as compute_pretrain_loss
from .utils.rewards import create_reward_scorer


def main() -> None:
    parser = ArgumentParser("Train MT model with DAPO")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--validation-split", type=str, default="validation")
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("data/checkpoints"))
    parser.add_argument("--runs-dir", type=Path, default=Path("data/runs"))
    parser.add_argument("--save-total-limit", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--src", type=str, default="ru")
    parser.add_argument("--tgt", type=str, default="en")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=125)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epoch-steps", type=int, default=125)
    parser.add_argument("--steps", type=int, default=6250)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--reward", choices=["chrf", "bleu", "comet", "mixed"], default="mixed")
    parser.add_argument("--comet-model-name", type=str, default="Unbabel/wmt22-comet-da")
    parser.add_argument("--comet-batch-size", type=int, default=100)
    parser.add_argument("--advantage-eps", type=float, default=1e-6)
    parser.add_argument("--reward-std-min", type=float, default=0.01)
    parser.add_argument("--clip-eps-low", type=float, default=0.2)
    parser.add_argument("--clip-eps-high", type=float, default=0.3)
    parser.add_argument("--kl-beta", type=float, default=0.005)
    parser.add_argument("--kl-clip", type=float, default=10.0)
    parser.add_argument("--sft-mu", type=float, default=0.0)
    parser.add_argument("--old-policy-update-steps", type=int, default=2)
    parser.add_argument("--no-load-weights", action="store_true")
    args = parser.parse_args()

    if args.kl_beta < 0.0:
        raise ValueError("kl-beta must be non-negative")

    if args.sft_mu < 0.0:
        raise ValueError("sft-mu must be non-negative")

    if args.kl_clip <= 0.0:
        raise ValueError("kl-clip must be positive")

    dataset = load_from_disk(args.dataset_path)
    model, tokenizers, _ = load_from_config(args.model_dir, load_weights=not args.no_load_weights)
    reward_scorer = create_reward_scorer(
        args.reward,
        comet_model_name=args.comet_model_name,
        comet_batch_size=args.comet_batch_size,
    )

    old_policy = deepcopy(model)
    old_policy.requires_grad_(False)
    old_policy.eval()

    reference_policy = deepcopy(model)
    reference_policy.requires_grad_(False)
    reference_policy.eval()

    decay, no_decay = split_decay_params(model)
    optimizer = optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
    )

    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=args.warmup_steps,
    )

    constant_scheduler = optim.lr_scheduler.ConstantLR(
        optimizer,
        factor=1.0,
        total_iters=args.steps - args.warmup_steps,
    )

    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, constant_scheduler],
        milestones=[args.warmup_steps],
    )

    if isinstance(model, EncoderDecoderBilingual):
        tokenizer = cast(BilingualBaseTokenizer, tokenizers)
        train_collate_fn = BilingualCollateFn(
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
        train_loss_fn = compute_encoder_decoder_dapo_loss
        train_loss_kwargs = {
            "old_policy": cast(EncoderDecoderBilingual, old_policy),
            "reference_policy": cast(EncoderDecoderBilingual, reference_policy),
            "tgt_tokenizer": tokenizer,
            "reward_scorer": reward_scorer,
            "num_generations": args.num_generations,
            "max_length": args.max_length,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "advantage_eps": args.advantage_eps,
            "clip_eps_low": args.clip_eps_low,
            "clip_eps_high": args.clip_eps_high,
            "kl_beta": args.kl_beta,
            "sft_mu": args.sft_mu,
            "reward_std_min": args.reward_std_min,
            "kl_clip": args.kl_clip,
        }
        evaluation_loss_fn = compute_bilingual_loss
        evaluation_loss_kwargs = {}
        predictions_fn = compute_predictions
        predictions_kwargs = {
            "tgt_tokenizer": tokenizer,
            "max_length": args.max_length,
        }

    elif isinstance(model, EncoderDecoder):
        src_tokenizer, tgt_tokenizer = cast(tuple[BaseTokenizer, BaseTokenizer], tokenizers)
        train_collate_fn = EvalCollateFn(
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            max_length=args.max_length,
            src_column=args.src,
            tgt_column=args.tgt,
        )
        evaluation_collate_fn = EvalCollateFn(
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            max_length=args.max_length,
            src_column=args.src,
            tgt_column=args.tgt,
        )
        train_loss_fn = compute_encoder_decoder_dapo_loss
        train_loss_kwargs = {
            "old_policy": cast(EncoderDecoder, old_policy),
            "reference_policy": cast(EncoderDecoder, reference_policy),
            "tgt_tokenizer": tgt_tokenizer,
            "reward_scorer": reward_scorer,
            "num_generations": args.num_generations,
            "max_length": args.max_length,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "advantage_eps": args.advantage_eps,
            "clip_eps_low": args.clip_eps_low,
            "clip_eps_high": args.clip_eps_high,
            "kl_beta": args.kl_beta,
            "sft_mu": args.sft_mu,
            "reward_std_min": args.reward_std_min,
            "kl_clip": args.kl_clip,
        }
        evaluation_loss_fn = compute_pretrain_loss
        evaluation_loss_kwargs = {}
        predictions_fn = compute_predictions
        predictions_kwargs = {
            "tgt_tokenizer": tgt_tokenizer,
            "max_length": args.max_length,
        }

    elif isinstance(model, DecoderOnly):
        tokenizer = cast(DecoderBaseTokenizer, tokenizers)
        train_collate_fn = DecoderEvalCollateFn(
            tokenizer=tokenizer,
            max_length=args.max_length,
            src_column=args.src,
            tgt_column=args.tgt,
        )
        evaluation_collate_fn = DecoderEvalCollateFn(
            tokenizer=tokenizer,
            max_length=args.max_length,
            src_column=args.src,
            tgt_column=args.tgt,
        )
        train_loss_fn = compute_decoder_only_dapo_loss
        train_loss_kwargs = {
            "old_policy": cast(DecoderOnly, old_policy),
            "reference_policy": cast(DecoderOnly, reference_policy),
            "tokenizer": tokenizer,
            "reward_scorer": reward_scorer,
            "num_generations": args.num_generations,
            "max_length": args.max_length,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "advantage_eps": args.advantage_eps,
            "clip_eps_low": args.clip_eps_low,
            "clip_eps_high": args.clip_eps_high,
            "kl_beta": args.kl_beta,
            "sft_mu": args.sft_mu,
            "reward_std_min": args.reward_std_min,
            "kl_clip": args.kl_clip,
        }
        evaluation_loss_fn = compute_decoder_loss
        evaluation_loss_kwargs = {}
        predictions_fn = compute_decoder_predictions
        predictions_kwargs = {
            "tokenizer": tokenizer,
            "max_length": args.max_length,
        }

    else:
        raise TypeError(f"Unsupported model type: {type(model)}")

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

    trainer = create_trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        compute_loss=train_loss_fn,
        compute_loss_kwargs=train_loss_kwargs,
        max_grad_norm=args.max_grad_norm,
    )

    evaluator = create_evaluator(
        model=model,
        compute_loss=evaluation_loss_fn,
        compute_loss_kwargs=evaluation_loss_kwargs,
        compute_predictions=predictions_fn,
        compute_predictions_kwargs=predictions_kwargs,
    )

    device = next(model.parameters()).device
    old_policy.to(device)
    reference_policy.to(device)

    def sync_old_policy() -> None:
        old_policy.load_state_dict(model.state_dict())
        old_policy.requires_grad_(False)
        old_policy.eval()

    @trainer.on(Events.STARTED)
    def _sync_old_policy_on_start(_):
        sync_old_policy()

    @trainer.on(Events.ITERATION_COMPLETED(every=args.old_policy_update_steps))
    def _sync_old_policy_on_interval(_):
        sync_old_policy()

    experiment = args.experiment or f"{args.model_dir.name}-dapo"
    checkpoints_dir = args.checkpoints_dir / experiment
    runs_dir = args.runs_dir / experiment

    checkpoint_handler = Checkpoint(
        {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "trainer": trainer,
        },
        DiskSaver(checkpoints_dir, create_dir=True, require_empty=False),
        n_saved=args.save_total_limit,
        filename_prefix="dapo",
    )

    best_checkpoint_handler = Checkpoint(
        {"model": model},
        DiskSaver(checkpoints_dir / "best", create_dir=True, require_empty=False),
        n_saved=1,
        filename_prefix="best",
        score_name="comet",
        score_function=lambda engine: engine.state.metrics["comet"],
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
