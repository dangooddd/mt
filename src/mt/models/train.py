from collections.abc import Callable
from pathlib import Path
from typing import Any, no_type_check

import torch
from ignite.engine import Engine, Events
from ignite.handlers.tensorboard_logger import TensorboardLogger, global_step_from_engine
from ignite.metrics import RunningAverage
from sacrebleu.metrics import BLEU
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.types import Device

from ..tokenizers import BaseTokenizer
from .luong import LuongSeq2Seq

Models = LuongSeq2Seq


class CollateFn:
    def __init__(
        self,
        src_tokenizer: BaseTokenizer,
        tgt_tokenizer: BaseTokenizer,
        src_column: str = "ru",
        tgt_column: str = "en",
        max_src_length: int | None = None,
        max_tgt_length: int | None = None,
    ):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_column = src_column
        self.tgt_column = tgt_column

        self.src_tokenizer.enable_padding(direction="right")
        self.tgt_tokenizer.enable_padding(direction="right")

        if max_src_length is not None:
            self.src_tokenizer.enable_truncation(max_src_length, direction="right")
        else:
            self.src_tokenizer.no_truncation()

        if max_tgt_length is not None:
            self.tgt_tokenizer.enable_truncation(max_tgt_length, direction="right")
        else:
            self.tgt_tokenizer.no_truncation()

    def __call__(self, batch: list[dict[str, str]]) -> dict[str, Tensor]:
        src_texts = [item[self.src_column] for item in batch]
        tgt_texts = [item[self.tgt_column] for item in batch]

        src_encodings = self.src_tokenizer.encode_batch(src_texts)
        tgt_encodings = self.tgt_tokenizer.encode_batch(tgt_texts)

        src_ids = [encoding.ids for encoding in src_encodings]
        tgt_ids = [encoding.ids for encoding in tgt_encodings]
        src_mask = [encoding.attention_mask for encoding in src_encodings]

        return {
            "src_mask": torch.tensor(src_mask, dtype=torch.bool),
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
        }


def build_collate_fn(
    src_tokenizer: BaseTokenizer,
    tgt_tokenizer: BaseTokenizer,
    src_column: str = "ru",
    tgt_column: str = "en",
    max_src_length: int | None = None,
    max_tgt_length: int | None = None,
) -> Callable[[list[dict[str, str]]], dict[str, Tensor]]:
    return CollateFn(
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_column=src_column,
        tgt_column=tgt_column,
        max_src_length=max_src_length,
        max_tgt_length=max_tgt_length,
    )


def compute_loss(
    model: Models,
    batch: dict[str, Tensor],
    criterion: CrossEntropyLoss,
    device: Device,
) -> Tensor:
    src_ids = batch["src_ids"].to(device=device)
    tgt_ids = batch["tgt_ids"].to(device=device)
    src_mask = batch["src_mask"].to(device=device)

    logits = model.forward(
        src_ids,
        tgt_ids,
        src_mask,
    )

    targets = tgt_ids[:, 1:]
    loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    return loss


def compute_predictions(
    model: Models,
    batch: dict[str, Tensor],
    device: Device,
    tgt_tokenizer: BaseTokenizer,
    max_length: int = 1024,
) -> tuple[list[str], list[str]]:
    src_ids = batch["src_ids"].to(device=device)
    src_mask = batch["src_mask"].to(device=device)

    generated_ids = model.inference(src_ids, src_mask, max_length)
    predictions = tgt_tokenizer.decode_batch(generated_ids.cpu().tolist())
    references = tgt_tokenizer.decode_batch(batch["tgt_ids"].tolist())

    return predictions, references


def create_trainer(
    model: Models,
    optimizer: Optimizer,
    criterion: CrossEntropyLoss,
    scheduler: LRScheduler,
    compute_loss: Callable[..., Tensor],
) -> Engine:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler = GradScaler(device.type, enabled=use_amp)
    model.to(device)

    def train_step(engine: Engine, batch: dict[str, Tensor]) -> dict[str, Any]:
        _ = engine
        model.train()
        optimizer.zero_grad()

        with autocast(device_type=device.type, enabled=use_amp):
            loss = compute_loss(model, batch, criterion=criterion, device=device)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        return {"loss": float(loss.detach())}

    trainer = Engine(train_step)
    RunningAverage(output_transform=lambda output: output["loss"]).attach(trainer, "loss")
    return trainer


def create_evaluator(
    model: Models,
    criterion: CrossEntropyLoss,
    compute_loss: Callable[..., Tensor],
    compute_predictions: Callable[..., tuple[list[str], list[str]]],
    compute_predictions_kwargs: dict,
) -> Engine:
    device = next(model.parameters()).device
    use_amp = device.type == "cuda"
    bleu = BLEU()

    @torch.inference_mode()
    def evaluate_step(engine: Engine, batch: dict[str, Tensor]):
        _ = engine
        model.eval()

        with autocast(device_type=device.type, enabled=use_amp):
            loss = compute_loss(model, batch, criterion=criterion, device=device)
            predictions, references = compute_predictions(
                model,
                batch,
                device,
                **compute_predictions_kwargs,
            )

        return {
            "loss": float(loss.detach()),
            "predictions": predictions,
            "references": references,
            "batch_size": len(references),
        }

    evaluator = Engine(evaluate_step)
    evaluator.state_dict_user_keys.append("predictions")
    evaluator.state_dict_user_keys.append("references")
    evaluator.state_dict_user_keys.append("loss_sum")
    evaluator.state_dict_user_keys.append("num_batches")

    @no_type_check
    @evaluator.on(Events.STARTED)
    def _reset(_):
        evaluator.state.predictions = []
        evaluator.state.references = []
        evaluator.state.loss_sum = 0.0
        evaluator.state.num_batches = 0

    @no_type_check
    @evaluator.on(Events.ITERATION_COMPLETED)
    def _accumulate(_):
        output = evaluator.state.output
        evaluator.state.predictions.extend(output["predictions"])
        evaluator.state.references.extend(output["references"])
        evaluator.state.loss_sum += output["loss"]
        evaluator.state.num_batches += 1

    @no_type_check
    @evaluator.on(Events.COMPLETED)
    def _finalize(_):
        loss_ = evaluator.state.loss_sum / max(evaluator.state.num_batches, 1)
        bleu_ = bleu.corpus_score(evaluator.state.predictions, [evaluator.state.references]).score
        evaluator.state.metrics["loss"] = float(loss_)
        evaluator.state.metrics["bleu"] = float(bleu_)
        print(f"Evaluation Loss: {float(loss_)}")
        print(f"Evaluation BLEU: {float(bleu_)}")

    return evaluator


def attach_tensorboard_logging(
    engine: Engine,
    log_dir: str | Path,
    tag: str,
    every: int | None = None,
) -> TensorboardLogger:
    logger = TensorboardLogger(log_dir=str(log_dir))

    logger.attach_output_handler(
        engine,
        event_name=(Events.COMPLETED if every is None else Events.ITERATION_COMPLETED(every=every)),
        tag=tag,
        metric_names="all",
    )

    return logger
