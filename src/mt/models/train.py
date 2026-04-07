from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from ignite.engine import Engine, Events
from ignite.handlers.tensorboard_logger import TensorboardLogger, global_step_from_engine
from ignite.metrics import RunningAverage
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer
from torch.types import Device

from ..tokenizers import BaseTokenizer


def build_collate_fn(
    src_tokenizer: BaseTokenizer,
    tgt_tokenizer: BaseTokenizer,
    src_column: str = "ru",
    tgt_column: str = "en",
    max_src_length: int | None = None,
    max_tgt_length: int | None = None,
) -> Callable[[list[dict[str, str]]], dict[str, Tensor]]:
    src_tokenizer.enable_padding(direction="right")
    tgt_tokenizer.enable_padding(direction="right")

    def collate_fn(batch: list[dict[str, str]]) -> dict[str, Tensor]:
        src_texts = [item[src_column] for item in batch]
        tgt_texts = [item[tgt_column] for item in batch]

        src_encodings = src_tokenizer.encode_batch(src_texts)
        tgt_encodings = tgt_tokenizer.encode_batch(tgt_texts)

        src_ids = [encoding.ids for encoding in src_encodings]
        tgt_ids = [encoding.ids for encoding in tgt_encodings]
        src_mask = [encoding.attention_mask for encoding in src_encodings]

        if max_src_length is not None:
            src_ids = [ids[:max_src_length] for ids in src_ids]
            src_mask = [mask[:max_src_length] for mask in src_mask]

        if max_tgt_length is not None:
            tgt_ids = [ids[:max_tgt_length] for ids in tgt_ids]

        return {
            "src_mask": torch.tensor(src_mask, dtype=torch.bool),
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
        }

    return collate_fn


def compute_loss(
    model: Module,
    batch: dict[str, Tensor],
    criterion: CrossEntropyLoss,
    device: Device,
) -> tuple[Tensor, dict[str, Any]]:
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
    return loss, {"loss": float(loss.detach())}


def create_trainer(
    model: Module,
    optimizer: Optimizer,
    criterion: CrossEntropyLoss,
    compute_loss: Callable,
) -> Engine:
    device = torch.device("cuda")
    scaler = GradScaler(device.type)
    model.to(device)

    def train_step(engine: Engine, batch: dict[str, Tensor]) -> dict[str, Any]:
        _ = engine
        model.train()
        optimizer.zero_grad()

        with autocast(device_type=device.type):
            loss, output = compute_loss(model, batch, criterion=criterion, device=device)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        return output

    return Engine(train_step)


def attach_running_average(
    trainer,
    metric_name: str = "loss",
    output_name: str = "loss",
):
    RunningAverage(output_transform=lambda output: output[output_name]).attach(trainer, metric_name)


def attach_tensorboard_loss_logging(
    trainer,
    log_dir: str | Path,
    tag: str = "train",
    every: int = 50,
):
    logger = TensorboardLogger(log_dir=str(log_dir))
    logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=every),
        tag=tag,
        output_transform=lambda output: {"loss": output["loss"]},
        global_step_transform=global_step_from_engine(trainer),
    )
