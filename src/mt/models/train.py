from collections.abc import Callable
from pathlib import Path
from typing import Any, no_type_check

import torch
from ignite.engine import Engine, Events
from ignite.handlers import ProgressBar, global_step_from_engine
from ignite.handlers.tensorboard_logger import TensorboardLogger
from ignite.metrics import Metric
from ignite.metrics.metric import BatchWise
from sacrebleu.metrics import BLEU
from torch import Tensor
from torch.amp import autocast
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.types import Device

from ..tokenizers import BaseTokenizer
from .luong import LuongSeq2Seq
from .mamba import MambaSeq2Seq
from .ssm import S4Seq2Seq

Models = LuongSeq2Seq | S4Seq2Seq | MambaSeq2Seq


class OutputMetric(Metric):
    required_output_keys = None

    def attach(self, engine, name, usage=BatchWise.usage_name):
        return super().attach(engine, name, usage=usage)

    def reset(self):
        self._value = None

    def update(self, output):
        self._value = output

    def compute(self):
        return self._value


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
            loss = compute_loss(model, batch, criterion=criterion, device=device)

        loss.backward()
        grad_norm = clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        return {
            "loss": loss.detach(),
            "lr": optimizer.param_groups[0]["lr"],
            "grad_norm": grad_norm,
        }

    trainer = Engine(train_step)
    OutputMetric(output_transform=lambda o: o["grad_norm"]).attach(trainer, "grad_norm")
    OutputMetric(output_transform=lambda o: o["loss"]).attach(trainer, "loss")
    OutputMetric(output_transform=lambda o: o["lr"]).attach(trainer, "lr")
    ProgressBar().attach(trainer, metric_names="all")
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
    pbar = ProgressBar(persist=True)

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

        if pbar.pbar is not None:
            pbar.pbar.set_postfix({"loss": f"{loss_:.6f}", "bleu": f"{bleu_:.4f}"})
            pbar.pbar.refresh()

    pbar.attach(evaluator, metric_names="all", closing_event_name=Events.COMPLETED)
    return evaluator


def attach_tensorboard_logging(
    engine: Engine,
    log_dir: str | Path,
    tag: str,
    every: int | None = None,
    global_step_transform: Callable | None = None,
) -> TensorboardLogger:
    logger = TensorboardLogger(log_dir=str(log_dir))

    if global_step_transform is None:
        global_step_transform = global_step_from_engine(engine)

    logger.attach_output_handler(
        engine,
        event_name=(Events.COMPLETED if every is None else Events.ITERATION_COMPLETED(every=every)),
        tag=tag,
        metric_names="all",
        global_step_transform=global_step_transform,
    )

    return logger
