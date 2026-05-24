from collections.abc import Callable
from pathlib import Path
from typing import Any, no_type_check

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import ProgressBar, global_step_from_engine
from ignite.handlers.tensorboard_logger import TensorboardLogger
from sacrebleu.metrics import BLEU, CHRF
from tokenizers import Encoding
from torch import Tensor
from torch.amp import autocast
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.types import Device

from mt.dataset.score import suppress_output
from mt.models.load import Model
from mt.models.modules import DecoderOnly, EncoderDecoder, EncoderDecoderBilingual
from mt.tokenizers import BaseTokenizer, BilingualBaseTokenizer, DecoderBaseTokenizer

COMET_MODEL_NAME = "Unbabel/wmt22-comet-da"
COMET_BATCH_SIZE = 100


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

    def __call__(self, batch: list[dict[str, str]]):
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


class EvalCollateFn:
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

    def __call__(self, batch: list[dict[str, str]]):
        src_texts = [item[self.src_column] for item in batch]
        targets = [item[self.tgt_column] for item in batch]

        src_encodings = self.src_tokenizer.encode_batch(src_texts)
        tgt_encodings = self.tgt_tokenizer.encode_batch(targets)

        src_ids = [encoding.ids for encoding in src_encodings]
        tgt_ids = [encoding.ids for encoding in tgt_encodings]
        src_mask = [encoding.attention_mask for encoding in src_encodings]

        return {
            "src_mask": torch.tensor(src_mask, dtype=torch.bool),
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "sources": src_texts,
            "targets": targets,
        }


class BilingualCollateFn:
    def __init__(
        self,
        tokenizer: BilingualBaseTokenizer,
        src_column: str = "ru",
        tgt_column: str = "en",
        max_length: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.src_column = src_column
        self.tgt_column = tgt_column
        self.tokenizer.enable_padding(direction="right")

        if max_length is not None:
            self.tokenizer.enable_truncation(max_length, direction="right")
        else:
            self.tokenizer.no_truncation()

    def __call__(self, batch: list[dict[str, str]]):
        src_texts = [item[self.src_column] for item in batch]
        tgt_texts = [item[self.tgt_column] for item in batch]
        sources = src_texts + tgt_texts
        targets = tgt_texts + src_texts

        src_encodings = self.tokenizer.encode_batch(
            sources,
            target_tokens=[f"<2{self.tgt_column}>"] * len(src_texts)
            + [f"<2{self.src_column}>"] * len(tgt_texts),
        )
        tgt_encodings = self.tokenizer.encode_batch(targets)

        src_ids = [encoding.ids for encoding in src_encodings]
        tgt_ids = [encoding.ids for encoding in tgt_encodings]
        src_mask = [encoding.attention_mask for encoding in src_encodings]

        return {
            "src_mask": torch.tensor(src_mask, dtype=torch.bool),
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "sources": sources,
            "targets": targets,
        }


class BilingualEvalCollateFn:
    def __init__(
        self,
        tokenizer: BilingualBaseTokenizer,
        src_column: str = "ru",
        tgt_column: str = "en",
        max_length: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.src_column = src_column
        self.tgt_column = tgt_column
        self.tokenizer.enable_padding(direction="right")

        if max_length is not None:
            self.tokenizer.enable_truncation(max_length, direction="right")
        else:
            self.tokenizer.no_truncation()

    def __call__(self, batch: list[dict[str, str]]):
        src_texts = [item[self.src_column] for item in batch]
        targets = [item[self.tgt_column] for item in batch]

        src_encodings = self.tokenizer.encode_batch(
            src_texts, target_tokens=f"<2{self.tgt_column}>"
        )
        tgt_encodings = self.tokenizer.encode_batch(targets)

        src_ids = [encoding.ids for encoding in src_encodings]
        tgt_ids = [encoding.ids for encoding in tgt_encodings]
        src_mask = [encoding.attention_mask for encoding in src_encodings]

        return {
            "src_mask": torch.tensor(src_mask, dtype=torch.bool),
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "sources": src_texts,
            "targets": targets,
        }


class DecoderCollateFn:
    def __init__(
        self,
        tokenizer: DecoderBaseTokenizer,
        src_column: str = "ru",
        tgt_column: str = "en",
        max_length: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.src_column = src_column
        self.tgt_column = tgt_column

        self.tokenizer.enable_padding(direction="right")

        if max_length is not None:
            self.tokenizer.enable_truncation(max_length, direction="right")
        else:
            self.tokenizer.no_truncation()

    def _pad(
        self,
        encodings: list[Encoding],
    ) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
        max_length = max(len(encoding.ids) for encoding in encodings)
        input_ids = []
        attention_mask = []
        type_ids = []

        for encoding in encodings:
            pad = max_length - len(encoding.ids)
            input_ids.append(encoding.ids + [self.tokenizer.pad_token_id] * pad)
            attention_mask.append(encoding.attention_mask + [0] * pad)
            type_ids.append(encoding.type_ids + [0] * pad)

        return input_ids, attention_mask, type_ids

    def __call__(self, batch: list[dict[str, str]]) -> dict[str, Tensor]:
        src_texts = [item[self.src_column] for item in batch]
        tgt_texts = [item[self.tgt_column] for item in batch]

        self.tokenizer.set_source_language(self.src_column)
        encodings = self.tokenizer.encode_batch(list(zip(src_texts, tgt_texts)))

        self.tokenizer.set_source_language(self.tgt_column)
        encodings += self.tokenizer.encode_batch(list(zip(tgt_texts, src_texts)))
        input_ids, attention_mask, type_ids = self._pad(encodings)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "type_ids": torch.tensor(type_ids, dtype=torch.long),
        }


class DecoderEvalCollateFn:
    def __init__(
        self,
        tokenizer: DecoderBaseTokenizer,
        src_column: str = "ru",
        tgt_column: str = "en",
        max_length: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.src_column = src_column
        self.tgt_column = tgt_column

        self.tokenizer.enable_padding(direction="right")

        if max_length is not None:
            self.tokenizer.enable_truncation(max_length, direction="right")
        else:
            self.tokenizer.no_truncation()

    def _pad(
        self,
        encodings: list[Encoding],
    ) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
        max_length = max(len(encoding.ids) for encoding in encodings)
        input_ids = []
        attention_mask = []
        type_ids = []

        for encoding in encodings:
            pad = max_length - len(encoding.ids)
            input_ids.append(encoding.ids + [self.tokenizer.pad_token_id] * pad)
            attention_mask.append(encoding.attention_mask + [0] * pad)
            type_ids.append(encoding.type_ids + [0] * pad)

        return input_ids, attention_mask, type_ids

    def __call__(self, batch: list[dict[str, str]]):
        src_texts = [item[self.src_column] for item in batch]
        targets = [item[self.tgt_column] for item in batch]

        self.tokenizer.set_source_language(self.src_column)
        inference_encodings = self.tokenizer.encode_batch(src_texts)
        inference_input_ids, inference_attention_mask, inference_type_ids = self._pad(
            inference_encodings
        )

        encodings = self.tokenizer.encode_batch(list(zip(src_texts, targets)))
        input_ids, attention_mask, type_ids = self._pad(encodings)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "type_ids": torch.tensor(type_ids, dtype=torch.long),
            "inference_input_ids": torch.tensor(inference_input_ids, dtype=torch.long),
            "inference_attention_mask": torch.tensor(inference_attention_mask, dtype=torch.bool),
            "inference_type_ids": torch.tensor(inference_type_ids, dtype=torch.long),
            "sources": src_texts,
            "targets": targets,
        }


def split_decay_params(model: nn.Module):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        name = name.lower()
        if name.endswith(".bias") or ("norm" in name) or ("embedding" in name):
            no_decay.append(param)
        else:
            decay.append(param)

    return decay, no_decay


def compute_comet(
    scorer,
    sources: list[str | None],
    predictions: list[str | None],
    references: list[str | None],
    batch_size: int = COMET_BATCH_SIZE,
) -> float:
    samples = [
        {"src": source or "", "mt": prediction or "", "ref": reference or ""}
        for source, prediction, reference in zip(
            sources,
            predictions,
            references,
            strict=True,
        )
    ]

    if not samples:
        return 0.0

    use_cuda = torch.cuda.is_available()
    with torch.inference_mode(), suppress_output():
        result = scorer.predict(
            samples,
            batch_size=batch_size,
            gpus=1 if use_cuda else 0,
            progress_bar=False,
            accelerator="auto" if use_cuda else "cpu",
            num_workers=0,
        )

    scores = [float(score) for score in result.scores]
    return sum(scores) / len(scores)


def compute_loss(
    model: EncoderDecoder,
    batch: dict[str, Any],
    device: Device,
    label_smoothing: float = 0.0,
) -> tuple[Tensor, dict[str, Any]]:
    src_ids = batch["src_ids"].to(device=device)
    tgt_ids = batch["tgt_ids"].to(device=device)
    src_mask = batch["src_mask"].to(device=device)

    logits = model.forward(
        src_ids,
        tgt_ids[:, :-1],
        src_mask,
    )

    targets = tgt_ids[:, 1:]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=model.tgt_pad_token_id,
        label_smoothing=label_smoothing,
    )

    return loss, {}


def compute_bilingual_loss(
    model: EncoderDecoderBilingual,
    batch: dict[str, Any],
    device: Device,
    label_smoothing: float = 0.0,
) -> tuple[Tensor, dict[str, Any]]:
    src_ids = batch["src_ids"].to(device=device)
    tgt_ids = batch["tgt_ids"].to(device=device)
    src_mask = batch["src_mask"].to(device=device)

    logits = model.forward(
        src_ids,
        tgt_ids[:, :-1],
        src_mask,
    )

    targets = tgt_ids[:, 1:]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=model.pad_token_id,
        label_smoothing=label_smoothing,
    )

    return loss, {}


def compute_decoder_loss(
    model: DecoderOnly,
    batch: dict[str, Any],
    device: Device,
    label_smoothing: float = 0.0,
) -> tuple[Tensor, dict[str, Any]]:
    input_ids = batch["input_ids"].to(device=device)
    attention_mask = batch["attention_mask"].to(device=device)
    type_ids = batch["type_ids"].to(device=device)

    logits = model.forward(
        input_ids[:, :-1],
        attention_mask[:, :-1],
        type_ids[:, :-1],
    )

    targets = input_ids[:, 1:]
    loss_mask = attention_mask[:, 1:] & type_ids[:, 1:].eq(1)
    if not loss_mask.any():
        raise ValueError("Decoder loss mask is empty: batch contains no target tokens")
    loss = F.cross_entropy(
        logits[loss_mask],
        targets[loss_mask],
        label_smoothing=label_smoothing,
    )
    return loss, {}


def compute_predictions(
    model: EncoderDecoder,
    batch: dict[str, Any],
    device: Device,
    tgt_tokenizer: BaseTokenizer,
    max_length: int = 1024,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> tuple[list[str], list[str]]:
    src_ids = batch["src_ids"].to(device=device)
    src_mask = batch["src_mask"].to(device=device)

    generated_ids = model.inference(
        src_ids,
        src_mask,
        max_length,
        temperature=temperature,
        top_p=top_p,
    )
    predictions = tgt_tokenizer.decode_batch(generated_ids.cpu().tolist())
    references = batch["targets"]
    return predictions, references


def compute_decoder_predictions(
    model: DecoderOnly,
    batch: dict[str, Any],
    device: Device,
    tokenizer: DecoderBaseTokenizer,
    max_length: int = 1024,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> tuple[list[str], list[str]]:
    input_ids = batch.get("inference_input_ids", batch["input_ids"]).to(device=device)
    attention_mask = batch.get("inference_attention_mask", batch["attention_mask"]).to(
        device=device
    )
    type_ids = batch.get("inference_type_ids", batch["type_ids"]).to(device=device)

    generated_ids = model.inference(
        input_ids,
        attention_mask,
        type_ids,
        max_length,
        temperature=temperature,
        top_p=top_p,
    )
    predictions = tokenizer.decode_batch(generated_ids.cpu().tolist())
    references = batch["targets"]

    return predictions, references


def create_trainer(
    model: Model,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    compute_loss: Callable[..., tuple[Tensor, dict[str, Any]]],
    compute_loss_kwargs: dict,
    max_grad_norm: float = 1.0,
    amp: bool = True,
) -> Engine:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = amp and device.type == "cuda"
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
        scheduler.step()

        return {
            "loss": loss.detach(),
            "lr": optimizer.param_groups[0]["lr"],
            "grad_norm": grad_norm,
            **metrics,
        }

    trainer = Engine(train_step)

    @no_type_check
    @trainer.on(Events.ITERATION_COMPLETED)
    def _metrics(engine: Engine):
        engine.state.metrics.update(engine.state.output)

    ProgressBar().attach(trainer, metric_names=["loss", "lr", "grad_norm"])
    return trainer


def create_evaluator(
    model: Model,
    compute_loss: Callable[..., tuple[Tensor, dict[str, Any]]],
    compute_loss_kwargs: dict,
    compute_predictions: Callable[..., tuple[list[str], list[str]]],
    compute_predictions_kwargs: dict,
    amp: bool = True,
) -> Engine:
    device = next(model.parameters()).device
    use_amp = amp and device.type == "cuda"
    bleu = BLEU()
    chrf = CHRF()

    with suppress_output():
        from comet import download_model, load_from_checkpoint

        comet_scorer = load_from_checkpoint(download_model(COMET_MODEL_NAME))
        comet_scorer.eval()

    @torch.inference_mode()
    def evaluate_step(engine: Engine, batch: dict[str, Tensor]):
        _ = engine
        model.eval()

        with autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
            loss, _ = compute_loss(model, batch, device=device, **compute_loss_kwargs)
            predictions, references = compute_predictions(
                model,
                batch,
                device,
                **compute_predictions_kwargs,
            )

        return {
            "loss": float(loss.detach()),
            "sources": batch["sources"],
            "predictions": predictions,
            "references": references,
            "batch_size": len(references),
        }

    evaluator = Engine(evaluate_step)
    pbar = ProgressBar(persist=True)

    evaluator.state_dict_user_keys.append("sources")
    evaluator.state_dict_user_keys.append("predictions")
    evaluator.state_dict_user_keys.append("references")
    evaluator.state_dict_user_keys.append("loss_sum")
    evaluator.state_dict_user_keys.append("num_batches")

    @no_type_check
    @evaluator.on(Events.STARTED)
    def _reset(_):
        evaluator.state.sources = []
        evaluator.state.predictions = []
        evaluator.state.references = []
        evaluator.state.loss_sum = 0.0
        evaluator.state.num_batches = 0

    @no_type_check
    @evaluator.on(Events.ITERATION_COMPLETED)
    def _accumulate(_):
        output = evaluator.state.output
        evaluator.state.sources.extend(output["sources"])
        evaluator.state.predictions.extend(output["predictions"])
        evaluator.state.references.extend(output["references"])
        evaluator.state.loss_sum += output["loss"]
        evaluator.state.num_batches += 1

    @no_type_check
    @evaluator.on(Events.COMPLETED)
    def _finalize(_):
        loss_ = evaluator.state.loss_sum / max(evaluator.state.num_batches, 1)
        bleu_ = bleu.corpus_score(evaluator.state.predictions, [evaluator.state.references]).score
        chrf_ = chrf.corpus_score(evaluator.state.predictions, [evaluator.state.references]).score
        comet_ = compute_comet(
            comet_scorer,
            evaluator.state.sources,
            evaluator.state.predictions,
            evaluator.state.references,
        )

        evaluator.state.metrics["loss"] = float(loss_)
        evaluator.state.metrics["bleu"] = float(bleu_)
        evaluator.state.metrics["chrf"] = float(chrf_)
        evaluator.state.metrics["comet"] = float(comet_)

        if pbar.pbar is not None:
            pbar.pbar.set_postfix(
                {
                    "loss": f"{loss_:.6f}",
                    "bleu": f"{bleu_:.4f}",
                    "chrf": f"{chrf_:.4f}",
                    "comet": f"{comet_:.4f}",
                }
            )
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
