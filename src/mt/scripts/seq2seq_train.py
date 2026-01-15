from pathlib import Path
from typing import Optional, cast

import torch
import typer
from datasets import DatasetDict, load_from_disk
from lightning import Trainer
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger

from ..datasets import (
    get_dataset_preset,
    limit_samples,
    load_dataset_cached,
    shuffle_splits,
)
from ..datasets.dataloader import Seq2SeqDataModule
from ..models.luong import LuongSeq2Seq
from ..models.sutskever import Seq2SeqLightningModule
from ..tokenizers import Seq2SeqTokenizer


def main(
    src_tokenizer_path: str = "data/tokenizers/seq2seq_en.json",
    tgt_tokenizer_path: str = "data/tokenizers/seq2seq_ru.json",
    dataset_preset: str = "opus-100",
    from_disk: bool = False,
    src_field: str = "en",
    tgt_field: str = "ru",
    embedding_dim: int = 512,
    hidden_dim: int = 512,
    num_layers: int = 4,
    dropout: float = 0.3,
    batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 0.001,
    weight_decay: float = 0.005,
    warmup_interval: float = 0.02,
    max_epochs: int = 10,
    max_samples: Optional[int] = None,
    max_gen_length: int = 512,
    max_length: int = 512,
    clip_grad_norm: float = 1.0,
    checkpoint_dir: str = "data/checkpoints",
    experiment_name: str = "sutskever",
    log_dir: str = "data/logs",
    num_workers: int = 4,
    pin_memory: bool = True,
    save_top_k: int = 3,
    seed: int = 42,
    val_check_interval: float = 0.5,
    checkpoint_every_n_train_steps: Optional[int] = None,
    ckpt_path: Optional[str] = None,
):
    torch.set_float32_matmul_precision("medium")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    src_tokenizer = Seq2SeqTokenizer.from_file(src_tokenizer_path)
    tgt_tokenizer = Seq2SeqTokenizer.from_file(tgt_tokenizer_path)

    src_vocab_size = src_tokenizer.tokenizer.get_vocab_size()
    tgt_vocab_size = tgt_tokenizer.tokenizer.get_vocab_size()

    model = LuongSeq2Seq(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        src_pad_token_id=src_tokenizer.pad_token_id,
        tgt_pad_token_id=tgt_tokenizer.pad_token_id,
    )

    if from_disk:
        dataset = cast(DatasetDict, load_from_disk(dataset_preset))
    else:
        dataset = load_dataset_cached(get_dataset_preset(dataset_preset))

    if max_samples is not None:
        dataset = limit_samples(dataset, max_samples)
    dataset = shuffle_splits(dataset, seed)

    datamodule = Seq2SeqDataModule(
        dataset=dataset,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_field=src_field,
        tgt_field=tgt_field,
        batch_size=batch_size,
        max_length=max_length,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    checkpoint_path = Path(checkpoint_dir) / experiment_name
    log_path = Path(log_dir) / experiment_name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_path),
        filename="{epoch}-{step}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=save_top_k,
        save_last=True,
        every_n_train_steps=checkpoint_every_n_train_steps,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    tensorboard_logger = TensorBoardLogger(
        save_dir=str(log_path),
        name="tensorboard",
        default_hp_metric=False,
    )

    trainer = Trainer(
        max_epochs=max_epochs,
        logger=tensorboard_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=clip_grad_norm,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=gradient_accumulation_steps,
        log_every_n_steps=10,
        accelerator="auto",
        devices="auto",
        precision="32",
        val_check_interval=val_check_interval,
    )

    lightning_module = Seq2SeqLightningModule(
        model=model,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_interval=warmup_interval,
        compute_bleu=True,
        max_gen_length=max_gen_length,
    )

    print(
        "Source special tokens (pad, bos, eos, unk): ",
        src_tokenizer.pad_token_id,
        src_tokenizer.bos_token_id,
        src_tokenizer.eos_token_id,
        src_tokenizer.unk_token_id,
    )

    print(
        "Target special tokens (pad, bos, eos, unk): ",
        tgt_tokenizer.pad_token_id,
        tgt_tokenizer.bos_token_id,
        tgt_tokenizer.eos_token_id,
        tgt_tokenizer.unk_token_id,
    )

    trainer.fit(lightning_module, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    typer.run(main)
