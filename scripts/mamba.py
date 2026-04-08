from multiprocessing import freeze_support
from typing import cast

import torch.optim as optim
from datasets import load_from_disk
from ignite.engine import Events
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset

from mt.models.mamba import MambaSeq2Seq
from mt.models.train import (
    attach_tensorboard_logging,
    build_collate_fn,
    compute_loss,
    compute_predictions,
    create_evaluator,
    create_trainer,
)
from mt.tokenizers import UnigramTokenizer


def main():
    MAX_LR = 0.001
    MIN_LR = 0.00001
    MAX_GRAD_NORM = 1.0
    BATCH_SIZE = 100
    EPOCH_STEPS = 2500
    WARMUP_STEPS = 5000
    STEPS = 50000
    MAX_LENGTH = 200
    EXPERIMENT = "mamba-v2"

    dataset = load_from_disk("data/datasets/opus-100-final")
    tokenizer_ru = UnigramTokenizer.from_file("data/tokenizers/ru-unigram-24000.json")
    tokenizer_en = UnigramTokenizer.from_file("data/tokenizers/en-unigram-24000.json")

    model = MambaSeq2Seq(
        src_vocab_size=tokenizer_ru.get_vocab_size(),
        tgt_vocab_size=tokenizer_en.get_vocab_size(),
        src_pad_token_id=tokenizer_ru.pad_token_id,
        tgt_pad_token_id=tokenizer_en.pad_token_id,
        tgt_bos_token_id=tokenizer_en.bos_token_id,
        tgt_eos_token_id=tokenizer_en.eos_token_id,
        embedding_dim=512,
        hidden_dim=512,
        num_layers=5,
        num_heads=8,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank=None,
    )

    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR)
    criterion = CrossEntropyLoss(ignore_index=tokenizer_en.pad_token_id, label_smoothing=0.1)

    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=WARMUP_STEPS,
    )

    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(STEPS - WARMUP_STEPS),
        eta_min=MIN_LR,
    )

    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_STEPS],
    )

    collate_fn = build_collate_fn(
        src_tokenizer=tokenizer_ru,
        tgt_tokenizer=tokenizer_en,
        max_src_length=MAX_LENGTH,
        max_tgt_length=MAX_LENGTH,
        src_column="ru",
        tgt_column="en",
    )

    train_loader = DataLoader(
        cast(Dataset, dataset["train"]),
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        num_workers=0,
        shuffle=True,
    )

    evaluation_loader = DataLoader(
        cast(Dataset, dataset["validation"]),
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        num_workers=0,
    )

    trainer = create_trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        compute_loss=compute_loss,
        max_grad_norm=MAX_GRAD_NORM,
    )

    evaluator = create_evaluator(
        model=model,
        criterion=criterion,
        compute_loss=compute_loss,
        compute_predictions=compute_predictions,
        compute_predictions_kwargs={"tgt_tokenizer": tokenizer_en, "max_length": MAX_LENGTH},
    )

    checkpoint_handler = Checkpoint(
        {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "trainer": trainer,
        },
        DiskSaver(f"data/checkpoints/{EXPERIMENT}", create_dir=True, require_empty=False),
        n_saved=10,
        filename_prefix="training",
    )

    best_checkpoint_handler = Checkpoint(
        {"model": model},
        DiskSaver(f"data/checkpoints/{EXPERIMENT}/best", create_dir=True, require_empty=False),
        n_saved=1,
        filename_prefix="best",
        score_name="bleu",
        score_function=lambda engine: engine.state.metrics["bleu"],
    )

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=EPOCH_STEPS),
        checkpoint_handler,
    )

    evaluator.add_event_handler(Events.COMPLETED, best_checkpoint_handler)

    train_tb_logger = attach_tensorboard_logging(
        trainer,
        f"data/runs/{EXPERIMENT}",
        tag="train",
        every=100,
    )

    evaluation_tb_logger = attach_tensorboard_logging(
        evaluator,
        f"data/runs/{EXPERIMENT}",
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
        max_epochs=STEPS // EPOCH_STEPS,
        epoch_length=EPOCH_STEPS,
    )


if __name__ == "__main__":
    freeze_support()
    main()
