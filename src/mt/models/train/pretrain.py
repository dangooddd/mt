from argparse import ArgumentParser
from multiprocessing import freeze_support
from pathlib import Path
from typing import cast

import torch.optim as optim
from datasets import load_from_disk
from ignite.engine import Events
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset

from mt.tokenizers import BaseTokenizer

from ..load import load_from_config
from .utils.pretrain import (
    BilingualCollateFn,
    BilingualEvalCollateFn,
    CollateFn,
    EvalCollateFn,
    attach_tensorboard_logging,
    compute_bilingual_loss,
    compute_bilingual_predictions,
    compute_loss,
    compute_predictions,
    create_evaluator,
    create_trainer,
)


def main() -> None:
    parser = ArgumentParser("Train MT model from config dir")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--validation-split", type=str, default="validation")
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("data/checkpoints"))
    parser.add_argument("--runs-dir", type=Path, default=Path("data/runs"))
    parser.add_argument("--save-total-limit", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--src", type=str, default="ru")
    parser.add_argument("--tgt", type=str, default="en")
    parser.add_argument("--max-lr", type=float, default=0.001)
    parser.add_argument("--min-lr", type=float, default=0.00001)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--epoch-steps", type=int, default=2500)
    parser.add_argument("--warmup-steps", type=int, default=10000)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset_path)
    model, tokenizers, _ = load_from_config(args.model_dir)

    optimizer = optim.AdamW(model.parameters(), lr=args.max_lr)

    if isinstance(tokenizers, tuple):
        src_tokenizer, tgt_tokenizer = cast(tuple[BaseTokenizer, BaseTokenizer], tokenizers)
        criterion = CrossEntropyLoss(
            ignore_index=tgt_tokenizer.pad_token_id,
            label_smoothing=args.label_smoothing,
        )
        train_collate_fn = CollateFn(
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
        loss_fn = compute_loss
        predictions_fn = compute_predictions
        predictions_kwargs = {
            "tgt_tokenizer": tgt_tokenizer,
            "max_length": args.max_length,
        }

    else:
        tokenizer = tokenizers
        criterion = CrossEntropyLoss(
            ignore_index=tokenizer.pad_token_id,
            label_smoothing=args.label_smoothing,
        )
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
        loss_fn = compute_bilingual_loss
        predictions_fn = compute_bilingual_predictions
        predictions_kwargs = {
            "tokenizer": tokenizer,
            "max_length": args.max_length,
        }

    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=args.warmup_steps,
    )

    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(args.steps - args.warmup_steps),
        eta_min=args.min_lr,
    )

    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_steps],
    )

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
        criterion=criterion,
        scheduler=scheduler,
        compute_loss=loss_fn,
        max_grad_norm=args.max_grad_norm,
    )

    evaluator = create_evaluator(
        model=model,
        criterion=criterion,
        compute_loss=loss_fn,
        compute_predictions=predictions_fn,
        compute_predictions_kwargs=predictions_kwargs,
    )

    experiment = args.experiment or args.model_dir.name
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
        filename_prefix="training",
    )

    best_checkpoint_handler = Checkpoint(
        {"model": model},
        DiskSaver(checkpoints_dir / "best", create_dir=True, require_empty=False),
        n_saved=1,
        filename_prefix="best",
        score_name="bleu",
        score_function=lambda engine: engine.state.metrics["bleu"],
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
