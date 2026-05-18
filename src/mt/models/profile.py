from argparse import ArgumentParser
from pathlib import Path
from typing import cast

import torch
from datasets import DatasetDict, load_from_disk
from torch.profiler import ProfilerActivity, profile, record_function, schedule
from torch.utils.data import DataLoader, Dataset

from mt.tokenizers import BaseTokenizer

from .load import load_from_config
from .train.utils.pretrain import CollateFn, DecoderCollateFn


def main() -> None:
    parser = ArgumentParser("Profile mt.models with torch.profiler")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--src", type=str, default="ru")
    parser.add_argument("--tgt", type=str, default="en")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--active-steps", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--row-limit", type=int, default=30)
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--with-flops", action="store_true")
    args = parser.parse_args()

    model, tokenizers, config = load_from_config(args.model_dir)
    max_length = config["max_length"] if "max_length" in config else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sort_by = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    use_amp = device.type == "cuda"

    model.to(device)
    model.eval()

    if isinstance(tokenizers, tuple):
        src_tokenizer, tgt_tokenizer = cast(tuple[BaseTokenizer, BaseTokenizer], tokenizers)
        collate_fn = CollateFn(
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            src_column=args.src,
            tgt_column=args.tgt,
            max_src_length=max_length,
            max_tgt_length=max_length,
        )
        decoder = False
    else:
        collate_fn = DecoderCollateFn(
            tokenizer=tokenizers,
            src_column=args.src,
            tgt_column=args.tgt,
            max_length=max_length,
        )
        decoder = True

    dataset = load_from_disk(args.dataset_path)
    data = dataset[args.split] if isinstance(dataset, DatasetDict) else dataset
    loader = DataLoader(
        cast(Dataset, data),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    on_trace_ready = None
    if args.trace_dir is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
        on_trace_ready = torch.profiler.tensorboard_trace_handler(str(args.trace_dir))

    def run_step(batch):
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device.type,
                enabled=use_amp,
                dtype=torch.bfloat16,
            ),
        ):
            with record_function("model_forward"):
                if decoder:
                    input_ids = batch["input_ids"].to(device=device)
                    attention_mask = batch["attention_mask"].to(device=device)
                    type_ids = batch["type_ids"].to(device=device)
                    _ = model.forward(
                        input_ids[:, :-1],
                        attention_mask[:, :-1],
                        type_ids[:, :-1],
                    )
                else:
                    src_ids = batch["src_ids"].to(device=device)
                    src_mask = batch["src_mask"].to(device=device)
                    tgt_ids = batch["tgt_ids"].to(device=device)
                    _ = model.forward(src_ids, tgt_ids[:, :-1], src_mask)

    with profile(
        activities=activities,
        schedule=schedule(wait=0, warmup=args.warmup_steps, active=args.active_steps, repeat=1),
        on_trace_ready=on_trace_ready,
        record_shapes=True,
        profile_memory=True,
        with_stack=args.with_stack,
        with_flops=args.with_flops,
    ) as prof:
        for _, batch in zip(range(args.warmup_steps + args.active_steps), loader):
            run_step(batch)
            prof.step()

    print(prof.key_averages().table(sort_by=sort_by, row_limit=args.row_limit))


if __name__ == "__main__":
    main()
