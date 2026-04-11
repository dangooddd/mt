import csv
from argparse import ArgumentParser
from pathlib import Path
from typing import cast

import evaluate
import torch
import transformers
from datasets import Dataset, DatasetDict, load_from_disk
from sacrebleu.metrics import BLEU, CHRF
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from mt.tokenizers import BaseTokenizer

from . import Models, load_from_config
from .train import CollateFn, compute_predictions

BATCH_SIZE = 64
SCORE_BATCH_SIZE = 8
SCORE_MODEL_NAME = "Unbabel/wmt23-comet-da-xl"


def load_split(dataset_path: str, split: str) -> Dataset:
    dataset = load_from_disk(dataset_path)

    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise ValueError(f"Split '{split}' not found. Available splits: {list(dataset.keys())}")
        return dataset[split]

    return dataset


@torch.inference_mode()
def generate_predictions(
    dataset: Dataset,
    model: Models,
    src_tokenizer: BaseTokenizer,
    tgt_tokenizer: BaseTokenizer,
    batch_size: int,
    max_length: int,
    device: torch.device,
    src_column: str,
    tgt_column: str,
) -> list[str]:
    collate_fn = CollateFn(
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_column=src_column,
        tgt_column=tgt_column,
        max_src_length=max_length,
        max_tgt_length=max_length,
    )

    loader = DataLoader(
        cast(TorchDataset, dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model.to(device)
    model.eval()

    predictions: list[str] = []
    amp_enabled = device.type == "cuda"

    for batch in tqdm(loader, desc="Inference"):
        with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=torch.bfloat16):
            batch_predictions, _ = compute_predictions(
                model=model,
                batch=batch,
                device=device,
                tgt_tokenizer=tgt_tokenizer,
                max_length=max_length,
            )

        predictions.extend(batch_predictions)

    return predictions


def add_similarity_scores(
    batch, metric, model: str, batch_size: int, feature_name: str, src: str, pred: str, tgt: str
):
    sources = [x if x is not None else "" for x in batch[src]]
    predictions = [x if x is not None else "" for x in batch[pred]]
    references = [x if x is not None else "" for x in batch[tgt]]

    result = metric.compute(
        predictions=predictions,
        sources=sources,
        references=references,
        model=model,
        batch_size=batch_size,
        gpus=1 if torch.cuda.is_available() else 0,
        progress_bar=False,
    )

    return {feature_name: result["scores"]}


def save_metrics(path: Path, bleu: float, chrf: float, comet: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        writer.writerow(["bleu", bleu])
        writer.writerow(["chrf", chrf])
        writer.writerow(["comet", comet])


def main() -> None:
    parser = ArgumentParser("Evaluate mt.models checkpoint on a dataset split")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--src", type=str, default="ru")
    parser.add_argument("--tgt", type=str, default="en")
    parser.add_argument("--pred", type=str, default="prediction")
    args = parser.parse_args()
    transformers.logging.set_verbosity_error()

    dataset = load_split(args.dataset_path, args.split)
    model, src_tokenizer, tgt_tokenizer, runtime_config = load_from_config(
        args.model_dir, load_weights=True
    )

    max_length = int(runtime_config.get("max_length", 256))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictions = generate_predictions(
        dataset=dataset,
        model=model,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        batch_size=BATCH_SIZE,
        max_length=max_length,
        device=device,
        src_column=args.src,
        tgt_column=args.tgt,
    )
    dataset = dataset.add_column(args.pred, predictions)

    metric = evaluate.load("comet")

    dataset = dataset.map(
        add_similarity_scores,
        fn_kwargs={
            "metric": metric,
            "model": SCORE_MODEL_NAME,
            "batch_size": SCORE_BATCH_SIZE,
            "feature_name": f"{args.pred}_score",
            "src": args.src,
            "pred": args.pred,
            "tgt": args.tgt,
        },
        batched=True,
        batch_size=SCORE_BATCH_SIZE,
        load_from_cache_file=False,
    )

    references = dataset[args.tgt]
    references_text = [text or "" for text in references]
    bleu = BLEU().corpus_score(predictions, [references_text]).score
    chrf = CHRF().corpus_score(predictions, [references_text]).score
    comet = sum(cast(list[float], dataset[f"{args.pred}_score"])) / len(dataset)

    output_path = Path(args.output_path)
    dataset.save_to_disk(str(output_path))
    save_metrics(output_path / "metrics.csv", bleu=bleu, chrf=chrf, comet=comet)

    print(f"Saved scored split to: {output_path}")
    print(f"Saved metrics to: {output_path / 'metrics.csv'}")
    print(f"BLEU={bleu:.4f} CHRF={chrf:.4f} COMET={comet:.4f}")


if __name__ == "__main__":
    main()
