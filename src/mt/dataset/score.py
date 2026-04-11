from argparse import ArgumentParser
from pathlib import Path

import evaluate
import torch
import transformers
from datasets import load_from_disk

DEFAULT_MODEL = "Unbabel/wmt23-cometkiwi-da-xl"
DEFAULT_BATCH_SIZE = 8


def add_similarity_scores(
    batch, metric, model: str, batch_size: int, feature_name: str, lhs: str, rhs: str
):
    sources = [x if x is not None else "" for x in batch[lhs]]
    predictions = [x if x is not None else "" for x in batch[rhs]]

    result = metric.compute(
        predictions=predictions,
        sources=sources,
        model=model,
        batch_size=batch_size,
        gpus=1 if torch.cuda.is_available() else 0,
        progress_bar=False,
    )

    return {feature_name: result["scores"]}


def main():
    parser = ArgumentParser("Validate augmented dataset using COMET")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--dataset-path", type=str)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--feature-name", type=str, default="score")
    parser.add_argument("--lhs", type=str, default="ru")
    parser.add_argument("--rhs", type=str, default="en")
    args = parser.parse_args()

    transformers.logging.set_verbosity_error()
    dataset = load_from_disk(args.dataset_path)
    metric = evaluate.load("comet")

    scored = dataset.map(
        add_similarity_scores,
        fn_kwargs={
            "metric": metric,
            "model": args.model,
            "batch_size": args.batch_size,
            "feature_name": args.feature_name,
            "lhs": args.lhs,
            "rhs": args.rhs,
        },
        batched=True,
        batch_size=args.batch_size,
        load_from_cache_file=False,
    )

    dataset_path = Path(args.dataset_path)
    scored.save_to_disk(dataset_path.parent / f"{dataset_path.name}-scored")


if __name__ == "__main__":
    main()
