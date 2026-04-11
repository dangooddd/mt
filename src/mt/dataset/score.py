from argparse import ArgumentParser
from pathlib import Path

import evaluate
import torch
import transformers
from datasets import load_from_disk

MODEL_NAME = "Unbabel/wmt23-cometkiwi-da-xl"
BATCH_SIZE = 8


def add_similarity_scores(
    batch, metric, model: str, batch_size: int, feature_name: str, src: str, tgt: str
):
    sources = [x if x is not None else "" for x in batch[src]]
    predictions = [x if x is not None else "" for x in batch[tgt]]

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
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--dataset-path", type=str)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--feature-name", type=str, default="score")
    parser.add_argument("--src", type=str, default="ru")
    parser.add_argument("--tgt", type=str, default="en")
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
            "src": args.src,
            "tgt": args.tgt,
        },
        batched=True,
        batch_size=args.batch_size,
        load_from_cache_file=False,
    )

    score_mean = sum(scored[args.feature_name]) / len(scored)

    dataset_path = Path(args.dataset_path)
    output_path = dataset_path.parent / f"{dataset_path.name}-scored"
    scored.save_to_disk(output_path)

    print(f"Saved scored dataset to: {output_path}")
    print(f"COMET={score_mean:.4f}")


if __name__ == "__main__":
    main()
