from argparse import ArgumentParser
from pathlib import Path

import torch
import transformers
from comet import download_model, load_from_checkpoint
from datasets import Dataset, DatasetDict, load_from_disk

MODEL_NAME = "Unbabel/wmt22-cometkiwi-da"
BATCH_SIZE = 100


def add_scores(
    dataset: Dataset,
    scorer,
    batch_size: int,
    feature_name: str,
    src: str,
    tgt: str,
) -> tuple[Dataset, float]:
    samples = [
        {
            "src": source if source is not None else "",
            "mt": prediction if prediction is not None else "",
        }
        for source, prediction in zip(dataset[src], dataset[tgt], strict=True)
    ]

    result = scorer.predict(
        samples,
        batch_size=batch_size,
        gpus=1 if torch.cuda.is_available() else 0,
        progress_bar=True,
        accelerator="cpu" if not torch.cuda.is_available() else "auto",
        num_workers=0,
    )

    return dataset.add_column(feature_name, result.scores), result.system_score


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
    scorer = load_from_checkpoint(download_model(args.model))
    scorer.eval()

    if isinstance(dataset, DatasetDict):
        scored = DatasetDict()
        for split, split_dataset in dataset.items():
            split_scored, split_score_mean = add_scores(
                dataset=split_dataset,
                scorer=scorer,
                batch_size=args.batch_size,
                feature_name=args.feature_name,
                src=args.src,
                tgt=args.tgt,
            )
            scored[split] = split_scored
            print(f"{split}: COMET={split_score_mean:.4f}")
    else:
        scored, score_mean = add_scores(
            dataset=dataset,
            scorer=scorer,
            batch_size=args.batch_size,
            feature_name=args.feature_name,
            src=args.src,
            tgt=args.tgt,
        )
        print(f"COMET={score_mean:.4f}")

    dataset_path = Path(args.dataset_path)
    output_path = dataset_path.parent / f"{dataset_path.name}-scored"
    scored.save_to_disk(output_path)


if __name__ == "__main__":
    main()
