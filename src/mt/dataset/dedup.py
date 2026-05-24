from argparse import ArgumentParser
from pathlib import Path
from typing import cast

from datasets import Dataset, DatasetDict, load_from_disk
from rapidfuzz import fuzz, process
from tqdm import tqdm


def normalize(text: str | None) -> str:
    return (text or "").strip().lower()


def get_bucket(text: str, bucket_prefix: int, bucket_length: int) -> tuple[str, int]:
    return text[:bucket_prefix], len(text) // bucket_length


def deduplicate(
    dataset: Dataset,
    column: str,
    threshold: float,
    bucket_prefix: int,
    bucket_length: int,
) -> Dataset:
    choices: dict[tuple[str, int], list[str]] = {}
    keep_indices: list[int] = []
    texts = [normalize(text) for text in dataset[column]]

    for index, text in enumerate(tqdm(texts, desc="Deduplication")):
        bucket = get_bucket(
            text=text,
            bucket_prefix=bucket_prefix,
            bucket_length=bucket_length,
        )
        bucket_choices = choices.setdefault(bucket, [])
        match = process.extractOne(
            text,
            bucket_choices,
            scorer=fuzz.ratio,
            score_cutoff=threshold,
        )

        if match is None:
            bucket_choices.append(text)
            keep_indices.append(index)

    return dataset.select(keep_indices)


def main():
    parser = ArgumentParser("Fuzzy deduplicate dataset by column")
    parser.add_argument("--dataset-path", type=str)
    parser.add_argument("--column", type=str)
    parser.add_argument("--threshold", type=float, default=95.0)
    parser.add_argument("--bucket-prefix", type=int, default=2)
    parser.add_argument("--bucket-length", type=int, default=30)
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset_path)

    if isinstance(dataset, DatasetDict):
        deduplicated = DatasetDict()
        for split, split_dataset in dataset.items():
            deduplicated[split] = deduplicate(
                dataset=split_dataset,
                column=args.column,
                threshold=args.threshold,
                bucket_prefix=args.bucket_prefix,
                bucket_length=args.bucket_length,
            )
    else:
        deduplicated = deduplicate(
            dataset=cast(Dataset, dataset),
            column=args.column,
            threshold=args.threshold,
            bucket_prefix=args.bucket_prefix,
            bucket_length=args.bucket_length,
        )

    dataset_path = Path(args.dataset_path)
    deduplicated.save_to_disk(dataset_path.parent / f"{dataset_path.name}-dedup")


if __name__ == "__main__":
    main()
