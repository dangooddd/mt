from argparse import ArgumentParser
from pathlib import Path

from datasets import concatenate_datasets, load_from_disk


def main():
    parser = ArgumentParser("Merge augment chunks")
    parser.add_argument("--dataset-path", type=Path)
    args = parser.parse_args()

    dataset_path = args.dataset_path
    chunks = []
    for chunk_path in dataset_path.iterdir():
        chunks.append(load_from_disk(str(chunk_path)))

    merged = concatenate_datasets(chunks)
    merged.save_to_disk(dataset_path.parent / f"{dataset_path.name}-merged")


if __name__ == "__main__":
    main()
