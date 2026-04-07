from argparse import ArgumentParser
from pathlib import Path

from datasets import concatenate_datasets, load_from_disk


def main():
    parser = ArgumentParser("Merge augment chunks")
    parser.add_argument("--dataset-path", type=Path)
    args = parser.parse_args()

    chunks = []
    for chunk_path in args.dataset_path.iterdir():
        chunks.append(load_from_disk(str(chunk_path)))

    merged = concatenate_datasets(chunks)
    merged.save_to_disk(args.dataset_path.with_suffix(".merged"))


if __name__ == "__main__":
    main()
