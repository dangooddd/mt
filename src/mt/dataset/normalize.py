from argparse import ArgumentParser
from pathlib import Path

from datasets import load_from_disk


def main():
    parser = ArgumentParser("Normalize opus-100 style dataset to flatstructure")
    parser.add_argument("--dataset-path", type=str)
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset_path)
    dataset = dataset.flatten()
    dataset = dataset.rename_columns(
        {
            "translation.ru": "ru",
            "translation.en": "en",
        }
    )

    print("New schema:", dataset, sep="\n")
    dataset_path = Path(args.dataset_path)
    dataset.save_to_disk(dataset_path.parent / f"{dataset_path.name}-normalized")


if __name__ == "__main__":
    main()
