from argparse import ArgumentParser
from pathlib import Path

from datasets import DatasetDict, load_from_disk


def keep_aug(example):
    if example["aug"] and len(example["aug_ru"]) > 0 and len(example["aug_en"]) > 0:
        return {
            "ru": example["aug_ru"],
            "en": example["aug_en"],
            "score": example.get("aug_score"),
            "source": example.get("source"),
            "aug": True,
        }

    return {
        "ru": example["ru"],
        "en": example["en"],
        "score": example.get("score"),
        "source": example.get("source"),
        "aug": False,
    }


def main():
    parser = ArgumentParser("Choose examples with valid augmentation")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--split", type=str, nargs="+", default=[])
    args = parser.parse_args()

    dataset_path = args.dataset_path
    dataset = load_from_disk(str(dataset_path))

    if isinstance(dataset, DatasetDict):
        splits = args.split

        if not splits:
            splits = list(dataset.keys())

        for split in splits:
            dataset[split] = dataset[split].map(
                keep_aug,
                remove_columns=dataset[split].column_names,
            )
    else:
        dataset = dataset.map(keep_aug, remove_columns=dataset.column_names)

    dataset.save_to_disk(dataset_path.parent / f"{dataset_path.name}-choosed")


if __name__ == "__main__":
    main()
