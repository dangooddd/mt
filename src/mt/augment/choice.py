from argparse import ArgumentParser
from pathlib import Path

from datasets import DatasetDict, load_from_disk


def keep_aug(example):
    if example["aug_message"] == "" and len(example["aug_ru"]) > 0 and len(example["aug_en"]) > 0:
        return {"ru": example["aug_ru"], "en": example["aug_en"], "source": "augmentation"}

    return {"ru": example["ru"], "en": example["en"], "source": "original"}


def main():
    parser = ArgumentParser("Choose examples with valid augmentation")
    parser.add_argument("--dataset-path", type=str)
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset_path)

    if isinstance(dataset, DatasetDict):
        for split in list(dataset.keys()):
            dataset[split] = dataset[split].map(
                keep_aug,
                remove_columns=dataset[split].column_names,
            )
    else:
        dataset = dataset.map(keep_aug, remove_columns=dataset.column_names)

    dataset.save_to_disk(Path(args.dataset_path).with_suffix(".choosed"))


if __name__ == "__main__":
    main()
