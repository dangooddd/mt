from argparse import ArgumentParser
from typing import cast

from datasets import Dataset, DatasetDict, load_from_disk


def main():
    parser = ArgumentParser("Merge regular datasets in DatasetDict")
    parser.add_argument("--train", type=str, default=None)
    parser.add_argument("--test", type=str, default=None)
    parser.add_argument("--validation", type=str, default=None)
    parser.add_argument("--output-path", type=str)
    args = parser.parse_args()

    dataset = DatasetDict()

    if args.train is not None:
        dataset["train"] = cast(Dataset, load_from_disk(args.train))

    if args.test is not None:
        dataset["test"] = cast(Dataset, load_from_disk(args.test))

    if args.validation is not None:
        dataset["validation"] = cast(Dataset, load_from_disk(args.validation))

    dataset.save_to_disk(args.output_path)


if __name__ == "__main__":
    main()
