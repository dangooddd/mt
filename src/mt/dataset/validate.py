from argparse import ArgumentParser
from pathlib import Path

import transformers
from datasets import load_from_disk
from transformers import pipeline

transformers.logging.set_verbosity_error()
pipe = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")


def add_language_validation(batch, feature_name: str, column: str, expected: str):
    texts = batch[column]
    results = pipe(texts, truncation=True)
    valid = [x["label"] == expected for x in results]
    return {feature_name: valid}


def main():
    parser = ArgumentParser("Determine correctness of language in dataset")
    parser.add_argument("--dataset-path", type=str)
    parser.add_argument("--feature-name", type=str)
    parser.add_argument("--column", type=str)
    parser.add_argument("--expected", type=str)
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset_path)

    scored = dataset.map(
        add_language_validation,
        fn_kwargs={
            "feature_name": args.feature_name,
            "column": args.column,
            "expected": args.expected,
        },
        batched=True,
        batch_size=1000,
        load_from_cache_file=False,
    )

    dataset_path = Path(args.dataset_path)
    scored.save_to_disk(dataset_path.parent / f"{dataset_path.name}-validated-{args.column}")


if __name__ == "__main__":
    main()
