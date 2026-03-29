from argparse import ArgumentParser

import transformers
from transformers import pipeline

transformers.logging.set_verbosity_error()
pipe = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")


def add_language_validation(batch, column: str, expected: str):
    texts = batch[column]
    pipe(texts, top_k=1, truncation=True)


def main():
    parser = ArgumentParser("Determine correctness of language in dataset")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--column", type=str)
    parser.add_argument("--expected", type=str)
    args = parser.parse_args()


if __name__ == "__main__":
    main()
