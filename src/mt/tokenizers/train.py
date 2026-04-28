from argparse import ArgumentParser
from itertools import chain
from pathlib import Path

from datasets import load_from_disk

from . import TOKENIZER_CLASSES


def main():
    parser = ArgumentParser("Train tokenizer on dataset")
    parser.add_argument("--lang", nargs="+", default=["ru"])
    parser.add_argument("--model", type=str, default="unigram")
    parser.add_argument("--vocab-size", type=int, default=24000)
    parser.add_argument("--dataset-path", type=str)
    parser.add_argument("--output-path", type=str)
    args = parser.parse_args()
    Path(args.output_path).parent.mkdir(exist_ok=True, parents=True)

    dataset = load_from_disk(args.dataset_path)["train"]

    if args.model not in TOKENIZER_CLASSES:
        raise ValueError(f"Unknown tokenizer. Supported: {list(TOKENIZER_CLASSES.keys())}")

    print(f"Model: {args.model}")
    print(f"Langs: {args.lang}")
    print(f"Vocab: {args.vocab_size}")

    tokenizer = TOKENIZER_CLASSES[args.model]()
    iterator = chain.from_iterable(dataset[lang] for lang in args.lang)
    tokenizer.train_from_iterator(iterator, args.vocab_size)
    tokenizer.save(args.output_path)


if __name__ == "__main__":
    main()
