from argparse import ArgumentParser

from datasets import load_from_disk

from . import get_tokenizer


def main():
    parser = ArgumentParser("Train tokenizer on dataset")
    parser.add_argument("--lang", type=str, default="ru")
    parser.add_argument("--model", type=str, default="unigram")
    parser.add_argument("--vocab-size", type=int, default=24000)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--output-path", type=str)
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset)["train"]
    tokenizer = get_tokenizer(args.model)
    tokenizer.train_from_iterator(dataset[args.lang], args.vocab_size)
    tokenizer.save(args.output_path)


if __name__ == "__main__":
    main()
