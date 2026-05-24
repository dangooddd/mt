from argparse import ArgumentParser
from pathlib import Path

from datasets import load_from_disk

BATCH_SIZE = 10000


def add_language_validation(
    batch: dict[str, list[str | None]],
    langs: list[str],
):
    import langid

    detected = {}
    for lang in langs:
        detected[f"{lang}_detected"] = [langid.classify(text or "")[0] == lang for text in batch[lang]]
    return detected


def main():
    parser = ArgumentParser("Determine correctness of language in dataset using langid")
    parser.add_argument("--dataset-path", type=str)
    parser.add_argument("--langs", nargs="+", default=["ru", "en"])
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    import langid

    langid.set_languages(args.langs)
    dataset = load_from_disk(args.dataset_path)

    validated = dataset.map(
        add_language_validation,
        fn_kwargs={"langs": args.langs},
        batched=True,
        batch_size=args.batch_size,
        load_from_cache_file=False,
    )

    dataset_path = Path(args.dataset_path)
    validated.save_to_disk(dataset_path.parent / f"{dataset_path.name}-langid")


if __name__ == "__main__":
    main()
