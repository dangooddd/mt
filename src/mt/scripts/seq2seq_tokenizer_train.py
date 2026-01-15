from pathlib import Path
from typing import Optional, cast

import typer
from datasets import DatasetDict, load_from_disk

from ..datasets import (
    get_dataset_preset,
    limit_samples,
    load_dataset_cached,
    shuffle_splits,
)
from ..tokenizers import Seq2SeqTokenizer


def main(
    output_path: str,
    lang: str = "ru",
    vocab_size: int = 32000,
    model_type: str = "unigram",
    dataset_preset: str = "opus-100",
    max_samples: Optional[int] = None,
    from_disk: bool = True,
):
    if from_disk:
        dataset = cast(DatasetDict, load_from_disk(dataset_preset))
    else:
        dataset = load_dataset_cached(get_dataset_preset(dataset_preset))

    if max_samples is not None:
        dataset = limit_samples(dataset, max_samples)
    dataset = shuffle_splits(dataset)

    def translation_iterator(dataset, lang):
        for sample in dataset["train"]:
            yield sample["translation"][lang]

    tokenizer = Seq2SeqTokenizer(model_type=model_type)
    tokenizer.train_from_iterator(
        translation_iterator(dataset, lang),
        vocab_size=vocab_size,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_path)


if __name__ == "__main__":
    typer.run(main)
