from pathlib import Path
from typing import Optional

import typer
from datasets import DatasetDict

from ..datasets import (
    add_length_ratio_score,
    add_punctuation_intersection_score,
    add_word_count_ratio_score,
    deduplicate_translation_pairs,
    filter_by_score_threshold,
    get_dataset_preset,
    limit_samples,
    load_dataset_cached,
)


def transform_tatoeba_item(example):
    return {
        "translation": {
            "en": example["source_text"],
            "ru": example["target_text"],
        }
    }


def main(
    save_path: str,
    word_filtering_threshold: Optional[float] = None,
    length_filtering_threshold: Optional[float] = None,
    punctuation_filtering_threshold: Optional[float] = None,
    deduplicate: bool = False,
    max_samples: Optional[int] = None,
):
    tatoeba = load_dataset_cached(get_dataset_preset("tatoeba_mt_train"))
    opus = load_dataset_cached(get_dataset_preset("opus-100"))

    if max_samples is not None:
        tatoeba = limit_samples(tatoeba, max_samples)

    tatoeba = tatoeba["train"].map(
        transform_tatoeba_item,
        remove_columns=["source_text", "target_text", "source_lang", "target_lang"],
        batched=False,
        num_proc=4,
    )

    if deduplicate:
        tatoeba = deduplicate_translation_pairs(tatoeba, max_samples=max_samples)

    if word_filtering_threshold is not None:
        tatoeba = add_word_count_ratio_score(tatoeba)
        tatoeba = filter_by_score_threshold(
            tatoeba,
            "word_count_ratio",
            word_filtering_threshold,
        )
        print(f"After word filtering: {len(tatoeba)}")

    if punctuation_filtering_threshold is not None:
        tatoeba = add_punctuation_intersection_score(tatoeba)
        tatoeba = filter_by_score_threshold(
            tatoeba,
            "punctuation_intersection",
            punctuation_filtering_threshold,
        )
        print(f"After punctuation filtering: {len(tatoeba)}")

    if length_filtering_threshold is not None:
        tatoeba = add_length_ratio_score(tatoeba)
        tatoeba = filter_by_score_threshold(
            tatoeba,
            "length_ratio",
            length_filtering_threshold,
        )
        print(f"After length filtering: {len(tatoeba)}")

    merged = DatasetDict(
        {
            "train": tatoeba,
            "test": opus["test"],
            "validation": opus["validation"],
        }
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    merged.save_to_disk(save_path, max_shard_size="200MB")


if __name__ == "__main__":
    typer.run(main)
