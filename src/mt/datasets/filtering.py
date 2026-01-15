import string
from typing import Optional, overload

from datasets import Dataset, DatasetDict
from tqdm import tqdm


@overload
def add_punctuation_intersection_score(
    dataset: Dataset,
    src_field: str = "en",
    tgt_field: str = "ru",
    translation_feature: str = "translation",
    score_feature: str = "punctuation_intersection",
) -> Dataset: ...


@overload
def add_punctuation_intersection_score(
    dataset: DatasetDict,
    src_field: str = "en",
    tgt_field: str = "ru",
    translation_feature: str = "translation",
    score_feature: str = "punctuation_intersection",
) -> DatasetDict: ...


def add_punctuation_intersection_score(
    dataset: Dataset | DatasetDict,
    src_field: str = "en",
    tgt_field: str = "ru",
    translation_feature: str = "translation",
    score_feature: str = "punctuation_intersection",
) -> Dataset | DatasetDict:
    def compute_score(example: dict) -> dict:
        src = example[translation_feature][src_field]
        tgt = example[translation_feature][tgt_field]

        src_punctuation = {char for char in src if char in string.punctuation}
        tgt_punctuation = {char for char in tgt if char in string.punctuation}
        intersection = src_punctuation.intersection(tgt_punctuation)

        if len(src) == 0 or len(tgt) == 0:
            score = 0.0
        elif max(len(src_punctuation), len(tgt_punctuation)) == 0:
            score = 1.0
        else:
            score = len(intersection) / max(len(src_punctuation), len(tgt_punctuation))

        return {score_feature: score}

    if isinstance(dataset, DatasetDict):
        return DatasetDict(
            {
                split: dataset[split].map(
                    compute_score, batched=False, desc=f"Adding {score_feature}"
                )
                for split in dataset
            }
        )
    else:
        return dataset.map(compute_score, batched=False, desc=f"Adding {score_feature}")


@overload
def add_word_count_ratio_score(
    dataset: Dataset,
    src_field: str = "en",
    tgt_field: str = "ru",
    translation_feature: str = "translation",
    score_feature: str = "word_count_ratio",
) -> Dataset: ...


@overload
def add_word_count_ratio_score(
    dataset: DatasetDict,
    src_field: str = "en",
    tgt_field: str = "ru",
    translation_feature: str = "translation",
    score_feature: str = "word_count_ratio",
) -> DatasetDict: ...


def add_word_count_ratio_score(
    dataset: Dataset | DatasetDict,
    src_field: str = "en",
    tgt_field: str = "ru",
    translation_feature: str = "translation",
    score_feature: str = "word_count_ratio",
) -> Dataset | DatasetDict:
    def compute_score(example: dict) -> dict:
        src = example[translation_feature][src_field]
        tgt = example[translation_feature][tgt_field]

        src_words = len(src.split())
        tgt_words = len(tgt.split())

        if src_words == 0 or tgt_words == 0:
            score = 0.0
        else:
            min_words = min(src_words, tgt_words)
            max_words = max(src_words, tgt_words)
            score = min_words / max_words

        return {score_feature: score}

    if isinstance(dataset, DatasetDict):
        return DatasetDict(
            {
                split: dataset[split].map(
                    compute_score, batched=False, desc=f"Adding {score_feature}"
                )
                for split in dataset
            }
        )
    else:
        return dataset.map(compute_score, batched=False, desc=f"Adding {score_feature}")


@overload
def add_length_ratio_score(
    dataset: Dataset,
    src_field: str = "en",
    tgt_field: str = "ru",
    translation_feature: str = "translation",
    score_feature: str = "length_ratio",
) -> Dataset: ...


@overload
def add_length_ratio_score(
    dataset: DatasetDict,
    src_field: str = "en",
    tgt_field: str = "ru",
    translation_feature: str = "translation",
    score_feature: str = "length_ratio",
) -> DatasetDict: ...


def add_length_ratio_score(
    dataset: Dataset | DatasetDict,
    src_field: str = "en",
    tgt_field: str = "ru",
    translation_feature: str = "translation",
    score_feature: str = "length_ratio",
) -> Dataset | DatasetDict:
    def compute_score(example: dict) -> dict:
        src = example[translation_feature][src_field]
        tgt = example[translation_feature][tgt_field]

        src_len = len(src)
        tgt_len = len(tgt)

        if src_len == 0 or tgt_len == 0:
            score = 0.0
        else:
            min_words = min(src_len, tgt_len)
            max_words = max(src_len, tgt_len)
            score = min_words / max_words

        return {score_feature: score}

    if isinstance(dataset, DatasetDict):
        return DatasetDict(
            {
                split: dataset[split].map(
                    compute_score, batched=False, desc=f"Adding {score_feature}"
                )
                for split in dataset
            }
        )
    else:
        return dataset.map(compute_score, batched=False, desc=f"Adding {score_feature}")


@overload
def deduplicate_translation_pairs(
    dataset: Dataset,
    src_field: str = "en",
    tgt_field: str = "ru",
    translation_feature: str = "translation",
    max_samples: Optional[int] = None,
) -> Dataset: ...


@overload
def deduplicate_translation_pairs(
    dataset: DatasetDict,
    src_field: str = "en",
    tgt_field: str = "ru",
    translation_feature: str = "translation",
    max_samples: Optional[int] = None,
) -> DatasetDict: ...


def deduplicate_translation_pairs(
    dataset: Dataset | DatasetDict,
    src_field: str = "en",
    tgt_field: str = "ru",
    translation_feature: str = "translation",
    max_samples: Optional[int] = None,
) -> Dataset | DatasetDict:
    def deduplicate(dataset: Dataset):
        seen = set()
        keep = []

        for i, example in tqdm(enumerate(dataset), desc="Deduplicating dataset"):
            translation = example[translation_feature]
            key = (translation[src_field], translation[tgt_field])

            if key not in seen:
                seen.add(key)
                keep.append(i)

            if max_samples is not None and len(keep) >= max_samples:
                break

        return dataset.select(keep)

    if isinstance(dataset, DatasetDict):
        return DatasetDict({split: deduplicate(dataset[split]) for split in dataset})
    else:
        return deduplicate(dataset)


@overload
def filter_by_score_threshold(
    dataset: Dataset,
    score_field: str,
    threshold: float,
    above: bool = True,
) -> Dataset: ...


@overload
def filter_by_score_threshold(
    dataset: DatasetDict,
    score_field: str,
    threshold: float,
    above: bool = True,
) -> DatasetDict: ...


def filter_by_score_threshold(
    dataset: Dataset | DatasetDict,
    score_field: str,
    threshold: float,
    above: bool = True,
) -> Dataset | DatasetDict:
    def filter_fn(examples):
        scores = examples[score_field]
        if above:
            return [score >= threshold for score in scores]
        else:
            return [score <= threshold for score in scores]

    if isinstance(dataset, DatasetDict):
        return DatasetDict(
            {
                split: dataset[split].filter(
                    filter_fn,
                    batched=True,
                    desc=f"Filtering by {score_field} {'>=' if above else '<='} {threshold}",
                )
                for split in dataset
            }
        )
    else:
        return dataset.filter(
            filter_fn,
            batched=True,
            desc=f"Filtering by {score_field} {'>=' if above else '<='} {threshold}",
        )
