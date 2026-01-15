from typing import Any, Optional, overload

from datasets import Dataset, DatasetDict, load_dataset

from ..constants import HF_CACHE_DIR

CONFIGS: dict[str, dict[str, Any]] = {
    "opus-100": {
        "path": "Helsinki-NLP/opus-100",
        "name": "en-ru",
    },
    "wikipedia": {
        "path": "Helsinki-NLP/opus_wikipedia",
        "name": "en-ru",
    },
    "opus_books": {
        "path": "Helsinki-NLP/opus_books",
        "name": "en-ru",
    },
    "tatoeba_mt_train": {
        "path": "Helsinki-NLP/tatoeba_mt_train",
        "name": "eng-rus",
    },
    "wmt19": {
        "path": "wmt19",
        "name": "ru-en",
    },
    "news_commentary": {
        "path": "Helsinki-NLP/news_commentary",
        "name": "en-ru",
    },
}


def get_dataset_preset(
    preset: str = "opus-100",
):
    return CONFIGS[preset]


def load_dataset_cached(
    config: dict[str, Any],
    cache_dir: Optional[str] = str(HF_CACHE_DIR),
) -> DatasetDict:
    """
    Args:
        config: Dataset preset name
        max_samples: Maximum number of samples per split
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        cache_dir: Directory to cache the dataset

    Returns:
        DatasetDict with loaded splits
    """
    dataset = load_dataset(
        **config,
        cache_dir=cache_dir,
    )

    return dataset


@overload
def limit_samples(dataset: Dataset, max_samples: int) -> Dataset: ...


@overload
def limit_samples(dataset: DatasetDict, max_samples: int) -> DatasetDict: ...


def limit_samples(
    dataset: Dataset | DatasetDict,
    max_samples: int,
) -> Dataset | DatasetDict:
    if isinstance(dataset, DatasetDict):
        for split in dataset:
            if len(dataset[split]) > max_samples:
                dataset[split] = dataset[split].select(range(max_samples))
    else:
        if len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))

    return dataset


def shuffle_splits(dataset: DatasetDict, seed: int = 42):
    for split in dataset:
        dataset[split] = dataset[split].shuffle(seed=seed)

    return dataset
