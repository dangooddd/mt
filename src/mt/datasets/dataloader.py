import logging
from functools import partial
from typing import List, Optional

import torch
from datasets import Dataset, DatasetDict
from lightning import LightningDataModule
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from ..tokenizers import Seq2SeqTokenizer

logger = logging.getLogger(__name__)


def collate_fn(
    batch: List[tuple[Tensor, Tensor]],
    src_pad_id: int = 0,
    tgt_pad_id: int = 0,
) -> tuple[Tensor, Tensor]:
    src_batch, tgt_batch = zip(*batch)

    src_padded = pad_sequence(
        src_batch,
        batch_first=True,
        padding_value=src_pad_id,
    )

    tgt_padded = pad_sequence(
        tgt_batch,
        batch_first=True,
        padding_value=tgt_pad_id,
    )

    return src_padded, tgt_padded


class Seq2SeqDataset(TorchDataset):
    def __init__(
        self,
        dataset: Dataset,
        src_tokenizer: Seq2SeqTokenizer,
        tgt_tokenizer: Seq2SeqTokenizer,
        src_field: str = "en",
        tgt_field: str = "ru",
        max_length: Optional[int] = None,
    ):
        self.dataset = dataset.map(
            partial(
                self._tokenize,
                src_tokenizer=src_tokenizer,
                tgt_tokenizer=tgt_tokenizer,
                src_field=src_field,
                tgt_field=tgt_field,
            ),
            batched=True,
            remove_columns=["translation"],
            desc="Tokenizing dataset",
        )

        if max_length is not None:
            original_length = len(self.dataset)
            self.dataset = self.dataset.filter(
                partial(self._filter, max_length=max_length),
                batched=True,
                desc=f"Filtering sequences with max_length={max_length}",
            )
            filtered_length = len(self.dataset)
            logger.info(
                f"Filtered dataset: {original_length} -> {filtered_length} "
                f"({filtered_length / original_length * 100:.2f}% retained)"
            )

    @staticmethod
    def _tokenize(
        examples,
        src_tokenizer: Seq2SeqTokenizer,
        tgt_tokenizer: Seq2SeqTokenizer,
        src_field: str,
        tgt_field: str,
    ):
        src_texts = [translation[src_field] for translation in examples["translation"]]
        tgt_texts = [translation[tgt_field] for translation in examples["translation"]]

        src_encodings = src_tokenizer.encode_batch(
            src_texts,
            add_special_tokens=True,
        )

        tgt_encodings = tgt_tokenizer.encode_batch(
            tgt_texts,
            add_special_tokens=True,
        )

        return {
            "src_ids": [encoding.ids for encoding in src_encodings],
            "tgt_ids": [encoding.ids for encoding in tgt_encodings],
        }

    @staticmethod
    def _filter(examples, max_length: int):
        return [
            len(src_ids) + len(tgt_ids) <= max_length
            for src_ids, tgt_ids in zip(examples["src_ids"], examples["tgt_ids"])
        ]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        item = self.dataset[index]

        src_tensor = torch.tensor(item["src_ids"], dtype=torch.long)
        tgt_tensor = torch.tensor(item["tgt_ids"], dtype=torch.long)

        return src_tensor, tgt_tensor


class Seq2SeqDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: DatasetDict,
        src_tokenizer: Seq2SeqTokenizer,
        tgt_tokenizer: Seq2SeqTokenizer,
        src_field: str = "en",
        tgt_field: str = "ru",
        batch_size: int = 32,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle_train: bool = True,
        max_length: Optional[int] = None,
    ):
        super().__init__()

        self.dataset = dataset
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_field = src_field
        self.tgt_field = tgt_field

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.test_batch_size = test_batch_size or batch_size

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_train = shuffle_train
        self.max_length = max_length

        self.train_dataset: Optional[Seq2SeqDataset] = None
        self.val_dataset: Optional[Seq2SeqDataset] = None
        self.test_dataset: Optional[Seq2SeqDataset] = None

    def setup(self, stage: str | None = None) -> None:
        if (stage == "fit" or stage is None) and "train" in self.dataset:
            self.train_dataset = Seq2SeqDataset(
                dataset=self.dataset["train"],
                src_tokenizer=self.src_tokenizer,
                tgt_tokenizer=self.tgt_tokenizer,
                src_field=self.src_field,
                tgt_field=self.tgt_field,
                max_length=self.max_length,
            )

        if (stage == "fit" or stage is None) and "validation" in self.dataset:
            self.val_dataset = Seq2SeqDataset(
                dataset=self.dataset["validation"],
                src_tokenizer=self.src_tokenizer,
                tgt_tokenizer=self.tgt_tokenizer,
                src_field=self.src_field,
                tgt_field=self.tgt_field,
                max_length=self.max_length,
            )

        if (stage == "test" or stage is None) and "test" in self.dataset:
            self.test_dataset = Seq2SeqDataset(
                dataset=self.dataset["test"],
                src_tokenizer=self.src_tokenizer,
                tgt_tokenizer=self.tgt_tokenizer,
                src_field=self.src_field,
                tgt_field=self.tgt_field,
                max_length=self.max_length,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup(stage='fit') first")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=partial(
                collate_fn,
                src_pad_id=self.src_tokenizer.pad_token_id,
                tgt_pad_id=self.tgt_tokenizer.pad_token_id,
            ),
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup(stage='validate') first")

        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=partial(
                collate_fn,
                src_pad_id=self.src_tokenizer.pad_token_id,
                tgt_pad_id=self.tgt_tokenizer.pad_token_id,
            ),
        )

    def test_dataloader(self) -> DataLoader:
        """DataLoader для тестовых данных."""
        if self.test_dataset is None:
            raise RuntimeError("Call setup(stage='test') first")

        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=partial(
                collate_fn,
                src_pad_id=self.src_tokenizer.pad_token_id,
                tgt_pad_id=self.tgt_tokenizer.pad_token_id,
            ),
        )
