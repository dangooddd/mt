from os import PathLike
from typing import Iterator, Optional, Self

from tokenizers import Encoding, Tokenizer
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.models import BPE, Unigram, WordPiece
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Digits, Metaspace, Punctuation, Sequence
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer, Trainer, UnigramTrainer, WordPieceTrainer


class BaseTokenizer:
    pad_token = "<pad>"
    bos_token = "<s>"
    eos_token = "</s>"
    unk_token = "<unk>"

    def __init__(self, tokenizer: Optional[Tokenizer] = None):
        self.tokenizer = self._create_tokenizer() if tokenizer is None else tokenizer

    @property
    def pad_token_id(self) -> int:
        id = self.tokenizer.token_to_id(self.pad_token)
        return id if (id is not None) else 1

    @property
    def bos_token_id(self) -> int:
        id = self.tokenizer.token_to_id(self.bos_token)
        return id if (id is not None) else 2

    @property
    def eos_token_id(self) -> int:
        id = self.tokenizer.token_to_id(self.eos_token)
        return id if (id is not None) else 3

    @property
    def unk_token_id(self) -> int:
        id = self.tokenizer.token_to_id(self.unk_token)
        return id if (id is not None) else 4

    def _create_model(self):
        raise NotImplementedError

    def _create_tokenizer(self) -> Tokenizer:
        tokenizer = Tokenizer(self._create_model())
        tokenizer.normalizer = NFKC()
        tokenizer.pre_tokenizer = Sequence(
            [
                Metaspace(),
                Punctuation(),
                Digits(individual_digits=True),
            ]
        )
        tokenizer.decoder = MetaspaceDecoder()
        return tokenizer

    def _create_post_processor(self):
        return TemplateProcessing(
            single=f"{self.bos_token} $A {self.eos_token}",
            special_tokens=[
                (self.bos_token, self.bos_token_id),
                (self.eos_token, self.eos_token_id),
            ],
        )

    def _create_trainer(
        self,
        vocab_size: int = 32000,
        show_progress: bool = True,
    ) -> Trainer:
        _ = vocab_size
        _ = show_progress
        raise NotImplementedError

    def encode(
        self,
        sequence: str | list[str],
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ) -> Encoding:
        return self.tokenizer.encode(
            sequence,
            is_pretokenized=is_pretokenized,
            add_special_tokens=add_special_tokens,
        )

    def encode_batch(
        self,
        input: list[str] | list[list[str]],
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ) -> list[Encoding]:
        return self.tokenizer.encode_batch(
            input,
            is_pretokenized=is_pretokenized,
            add_special_tokens=add_special_tokens,
        )

    def decode(self, ids: list[str], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(
            ids=ids,
            skip_special_tokens=skip_special_tokens,
        )

    def decode_batch(self, sequences: list[list[str]], skip_special_tokens: bool = True):
        return self.tokenizer.decode_batch(sequences, skip_special_tokens)

    def enable_padding(self, direction: str):
        self.tokenizer.enable_padding(
            direction=direction,
            pad_id=self.pad_token_id,
            pad_token=self.pad_token,
        )

    def no_padding(self):
        self.tokenizer.no_padding()

    def enable_truncation(self, max_length: int, direction: str = "right"):
        self.tokenizer.enable_truncation(max_length=max_length, direction=direction)

    def no_truncation(self):
        self.tokenizer.no_truncation()

    def get_vocab_size(self, with_added_tokens: bool = True) -> int:
        return self.tokenizer.get_vocab_size(with_added_tokens)

    def train_from_iterator(
        self,
        iterator: Iterator[str],
        vocab_size: int = 32000,
        show_progress: bool = True,
    ):
        trainer = self._create_trainer(
            vocab_size=vocab_size,
            show_progress=show_progress,
        )
        self.tokenizer = self._create_tokenizer()
        self.tokenizer.train_from_iterator(iterator, trainer=trainer)
        self.tokenizer.post_processor = self._create_post_processor()

    def train(
        self,
        files: list[str | PathLike],
        vocab_size: int = 32000,
        show_progress: bool = True,
    ):
        trainer = self._create_trainer(
            vocab_size=vocab_size,
            show_progress=show_progress,
        )
        self.tokenizer = self._create_tokenizer()
        self.tokenizer.train(files, trainer)
        self.tokenizer.post_processor = self._create_post_processor()

    def save(self, path: str, pretty: bool = True):
        self.tokenizer.save(path, pretty)

    @classmethod
    def from_file(cls, path: str | PathLike[str]) -> Self:
        return cls(Tokenizer.from_file(str(path)))


class UnigramTokenizer(BaseTokenizer):
    def _create_model(self):
        return Unigram()

    def _create_trainer(
        self,
        vocab_size: int = 32000,
        show_progress: bool = True,
    ) -> Trainer:
        return UnigramTrainer(
            vocab_size=vocab_size,
            show_progress=show_progress,
            unk_token=self.unk_token,
            special_tokens=[
                self.pad_token,
                self.bos_token,
                self.eos_token,
                self.unk_token,
            ],
        )


class BpeTokenizer(BaseTokenizer):
    def _create_model(self):
        return BPE()

    def _create_trainer(
        self,
        vocab_size: int = 32000,
        show_progress: bool = True,
    ) -> Trainer:
        return BpeTrainer(
            vocab_size=vocab_size,
            show_progress=show_progress,
            special_tokens=[
                self.pad_token,
                self.bos_token,
                self.eos_token,
                self.unk_token,
            ],
        )


class WordPieceTokenizer(BaseTokenizer):
    def _create_model(self):
        return WordPiece()

    def _create_trainer(
        self,
        vocab_size: int = 32000,
        show_progress: bool = True,
    ) -> Trainer:
        return WordPieceTrainer(
            vocab_size=vocab_size,
            show_progress=show_progress,
            special_tokens=[
                self.pad_token,
                self.bos_token,
                self.eos_token,
                self.unk_token,
            ],
        )
