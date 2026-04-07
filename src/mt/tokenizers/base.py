from os import PathLike
from typing import Iterator, Optional

from tokenizers import Encoding, Tokenizer
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Digits, Metaspace, Punctuation, Sequence
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import Trainer


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
        return id if (id is not None) else 1

    @property
    def eos_token_id(self) -> int:
        id = self.tokenizer.token_to_id(self.eos_token)
        return id if (id is not None) else 1

    @property
    def unk_token_id(self) -> int:
        id = self.tokenizer.token_to_id(self.unk_token)
        return id if (id is not None) else 1

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

    def enable_padding(self, direction: str):
        self.tokenizer.enable_padding(
            direction=direction,
            pad_id=self.pad_token_id,
            pad_token=self.pad_token,
        )

    def no_padding(self):
        self.tokenizer.no_padding()

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
    def from_file(cls, path: str | PathLike[str]) -> BaseTokenizer:
        return cls(Tokenizer.from_file(str(path)))
