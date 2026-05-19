from collections import defaultdict
from os import PathLike
from typing import Iterator, Optional, Self

from tokenizers import Encoding, Tokenizer
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.models import Unigram
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Digits, Metaspace, Punctuation, Sequence
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import Trainer, UnigramTrainer


class DecoderBaseTokenizer:
    pad_token = "<pad>"
    bos_token = "<s>"
    eos_token = "</s>"
    unk_token = "<unk>"
    ru_token = "<ru>"
    en_token = "<en>"

    def __init__(self, tokenizer: Optional[Tokenizer] = None):
        self.tokenizer = self._create_tokenizer() if tokenizer is None else tokenizer
        self.post_processors = defaultdict(
            lambda: self.post_processors["ru"],
            {
                "ru": self._create_post_processor("ru"),
                "en": self._create_post_processor("en"),
            },
        )

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
        return id if (id is not None) else 5

    @property
    def ru_token_id(self) -> int:
        id = self.tokenizer.token_to_id(self.ru_token)
        return id if (id is not None) else 6

    @property
    def en_token_id(self) -> int:
        id = self.tokenizer.token_to_id(self.en_token)
        return id if (id is not None) else 7

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

    def _create_post_processor(self, lang: str):
        if lang == "ru":
            src_token = self.ru_token
            tgt_token = self.en_token
        else:
            src_token = self.en_token
            tgt_token = self.ru_token

        bos = f"{self.bos_token} {src_token}"
        sep = f"{tgt_token}"
        eos = f"{self.eos_token}"

        return TemplateProcessing(
            single=f"{bos} $A {sep}",
            pair=f"{bos} $A {sep}:1 $B:1 {eos}:1",
            special_tokens=[
                (self.bos_token, self.bos_token_id),
                (self.eos_token, self.eos_token_id),
                (self.ru_token, self.ru_token_id),
                (self.en_token, self.en_token_id),
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

    def set_source_language(self, lang: str):
        self.tokenizer.post_processor = self.post_processors[lang]

    def encode(
        self,
        sequence: str | list[str],
        pair: str | list[str] | None = None,
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ) -> Encoding:
        return self.tokenizer.encode(
            sequence,
            pair=pair,
            is_pretokenized=is_pretokenized,
            add_special_tokens=add_special_tokens,
        )

    def encode_batch(
        self,
        input: list[str] | list[list[str]] | list[tuple[str, str]],
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
        self.tokenizer.post_processor = self.post_processors["ru"]

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
        self.tokenizer.post_processor = self.post_processors["ru"]

    def save(self, path: str, pretty: bool = True):
        self.tokenizer.save(path, pretty)

    @classmethod
    def from_file(cls, path: str | PathLike[str]) -> Self:
        return cls(Tokenizer.from_file(str(path)))


class DecoderUnigramTokenizer(DecoderBaseTokenizer):
    def _create_model(self):
        return Unigram()

    def _create_trainer(
        self,
        vocab_size: int = 64000,
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
                self.ru_token,
                self.en_token,
            ],
        )
