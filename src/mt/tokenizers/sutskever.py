from os import PathLike
from typing import Iterator, Optional

from tokenizers import Encoding, Tokenizer
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.models import BPE, Unigram, WordPiece
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer, Trainer, UnigramTrainer, WordPieceTrainer


class Seq2SeqTokenizer:
    pad_token = "<pad>"
    bos_token = "<s>"
    eos_token = "</s>"
    unk_token = "<unk>"

    def __init__(
        self,
        tokenizer: Optional[Tokenizer] = None,
        model_type: str = "unigram",
    ):
        self.model_type = model_type
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

    def _create_tokenizer(self) -> Tokenizer:
        if self.model_type == "unigram":
            model = Unigram()
        elif self.model_type == "bpe":
            model = BPE()
        else:
            model = WordPiece()

        tokenizer = Tokenizer(model)
        tokenizer.normalizer = NFKC()
        tokenizer.pre_tokenizer = Metaspace()
        tokenizer.decoder = MetaspaceDecoder()
        return tokenizer

    def _create_post_processor(self):
        return TemplateProcessing(
            single=f"{self.bos_token} $A {self.eos_token}",
            special_tokens=[
                (self.pad_token, self.pad_token_id),
                (self.bos_token, self.bos_token_id),
                (self.eos_token, self.eos_token_id),
                (self.unk_token, self.unk_token_id),
            ],
        )

    def _create_trainer(
        self,
        vocab_size: int = 32000,
        show_progress: bool = True,
    ) -> Trainer:
        if self.model_type == "unigram":
            trainer = UnigramTrainer(
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
        elif self.model_type == "bpe":
            trainer = BpeTrainer(
                vocab_size=vocab_size,
                show_progress=show_progress,
                special_tokens=[
                    self.pad_token,
                    self.bos_token,
                    self.eos_token,
                    self.unk_token,
                ],
            )
        else:
            trainer = WordPieceTrainer(
                vocab_size=vocab_size,
                show_progress=show_progress,
                special_tokens=[
                    self.pad_token,
                    self.bos_token,
                    self.eos_token,
                    self.unk_token,
                ],
            )

        return trainer

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
    def from_file(cls, path: str) -> Seq2SeqTokenizer:
        return cls(Tokenizer.from_file(path))
