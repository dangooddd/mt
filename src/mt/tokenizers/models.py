from tokenizers.models import BPE, Unigram, WordPiece
from tokenizers.trainers import BpeTrainer, Trainer, UnigramTrainer, WordPieceTrainer

from .base import BaseTokenizer


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
