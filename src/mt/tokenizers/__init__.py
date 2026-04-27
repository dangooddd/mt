from .base import BaseTokenizer, BpeTokenizer, UnigramTokenizer, WordPieceTokenizer
from .bilingual import BilingualBaseTokenizer, BilingualUnigramTokenizer

TOKENIZER_CLASSES = {
    "bpe": BpeTokenizer,
    "wordpiece": WordPieceTokenizer,
    "unigram": UnigramTokenizer,
    "bilingual-unigram": BilingualUnigramTokenizer,
}

__all__ = [
    "BaseTokenizer",
    "BilingualBaseTokenizer",
    "BpeTokenizer",
    "UnigramTokenizer",
    "WordPieceTokenizer",
    "BilingualUnigramTokenizer",
    "TOKENIZER_CLASSES",
]
