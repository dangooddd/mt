from .base import BaseTokenizer, BpeTokenizer, UnigramTokenizer, WordPieceTokenizer
from .bilingual import (
    BilingualBaseTokenizer,
    BilingualBpeTokenizer,
    BilingualUnigramTokenizer,
    BilingualWordPieceTokenizer,
)
from .decoder import DecoderBaseTokenizer, DecoderUnigramTokenizer

TOKENIZER_CLASSES = {
    "bpe": BpeTokenizer,
    "wordpiece": WordPieceTokenizer,
    "unigram": UnigramTokenizer,
    "decoder-unigram": DecoderUnigramTokenizer,
    "bilingual-unigram": BilingualUnigramTokenizer,
    "bilingual-bpe": BilingualBpeTokenizer,
    "bilingual-wordpiece": BilingualWordPieceTokenizer,
}

__all__ = [
    "BaseTokenizer",
    "BpeTokenizer",
    "UnigramTokenizer",
    "WordPieceTokenizer",
    "DecoderBaseTokenizer",
    "DecoderUnigramTokenizer",
    "BilingualBaseTokenizer",
    "BilingualUnigramTokenizer",
    "BilingualBpeTokenizer",
    "BilingualWordPieceTokenizer",
    "TOKENIZER_CLASSES",
]
