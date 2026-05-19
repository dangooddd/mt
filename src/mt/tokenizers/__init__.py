from .base import BaseTokenizer, BpeTokenizer, UnigramTokenizer, WordPieceTokenizer
from .bilingual import BilingualBaseTokenizer, BilingualUnigramTokenizer
from .decoder import DecoderBaseTokenizer, DecoderUnigramTokenizer

TOKENIZER_CLASSES = {
    "bpe": BpeTokenizer,
    "wordpiece": WordPieceTokenizer,
    "unigram": UnigramTokenizer,
    "decoder-unigram": DecoderUnigramTokenizer,
    "bilingual-unigram": BilingualUnigramTokenizer,
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
    "TOKENIZER_CLASSES",
]
