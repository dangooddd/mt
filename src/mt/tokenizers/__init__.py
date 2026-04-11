from .base import BaseTokenizer
from .models import BpeTokenizer, UnigramTokenizer, WordPieceTokenizer

TOKENIZER_CLASSES = {
    "bpe": BpeTokenizer,
    "wordpiece": WordPieceTokenizer,
    "unigram": UnigramTokenizer,
}
