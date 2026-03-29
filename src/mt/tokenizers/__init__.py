from os import PathLike

from .base import BaseTokenizer
from .models import BpeTokenizer, UnigramTokenizer, WordPieceTokenizer


def get_tokenizer(model: str = "unigram", file: str | PathLike[str] | None = None) -> BaseTokenizer:
    tokenizer_class = None

    match model:
        case "bpe":
            tokenizer_class = BpeTokenizer
        case "wordpiece":
            tokenizer_class = WordPieceTokenizer
        case "unigram":
            tokenizer_class = UnigramTokenizer
        case _:
            raise ValueError("Unknown tokenizer model")

    if file is not None:
        return tokenizer_class.from_file(file)
    else:
        return tokenizer_class()
