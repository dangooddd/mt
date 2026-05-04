import json
from pathlib import Path
from typing import Any

import torch

from ..tokenizers import TOKENIZER_CLASSES, BaseTokenizer, BilingualBaseTokenizer
from .modules import (
    DecoderOnly,
    EncoderDecoder,
    LstmSeq2Seq,
    LuongSeq2Seq,
    MambaDecoder,
    MambaHybridSeq2Seq,
    MambaSeq2Seq,
    S4Seq2Seq,
    TransformerSeq2Seq,
)

Tokenizer = tuple[BaseTokenizer, BaseTokenizer] | BilingualBaseTokenizer
Model = EncoderDecoder | DecoderOnly

MODEL_CLASSES: dict[str, Model] = {
    "lstm": LstmSeq2Seq,
    "luong": LuongSeq2Seq,
    "mamba": MambaSeq2Seq,
    "mamba-decoder": MambaDecoder,
    "mamba-hybrid": MambaHybridSeq2Seq,
    "ssm": S4Seq2Seq,
    "transformer": TransformerSeq2Seq,
}


def load_from_config(
    model_dir: Path,
    load_weights: bool = False,
) -> tuple[Model, Tokenizer, dict[str, Any]]:
    config_path = model_dir / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Required file not found: {config_path}")

    config = json.loads(config_path.read_text())
    supported_models = list(MODEL_CLASSES.keys())
    supported_tokenizers = list(TOKENIZER_CLASSES.keys())

    model_type = config.get("model")

    if model_type not in supported_models:
        raise ValueError(f"Unknown model. Supported: {supported_models}")

    model_args = dict(config.get("args") or {})

    if "tokenizer" in config:
        tokenizer_path = model_dir / "tokenizer.json"
        tokenizer_type = config.get("tokenizer")

        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Required file not found: {tokenizer_path}")

        if tokenizer_type not in supported_tokenizers:
            raise ValueError(f"Unknown tokenizer. Supported: {supported_tokenizers}")

        tokenizer = TOKENIZER_CLASSES[tokenizer_type].from_file(tokenizer_path)
        assert isinstance(tokenizer, BilingualBaseTokenizer)

        if not isinstance(tokenizer, BilingualBaseTokenizer):
            raise ValueError("The 'tokenizer' config field requires a bilingual tokenizer")

        model_args["vocab_size"] = tokenizer.get_vocab_size()
        model_args["pad_token_id"] = tokenizer.pad_token_id
        model_args["bos_token_id"] = tokenizer.bos_token_id
        model_args["eos_token_id"] = tokenizer.eos_token_id
        tokenizers = tokenizer

    else:
        src_tokenizer_path = model_dir / "src_tokenizer.json"
        tgt_tokenizer_path = model_dir / "tgt_tokenizer.json"
        src_tokenizer_type = config.get("src_tokenizer")
        tgt_tokenizer_type = config.get("tgt_tokenizer")

        for path in [src_tokenizer_path, tgt_tokenizer_path]:
            if not path.exists():
                raise FileNotFoundError(f"Required file not found: {path}")

        if src_tokenizer_type not in supported_tokenizers:
            raise ValueError(f"Unknown src tokenizer. Supported: {supported_tokenizers}")

        if tgt_tokenizer_type not in supported_tokenizers:
            raise ValueError(f"Unknown tgt tokenizer. Supported: {supported_tokenizers}")

        src_tokenizer = TOKENIZER_CLASSES[src_tokenizer_type].from_file(src_tokenizer_path)
        tgt_tokenizer = TOKENIZER_CLASSES[tgt_tokenizer_type].from_file(tgt_tokenizer_path)
        assert isinstance(src_tokenizer, BaseTokenizer)
        assert isinstance(tgt_tokenizer, BaseTokenizer)

        model_args["src_vocab_size"] = src_tokenizer.get_vocab_size()
        model_args["tgt_vocab_size"] = tgt_tokenizer.get_vocab_size()
        model_args["src_pad_token_id"] = src_tokenizer.pad_token_id
        model_args["tgt_pad_token_id"] = tgt_tokenizer.pad_token_id
        model_args["tgt_bos_token_id"] = tgt_tokenizer.bos_token_id
        model_args["tgt_eos_token_id"] = tgt_tokenizer.eos_token_id
        tokenizers = (src_tokenizer, tgt_tokenizer)

    model = MODEL_CLASSES[model_type](**model_args)

    if load_weights:
        model_path = model_dir / "model.pt"
        checkpoint = torch.load(model_path, map_location="cpu")

        if (
            isinstance(checkpoint, dict)
            and "model" in checkpoint
            and isinstance(checkpoint["model"], dict)
        ):
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)

    return model, tokenizers, config
