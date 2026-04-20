import json
from pathlib import Path
from typing import Any

import torch

from mt.tokenizers import TOKENIZER_CLASSES, BaseTokenizer

from .lstm import LstmSeq2Seq
from .luong import LuongSeq2Seq
from .mamba import MambaSeq2Seq
from .ssm import S4Seq2Seq

MODEL_CLASSES = {
    "lstm": LstmSeq2Seq,
    "luong": LuongSeq2Seq,
    "mamba": MambaSeq2Seq,
    "ssm": S4Seq2Seq,
}

Models = LstmSeq2Seq | LuongSeq2Seq | MambaSeq2Seq | S4Seq2Seq


def load_from_config(
    model_dir: Path,
    load_weights: bool = False,
) -> tuple[Models, BaseTokenizer, BaseTokenizer, dict[str, Any]]:
    config_path = model_dir / "config.json"
    src_tokenizer_path = model_dir / "src_tokenizer.json"
    tgt_tokenizer_path = model_dir / "tgt_tokenizer.json"

    for path in [config_path, src_tokenizer_path, tgt_tokenizer_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    config = json.loads(config_path.read_text())
    supported_models = list(MODEL_CLASSES.keys())
    supported_tokenizers = list(TOKENIZER_CLASSES.keys())

    model_type = config.get("model")
    src_tokenizer_type = config.get("src_tokenizer")
    tgt_tokenizer_type = config.get("tgt_tokenizer")

    if model_type not in supported_models:
        raise ValueError(f"Unknown model. Supported: {supported_models}")

    if src_tokenizer_type not in supported_tokenizers:
        raise ValueError(f"Unknown src tokenizer. Supported: {supported_tokenizers}")

    if tgt_tokenizer_type not in supported_tokenizers:
        raise ValueError(f"Unknown tgt tokenizer. Supported: {supported_tokenizers}")

    src_tokenizer = TOKENIZER_CLASSES[src_tokenizer_type].from_file(src_tokenizer_path)
    tgt_tokenizer = TOKENIZER_CLASSES[tgt_tokenizer_type].from_file(tgt_tokenizer_path)

    model_args = dict(config.get("args") or {})
    model_args["src_vocab_size"] = src_tokenizer.get_vocab_size()
    model_args["tgt_vocab_size"] = tgt_tokenizer.get_vocab_size()
    model_args["src_pad_token_id"] = src_tokenizer.pad_token_id
    model_args["tgt_pad_token_id"] = tgt_tokenizer.pad_token_id
    model_args["tgt_bos_token_id"] = tgt_tokenizer.bos_token_id
    model_args["tgt_eos_token_id"] = tgt_tokenizer.eos_token_id

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

    return model, src_tokenizer, tgt_tokenizer, config
