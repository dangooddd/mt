from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, cast
import json

from torch import nn

from mt.models.evaluation import MODEL_CLASSES, resolve_tokenizer


def create_model(model_dir: Path) -> tuple[nn.Module, dict[str, Any]]:
    config_path = model_dir / "config.json"
    src_tokenizer_path = model_dir / "src_tokenizer.json"
    tgt_tokenizer_path = model_dir / "tgt_tokenizer.json"

    for path in [config_path, src_tokenizer_path, tgt_tokenizer_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    config = cast(dict[str, Any], json.loads(config_path.read_text()))

    class_name = config.get("class") or config.get("name") or config.get("model")
    if class_name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model class '{class_name}'. Supported: {sorted(MODEL_CLASSES)}")

    model_args = dict(config.get("args") or config.get("model_args") or {})
    src_tokenizer = resolve_tokenizer(config, "src_tokenizer", src_tokenizer_path)
    tgt_tokenizer = resolve_tokenizer(config, "tgt_tokenizer", tgt_tokenizer_path)

    model_args["src_vocab_size"] = src_tokenizer.get_vocab_size()
    model_args["tgt_vocab_size"] = tgt_tokenizer.get_vocab_size()
    model_args["src_pad_token_id"] = src_tokenizer.pad_token_id
    model_args["tgt_pad_token_id"] = tgt_tokenizer.pad_token_id
    model_args["tgt_bos_token_id"] = tgt_tokenizer.bos_token_id
    model_args["tgt_eos_token_id"] = tgt_tokenizer.eos_token_id

    model = MODEL_CLASSES[class_name](**model_args)
    return model, config


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def format_int(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def main() -> None:
    parser = ArgumentParser("Show parameter counts for an MT model config")
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()

    model, config = create_model(args.model_dir)
    total, trainable = count_parameters(model)
    non_trainable = total - trainable

    class_name = config.get("class") or config.get("name") or config.get("model")

    print(f"model_dir: {args.model_dir}")
    print(f"architecture: {class_name}")
    print(f"total parameters: {format_int(total)}")
    print(f"trainable parameters: {format_int(trainable)}")
    print(f"non-trainable parameters: {format_int(non_trainable)}")


if __name__ == "__main__":
    main()
