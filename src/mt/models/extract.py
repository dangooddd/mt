from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


StateDict = dict[str, Tensor]


def is_state_dict(value: Any) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False

    return all(isinstance(k, str) and isinstance(v, Tensor) for k, v in value.items())


def extract_model_state_dict(checkpoint: Any) -> StateDict:
    if is_state_dict(checkpoint):
        return dict(checkpoint)

    if isinstance(checkpoint, Mapping) and "model" in checkpoint:
        model = checkpoint["model"]

        if is_state_dict(model):
            return dict(model)

        if hasattr(model, "state_dict"):
            state_dict = model.state_dict()
            if is_state_dict(state_dict):
                return dict(state_dict)

    raise ValueError(
        "Could not extract model weights. Expected either a raw state_dict or a checkpoint "
        'with a "model" entry.'
    )


def main() -> None:
    parser = ArgumentParser("Extract model weights from a training checkpoint")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = extract_model_state_dict(checkpoint)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, args.output_path)

    print(f"Saved model weights to {args.output_path}")
    print(f"Parameters tensors: {len(state_dict)}")


if __name__ == "__main__":
    main()
