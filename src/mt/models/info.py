from argparse import ArgumentParser
from pathlib import Path

from torch import nn

from . import load_from_config


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def main() -> None:
    parser = ArgumentParser("Show parameter counts for an MT model config")
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()

    model, _, _ = load_from_config(args.model_dir)
    total, trainable = count_parameters(model)

    print(f"Model dir: {args.model_dir}")
    print(f"Total parameters: {total}")
    print(f"Trainable parameters: {trainable}")


if __name__ == "__main__":
    main()
