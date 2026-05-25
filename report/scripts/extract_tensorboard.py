from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "data" / "runs"
CSV_DIR = ROOT / "report" / "csv"
IMAGES_DIR = ROOT / "report" / "images"
MAX_STEP = 100_000


@dataclass(frozen=True)
class SeriesSpec:
    run: str
    label: str
    tag: str


def read_scalars(run: str, tag: str) -> list[tuple[int, float]]:
    values: list[tuple[int, float]] = []
    for event_file in sorted((RUNS_DIR / run).glob("events.out.tfevents*")):
        accumulator = event_accumulator.EventAccumulator(
            str(event_file), size_guidance={"scalars": 0}
        )
        accumulator.Reload()
        if tag not in accumulator.Tags().get("scalars", []):
            continue
        for index, event in enumerate(accumulator.Scalars(tag), start=1):
            step = event.step if event.step > 0 else index
            values.append((step, float(event.value)))
    values.sort(key=lambda item: item[0])
    return values


def ensure_finite(spec: SeriesSpec, values: list[tuple[int, float]]) -> list[tuple[int, float]]:
    bad = [(step, value) for step, value in values if not math.isfinite(value)]
    if bad:
        raise ValueError(f"Run {spec.run}, tag {spec.tag} contains NaN or Inf")
    return values


def trim_train(values: list[tuple[int, float]]) -> list[tuple[int, float]]:
    return [(step, value) for step, value in values if step <= MAX_STEP]


def write_series_csv(filename: str, specs: list[SeriesSpec]) -> dict[str, list[tuple[int, float]]]:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    series: dict[str, list[tuple[int, float]]] = {}

    with (CSV_DIR / filename).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "run", "tag", "step", "value"])
        for spec in specs:
            values = ensure_finite(spec, read_scalars(spec.run, spec.tag))
            if spec.tag == "train/loss":
                values = trim_train(values)
            series[spec.label] = values
            for step, value in values:
                writer.writerow([spec.label, spec.run, spec.tag, step, value])
    return series


def plot_series(
    series: dict[str, list[tuple[int, float]]],
    filename: str,
    xlabel: str,
    ylabel: str,
    step_scale: float = 1.0,
) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    line_styles = ["-", "--", "-.", ":"]

    for i, (label, values) in enumerate(series.items()):
        if not values:
            continue
        x = [step / step_scale for step, _ in values]
        y = [value for _, value in values]
        ax.plot(
            x,
            y,
            label=label,
            linewidth=1.8,
            linestyle=line_styles[i % len(line_styles)],
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(frameon=True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    fig.savefig(IMAGES_DIR / filename)
    plt.close(fig)


def main() -> None:
    train_specs = [
        SeriesSpec("lstm-v2", "LSTM", "train/loss"),
        SeriesSpec("transformer-v2", "Transformer", "train/loss"),
        SeriesSpec("mamba-hybrid-v3", "Mamba encoder-decoder", "train/loss"),
        SeriesSpec("mamba-decoder-v3", "Mamba decoder-only", "train/loss"),
    ]
    bleu_specs = [
        SeriesSpec("lstm-v2", "LSTM", "evaluation/bleu"),
        SeriesSpec("transformer-v2", "Transformer", "evaluation/bleu"),
        SeriesSpec("mamba-hybrid-v3", "Mamba encoder-decoder", "evaluation/bleu"),
        SeriesSpec("mamba-decoder-v3", "Mamba decoder-only", "evaluation/bleu"),
    ]

    train_series = write_series_csv("train_loss_successful.csv", train_specs)
    bleu_series = write_series_csv("validation_bleu_successful.csv", bleu_specs)

    plot_series(
        train_series,
        "train_loss_successful.pdf",
        xlabel="Итерации, тыс.",
        ylabel="Значение функции потерь",
        step_scale=1000.0,
    )
    plot_series(
        bleu_series,
        "validation_bleu_successful.pdf",
        xlabel="Номер проверки",
        ylabel="BLEU",
        step_scale=1.0,
    )


if __name__ == "__main__":
    main()
