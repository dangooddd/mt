from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "data" / "runs"
CSV_DIR = ROOT / "report" / "csv"


def safe_name(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip("_")


def main() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    rows_by_tag: dict[str, list[tuple[str, str, int, float, float, str]]] = defaultdict(list)

    for run_dir in sorted(path for path in RUNS_DIR.iterdir() if path.is_dir()):
        for event_file in sorted(run_dir.glob("events.out.tfevents*")):
            accumulator = event_accumulator.EventAccumulator(
                str(event_file), size_guidance={"scalars": 0}
            )
            accumulator.Reload()
            tags = [tag for tag in accumulator.Tags().get("scalars", []) if tag.startswith("evaluation/")]
            for tag in sorted(tags):
                for index, event in enumerate(accumulator.Scalars(tag), start=1):
                    step = event.step if event.step > 0 else index
                    value = float(event.value)
                    if not math.isfinite(value):
                        continue
                    rows_by_tag[tag].append(
                        (run_dir.name, tag, step, value, float(event.wall_time), event_file.name)
                    )

    for tag, rows in sorted(rows_by_tag.items()):
        rows.sort(key=lambda row: (row[0], row[5], row[2], row[4]))
        path = CSV_DIR / f"{safe_name(tag)}.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["run", "tag", "step", "value", "wall_time", "event_file"])
            writer.writerows(rows)
        print(f"{path}: {len(rows)} rows")


if __name__ == "__main__":
    main()
