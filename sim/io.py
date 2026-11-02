"""Writers and readers for run folders.

A run folder holds three files:

* data.csv          timestamp column, then one column per channel
* labels.json       the ground truth label records
* config_used.yaml  the exact config (including the seed) that made the run

In data.csv an EMPTY cell means the channel has no sample at that time. A
dropout (a sample whose reading is missing) is written as the text `nan`, so
the two cases stay distinguishable.

Output is deterministic: fixed number formatting, fixed key order, "\\n" line
endings, so the same seed and config give byte-identical files.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import yaml

from .stream import StreamResult, merge_samples

VALUE_FORMAT = ".10g"


def _format_value(value: float) -> str:
    return "nan" if value != value else format(value, VALUE_FORMAT)


def write_data_csv(result: StreamResult, path: Path) -> None:
    names = list(result.channels)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["timestamp", *names])
        for t_us, event in merge_samples(result.channels):
            row = [result.timestamp(t_us).isoformat()]
            row += [_format_value(event[n]) if n in event else "" for n in names]
            writer.writerow(row)


def write_labels(labels: list[dict[str, Any]], path: Path) -> None:
    path.write_text(json.dumps(labels, indent=2) + "\n")


def write_config(raw_config: dict[str, Any], path: Path) -> None:
    path.write_text(yaml.safe_dump(raw_config, sort_keys=False))


def write_run(result: StreamResult, out_dir: str | Path) -> Path:
    """Write data.csv, labels.json and config_used.yaml into `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_data_csv(result, out_dir / "data.csv")
    write_labels(result.labels, out_dir / "labels.json")
    write_config(result.config.raw, out_dir / "config_used.yaml")
    return out_dir


def read_data_csv(path: str | Path) -> Iterator[tuple[datetime, dict[str, float]]]:
    """Read data.csv back as replay-style events (empty cells are left out)."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        names = header[1:]
        for row in reader:
            event = {name: float(cell) for name, cell in zip(names, row[1:]) if cell != ""}
            yield datetime.fromisoformat(row[0]), event


def read_labels(path: str | Path) -> list[dict[str, Any]]:
    """Read labels.json. This is the ONLY way to get ground truth from disk."""
    return json.loads(Path(path).read_text())
