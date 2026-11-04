"""Writers and readers for wide-feed run folders (Phase R2).

A layout run folder holds::

    feeds/<Table>.csv   id, the time column (UTC 'YYYY-MM-DD HH:MM:SS'), then the value
                        columns with their real db_column names. An empty cell is NULL.
    labels.json         ground truth (channel keys '<table>.<db_column>'; a feed_outage
                        label uses the table name as its channel)
    meta.json           seed, timing per feed, and "synthetic": true
    config_used.yaml    the layout config that made the run

Values are written with ``repr(float)``, which reads back to exactly the same
number, so a write and read round trip is exact.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .layout_scenario import Feed, FeedSettings, LayoutRun, sql_to_us, us_to_sql


def _cell(v: float) -> str:
    return "" if v != v else repr(float(v))


def write_feed_csv(feed: Feed, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["id", feed.time_column, *feed.columns])
        cols = [feed.values[c] for c in feed.columns]
        for i in range(feed.n_rows):
            w.writerow([int(feed.ids[i]), us_to_sql(int(feed.t_us[i])), *(_cell(c[i]) for c in cols)])


def write_layout_run(run: LayoutRun, out_dir: str | Path) -> Path:
    """Write a LayoutRun to a folder. Returns the folder."""
    out = Path(out_dir)
    (out / "feeds").mkdir(parents=True, exist_ok=True)
    for name, feed in run.feeds.items():
        write_feed_csv(feed, out / "feeds" / f"{name}.csv")
    (out / "labels.json").write_text(json.dumps(run.labels, indent=2) + "\n", encoding="utf-8")
    meta = {**run.meta, "time_columns": {n: f.time_column for n, f in run.feeds.items()},
            "columns": {n: f.columns for n, f in run.feeds.items()},
            "channel_models": run.channel_models}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (out / "config_used.yaml").write_text(yaml.safe_dump(run.config, sort_keys=False), encoding="utf-8")
    return out


def read_meta(run_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(run_dir) / "meta.json").read_text(encoding="utf-8"))


def read_feed_csv(path: Path, settings: FeedSettings) -> Feed:
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        rows = list(r)
    columns = header[2:]
    ids = np.array([int(row[0]) for row in rows], dtype=np.int64)
    t_us = np.array([sql_to_us(row[1]) for row in rows], dtype=np.int64)
    values = {c: np.array([float(row[2 + j]) if row[2 + j] != "" else np.nan for row in rows])
              for j, c in enumerate(columns)}
    return Feed(path.stem, header[1], columns, ids, t_us, values, settings)


def read_feeds(run_dir: str | Path) -> dict[str, Feed]:
    """Read every feeds/<Table>.csv back. labels.json is not opened."""
    run_dir = Path(run_dir)
    meta = read_meta(run_dir)
    feeds = {}
    for name, info in meta["feeds"].items():
        st = FeedSettings(info["nominal_period_s"], info["jitter_s"], info["gap_fraction"],
                          info["timestamp_resolution_s"])
        feeds[name] = read_feed_csv(run_dir / "feeds" / f"{name}.csv", st)
    return feeds


def read_layout_labels(run_dir: str | Path) -> list[dict[str, Any]]:
    """Ground truth. The only way to get labels from a layout run folder."""
    return json.loads((Path(run_dir) / "labels.json").read_text(encoding="utf-8"))
