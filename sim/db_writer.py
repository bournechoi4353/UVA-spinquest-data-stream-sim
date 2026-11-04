"""Write a simulated wide-feed run into a local SQLite file shaped like ptdb (Phase R2).

Each included feed becomes a table with the real table and column names:

* ``id INTEGER PRIMARY KEY``: assigned in insertion order (1, 2, 3, ...).
* the time column as ``TEXT`` in UTC 'YYYY-MM-DD HH:MM:SS'. SQLite has no
  timestamp type, and this is exactly what MySQL returns when the session
  time zone is '+00:00'. The column has an index, as in ptdb.
* value columns as ``REAL``, NULL allowed.

Stress options (for testing the future ingestion reader against ING-6) are
applied in a fixed order with their own seed stream, so the same options and
seed always give the same file:

1. ``null_fraction``: that share of value cells becomes NULL.
2. ``dup_fraction``: that share of rows is inserted a second time, a few rows
   later, with a new id.
3. ``late_fraction``: that share of rows is moved later in insertion order (by
   1 to ``MAX_LATE_ROWS`` rows), so its id is larger than the ids of rows with
   later timestamps. Real ptdb rows probably rarely arrive out of order,
   because ``Timestamp`` is set on insert, so treat this as a stress test.

Read the fixture exactly as the real database will be read, paging by id::

    SELECT id, Timestamp, buffer_pressure, ... FROM Labjack WHERE id > ? ORDER BY id LIMIT 5000
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .layout_scenario import Feed, LayoutRun, sql_to_us, us_to_sql

MAX_LATE_ROWS = 10
MAX_DUP_DELAY_ROWS = 5
DEFAULT_CHUNK_ROWS = 5000


@dataclass
class StressCounts:
    nulls: int = 0
    duplicates: int = 0
    late: int = 0


def _q(name: str) -> str:
    """Quote an identifier for SQLite (names come from the registry, never from users)."""
    return '"' + name.replace('"', '""') + '"'


def _stress_rng(seed: int, feed_index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(2_000_000 + feed_index,)))


def _rows_with_stress(feed: Feed, rng: np.random.Generator, null_fraction: float,
                      dup_fraction: float, late_fraction: float) -> tuple[list[tuple], StressCounts]:
    counts = StressCounts()
    n = feed.n_rows
    values = np.column_stack([feed.values[c] for c in feed.columns]) if feed.columns else np.empty((n, 0))
    values = values.copy()
    if null_fraction > 0 and values.size:
        mask = rng.random(values.shape) < null_fraction
        counts.nulls = int((mask & ~np.isnan(values)).sum())
        values[mask] = np.nan

    rows = [(us_to_sql(int(feed.t_us[i])), *(None if v != v else float(v) for v in values[i]))
            for i in range(n)]

    if dup_fraction > 0 and n:
        picks = np.nonzero(rng.random(n) < dup_fraction)[0]
        delays = rng.integers(1, MAX_DUP_DELAY_ROWS + 1, len(picks))
        # Insert from the back so earlier positions stay valid.
        for i, d in sorted(zip(picks.tolist(), delays.tolist()), reverse=True):
            rows.insert(min(i + 1 + d, len(rows)), rows[i])
        counts.duplicates = len(picks)

    if late_fraction > 0 and rows:
        m = len(rows)
        picks = np.nonzero(rng.random(m) < late_fraction)[0]
        delays = rng.integers(1, MAX_LATE_ROWS + 1, len(picks))
        for i, d in sorted(zip(picks.tolist(), delays.tolist()), reverse=True):
            row = rows.pop(i)
            rows.insert(min(i + d, len(rows)), row)
        counts.late = len(picks)
    return rows, counts


def write_fixture(run: LayoutRun, path: str | Path, null_fraction: float = 0.0,
                  dup_fraction: float = 0.0, late_fraction: float = 0.0,
                  seed: int | None = None) -> dict[str, StressCounts]:
    """Write every feed of ``run`` into a new SQLite file at ``path``.

    ``seed`` defaults to the run's seed. An existing file is replaced.
    Returns the number of NULLs, duplicates and late rows added per table.
    """
    for name, frac in (("null_fraction", null_fraction), ("dup_fraction", dup_fraction),
                       ("late_fraction", late_fraction)):
        if not 0.0 <= frac < 0.5:
            raise ValueError(f"{name} must be in [0, 0.5), got {frac}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    seed = run.meta["seed"] if seed is None else seed

    out: dict[str, StressCounts] = {}
    con = sqlite3.connect(path)
    try:
        for fi, (name, feed) in enumerate(run.feeds.items()):
            cols = ", ".join(f"{_q(c)} REAL" for c in feed.columns)
            con.execute(f"CREATE TABLE {_q(name)} (id INTEGER PRIMARY KEY, "
                        f"{_q(feed.time_column)} TEXT NOT NULL, {cols})")
            con.execute(f"CREATE INDEX {_q(name + '_' + feed.time_column)} "
                        f"ON {_q(name)} ({_q(feed.time_column)})")
            rows, counts = _rows_with_stress(feed, _stress_rng(seed, fi), null_fraction,
                                             dup_fraction, late_fraction)
            placeholders = ", ".join("?" * (1 + len(feed.columns)))
            names = ", ".join(_q(c) for c in [feed.time_column, *feed.columns])
            con.executemany(f"INSERT INTO {_q(name)} ({names}) VALUES ({placeholders})", rows)
            out[name] = counts
        con.commit()
    finally:
        con.close()
    return out


def read_fixture(path: str | Path, table: str, chunk_rows: int = DEFAULT_CHUNK_ROWS) -> pd.DataFrame:
    """Read one table back, paging by id with the same query shape the real reader will use.

    Returns a DataFrame with ``id``, ``t_us`` (UTC microseconds) and the value
    columns (NULL becomes NaN), ordered by id.
    """
    con = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
    try:
        info = con.execute(f"PRAGMA table_info({_q(table)})").fetchall()
        columns = [row[1] for row in info]
        if not columns:
            raise KeyError(f"no table {table!r} in {path}")
        select = ", ".join(_q(c) for c in columns)
        frames, last = [], 0
        while True:
            chunk = con.execute(f"SELECT {select} FROM {_q(table)} WHERE id > ? ORDER BY id LIMIT ?",
                                (last, chunk_rows)).fetchall()
            if not chunk:
                break
            frames.append(pd.DataFrame(chunk, columns=columns))
            last = chunk[-1][0]
    finally:
        con.close()
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    time_col = columns[1]
    df.insert(1, "t_us", [sql_to_us(s) for s in df[time_col]])
    df = df.drop(columns=[time_col])
    for c in columns[2:]:
        df[c] = df[c].astype(float)
    return df
