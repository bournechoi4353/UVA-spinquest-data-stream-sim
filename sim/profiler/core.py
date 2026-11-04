"""The profiler: plans the queries, reads chunks, and builds the aggregate report.

Sampling plan, per included table (no full scans, no COUNT(*)):

1. Bounds: ``SELECT MIN(id), MAX(id)`` (primary key), and the time column at
   those two ids. With --since or --until, the bounds come from the time
   index instead.
2. Chunks: ``chunks`` evenly spaced id ranges of ``chunk_rows`` consecutive
   ids. Consecutive rows are needed for spacing, autocorrelation and stuck
   values.
3. Timeline: ``timeline_probes`` evenly spaced ids, looked up by primary key
   in batches. Comparing elapsed time with the id difference locates long
   gaps (outages) between probes that the chunks would miss.
4. Correlation windows: three windows of ``window_hours`` (start, middle and
   end of the range, or one window when the range is short), read in chunks
   by id and put on a 60 s grid with last observation carried forward.

Raw rows only exist in memory, one chunk at a time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..registry import Registry, Table
from .adapters import Adapter, DryRunAdapter
from .stats import MAX_GAPS, ChannelAccumulator, TimingAccumulator, _r, correlation_pairs

BANNER = "AGGREGATES ONLY. Review before sharing."
GRID_SECONDS = 60
TIMELINE_BATCH = 500
GAP_REPORT_PERIODS = 3.0


@dataclass
class ProfileSettings:
    chunks: int = 200
    chunk_rows: int = 5000
    timeline_probes: int = 2000
    window_hours: float = 24.0
    since: datetime | None = None
    until: datetime | None = None
    tables: list[str] | None = None


def _iso(t_us: int | None) -> str | None:
    if t_us is None:
        return None
    return datetime.fromtimestamp(t_us / 1e6, tz=timezone.utc).isoformat()


class Profiler:
    def __init__(self, adapter: Adapter, registry: Registry, settings: ProfileSettings,
                 log: Callable[[str], None] = lambda s: None):
        self.a = adapter
        self.reg = registry
        self.s = settings
        self.log = log
        self.p = adapter.param

    # ---- small query helpers -------------------------------------------------
    def _tables(self) -> list[Table]:
        tables = self.reg.included_tables()
        if self.s.tables:
            unknown = sorted(set(self.s.tables) - {t.name for t in self.reg.tables})
            if unknown:
                raise ValueError(f"unknown table(s): {unknown}")
            tables = [self.reg.table(n) for n in self.s.tables]
        return tables

    def _sql_time(self, dt: datetime) -> str:
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def _bounds(self, t: Table) -> tuple[int | None, int | None]:
        q, tc = self.a.quote(t.name), self.a.quote(t.time_column)
        if self.s.since is None and self.s.until is None:
            rows = self.a.execute(f"SELECT MIN(id), MAX(id) FROM {q}")
            return (rows[0][0], rows[0][1]) if rows else (None, None)
        lo = hi = None
        if self.s.since is not None:
            rows = self.a.execute(f"SELECT MIN(id) FROM {q} WHERE {tc} >= {self.p}",
                                  (self._sql_time(self.s.since),))
            lo = rows[0][0] if rows else None
        else:
            rows = self.a.execute(f"SELECT MIN(id) FROM {q}")
            lo = rows[0][0] if rows else None
        if self.s.until is not None:
            rows = self.a.execute(f"SELECT MAX(id) FROM {q} WHERE {tc} < {self.p}",
                                  (self._sql_time(self.s.until),))
            hi = rows[0][0] if rows else None
        else:
            rows = self.a.execute(f"SELECT MAX(id) FROM {q}")
            hi = rows[0][0] if rows else None
        return lo, hi

    def _time_at(self, t: Table, row_id: int) -> int | None:
        rows = self.a.execute(f"SELECT {self.a.quote(t.time_column)} FROM {self.a.quote(t.name)} "
                              f"WHERE id = {self.p}", (int(row_id),))
        return self.a.to_us(rows[0][0]) if rows else None

    def _select_range(self, t: Table, lo: int, hi: int) -> list[tuple]:
        cols = ", ".join(self.a.quote(c) for c in t.select_columns())
        return self.a.execute(f"SELECT {cols} FROM {self.a.quote(t.name)} "
                              f"WHERE id >= {self.p} AND id < {self.p} ORDER BY id", (int(lo), int(hi)))

    def _chunk_arrays(self, rows: list[tuple], n_cols: int):
        ids = np.array([r[0] for r in rows], dtype=np.int64)
        t_us = np.array([self.a.to_us(r[1]) for r in rows], dtype=np.int64)
        vals = np.array([[np.nan if v is None else float(v) for v in r[2:]] for r in rows],
                        dtype=np.float64).reshape(len(rows), n_cols)
        return ids, t_us, vals

    # ---- the plan --------------------------------------------------------------
    def chunk_starts(self, lo: int, hi: int) -> list[int]:
        span = hi - lo + 1
        if span <= self.s.chunks * self.s.chunk_rows:
            return list(range(lo, hi + 1, self.s.chunk_rows))
        step = (span - self.s.chunk_rows) / max(self.s.chunks - 1, 1)
        return sorted({lo + int(round(k * step)) for k in range(self.s.chunks)})

    def dry_run(self) -> None:
        """Print every planned statement shape. Connects to nothing."""
        a = self.a
        assert isinstance(a, DryRunAdapter), "dry_run needs a DryRunAdapter"
        a.server_info()
        for t in self._tables():
            a.printer(f"\n  -- table {t.name}")
            self._bounds(t)
            a.printer(f"  (then the {t.time_column} at MIN(id) and MAX(id))")
            a.execute(f"SELECT {a.quote(t.time_column)} FROM {a.quote(t.name)} WHERE id = {a.param}", ("<id>",))
            a.printer(f"  (then up to {self.s.chunks} chunks of {self.s.chunk_rows} consecutive ids, "
                      f"evenly spaced between MIN(id) and MAX(id), like:)")
            cols = ", ".join(a.quote(c) for c in t.select_columns())
            a.execute(f"SELECT {cols} FROM {a.quote(t.name)} WHERE id >= {a.param} AND id < {a.param} "
                      f"ORDER BY id", ("<start>", "<start + chunk_rows>"))
            a.printer(f"  (then {self.s.timeline_probes} timeline probes by primary key, "
                      f"{TIMELINE_BATCH} ids per query, like:)")
            a.execute(f"SELECT id, {a.quote(t.time_column)} FROM {a.quote(t.name)} WHERE id IN "
                      f"({a.param}, {a.param}, ...) ORDER BY id", ("<id1>", "<id2>"))
        a.printer(f"\n  -- correlation windows: 3 windows of {self.s.window_hours:g} h, located by the time index:")
        t = self._tables()[0]
        a.execute(f"SELECT MIN(id) FROM {a.quote(t.name)} WHERE {a.quote(t.time_column)} >= {a.param}",
                  ("<window start>",))
        a.printer("  (then the window is read with the chunk query above, chunk_rows ids at a time)")

    # ---- profiling ---------------------------------------------------------------
    def _timeline(self, t: Table, lo: int, hi: int, timing: TimingAccumulator,
                  median_period_s: float | None) -> dict:
        n = min(self.s.timeline_probes, hi - lo + 1)
        if n < 2:
            return {"probes": n, "estimated_downtime_s": None}
        ids = sorted({int(round(lo + k * (hi - lo) / (n - 1))) for k in range(n)})
        found: list[tuple[int, int]] = []
        for i in range(0, len(ids), TIMELINE_BATCH):
            batch = ids[i:i + TIMELINE_BATCH]
            marks = ", ".join([self.p] * len(batch))
            rows = self.a.execute(f"SELECT id, {self.a.quote(t.time_column)} FROM {self.a.quote(t.name)} "
                                  f"WHERE id IN ({marks}) ORDER BY id", tuple(batch))
            found.extend((int(r[0]), self.a.to_us(r[1])) for r in rows)
        if len(found) < 2 or not median_period_s:
            return {"probes": len(found), "estimated_downtime_s": None}
        pid = np.array([f[0] for f in found], dtype=np.int64)
        pt = np.array([f[1] for f in found], dtype=np.int64)
        elapsed = np.diff(pt) / 1e6
        expected = np.diff(pid) * median_period_s
        excess = elapsed - expected
        # An interval counts as holding a gap when it ran over by more than 5 periods
        # and by more than 10 percent of its expected length.
        mask = (excess > 5 * median_period_s) & (excess > 0.1 * expected)
        for i in np.nonzero(mask)[0]:
            timing.offer_gap(float(excess[i]), int(pt[i]), int(pt[i + 1]), False)
        return {"probes": len(found), "estimated_downtime_s": _r(float(excess[mask].sum()))}

    def profile_table(self, t: Table) -> tuple[dict, dict[str, ChannelAccumulator]]:
        lo, hi = self._bounds(t)
        out: dict[str, Any] = {"id_min": lo, "id_max": hi}
        if lo is None or hi is None:
            out.update(row_estimate=0, id_min_is_1=None, first_ts=None, last_ts=None, channels={})
            return out, {}
        out["row_estimate"] = int(hi - lo + 1)
        out["id_min_is_1"] = bool(lo == 1)
        out["first_ts"] = _iso(self._time_at(t, lo))
        out["last_ts"] = _iso(self._time_at(t, hi))

        timing = TimingAccumulator()
        accs = {c.db_column: ChannelAccumulator() for c in t.channels}
        starts = self.chunk_starts(lo, hi)
        for k, start in enumerate(starts):
            rows = self._select_range(t, start, min(start + self.s.chunk_rows, hi + 1))
            if not rows:
                continue
            ids, t_us, vals = self._chunk_arrays(rows, len(t.channels))
            order = np.argsort(ids, kind="stable")
            timing.add_chunk(ids[order], t_us[order])
            for j, c in enumerate(t.channels):
                accs[c.db_column].add_chunk(vals[order, j])
            del rows, ids, t_us, vals
            if (k + 1) % 25 == 0:
                self.log(f"    {t.name}: {k + 1}/{len(starts)} chunks")

        sp = timing.spacing_result()
        out.update({k: sp[k] for k in ("id_step_1_share", "backward_ts_share", "duplicate_ts_share",
                                        "spacing_s", "spacing_share", "mean_spacing_s",
                                        "gaps_gt_5x_median", "long_step_share", "sampled_hours")})
        period = sp["mean_spacing_s"]
        out["timeline"] = self._timeline(t, lo, hi, timing, period)
        # Only real gaps: longer than GAP_REPORT_PERIODS mean spacings. Ordinary steps are not listed.
        # A timeline estimate is dropped when a chunk measured the same gap exactly.
        min_gap = GAP_REPORT_PERIODS * (period or 0.0)
        exact_gaps = [(s, e) for _, s, e, ex in timing.gaps if ex]
        kept = [g for g in sorted(timing.gaps, reverse=True) if g[0] > min_gap
                and (g[3] or not any(g[1] <= s and e <= g[2] for s, e in exact_gaps))]
        out["largest_gaps"] = [
            {"start": _iso(s), "end": _iso(e), "length_s": _r(length), "exact": exact}
            for length, s, e, exact in kept[:MAX_GAPS]
        ]
        out["channels"] = {c: a.result() for c, a in accs.items()}
        return out, accs

    def _window_grid(self, t: Table, start_us: int, end_us: int) -> pd.DataFrame:
        q, tc = self.a.quote(t.name), self.a.quote(t.time_column)
        start_txt = self._sql_time(datetime.fromtimestamp(start_us / 1e6, tz=timezone.utc))
        end_txt = self._sql_time(datetime.fromtimestamp(end_us / 1e6, tz=timezone.utc))
        lo = self.a.execute(f"SELECT MIN(id) FROM {q} WHERE {tc} >= {self.p}", (start_txt,))
        hi = self.a.execute(f"SELECT MAX(id) FROM {q} WHERE {tc} < {self.p}", (end_txt,))
        lo, hi = (lo[0][0] if lo else None), (hi[0][0] if hi else None)
        if lo is None or hi is None or hi < lo:
            return pd.DataFrame()
        grid = pd.date_range(pd.Timestamp(start_us, unit="us", tz="UTC"),
                             pd.Timestamp(end_us, unit="us", tz="UTC"), freq=f"{GRID_SECONDS}s")
        series: dict[str, pd.Series] = {}
        last_by_col: dict[str, list[pd.Series]] = {c.db_column: [] for c in t.channels}
        for s in range(int(lo), int(hi) + 1, self.s.chunk_rows):
            rows = self._select_range(t, s, min(s + self.s.chunk_rows, hi + 1))
            if not rows:
                continue
            _, t_us, vals = self._chunk_arrays(rows, len(t.channels))
            idx = pd.to_datetime(t_us, unit="us", utc=True)
            for j, c in enumerate(t.channels):
                sr = pd.Series(vals[:, j], index=idx).dropna()
                if len(sr):
                    # Keep only the last value per grid step: that is all LOCF needs.
                    last_by_col[c.db_column].append(sr.groupby(sr.index.floor(f"{GRID_SECONDS}s")).last())
            del rows, t_us, vals
        for c in t.channels:
            parts = last_by_col[c.db_column]
            if parts:
                s = pd.concat(parts).groupby(level=0).last().sort_index()
                series[f"{t.name}.{c.db_column}"] = s.reindex(grid, method="ffill")
        return pd.DataFrame(series, index=grid)

    def correlations(self, tables: list[Table], ranges: dict[str, tuple[int, int]]) -> dict:
        if not ranges:
            return {"windows": [], "diff_top30": [], "level_top30": []}
        first = min(r[0] for r in ranges.values())
        last = max(r[1] for r in ranges.values())
        w = int(self.s.window_hours * 3600e6)
        if last - first <= 3 * w:
            windows = [(first, last)]
        else:
            mid = first + (last - first) // 2
            windows = [(first, first + w), (mid - w // 2, mid + w // 2), (last - w, last)]
        frames = []
        for k, (a, b) in enumerate(windows):
            parts = [self._window_grid(t, a, b) for t in tables]
            parts = [p for p in parts if len(p.columns)]
            if parts:
                f = pd.concat(parts, axis=1)
                f["window"] = k
                frames.append(f)
        if not frames:
            return {"windows": [], "diff_top30": [], "level_top30": []}
        frame = pd.concat(frames)
        diff, level = correlation_pairs(frame)
        return {"windows": [{"start": _iso(a), "end": _iso(b)} for a, b in windows],
                "grid_seconds": GRID_SECONDS, "diff_top30": diff, "level_top30": level}

    def run(self) -> dict:
        report: dict[str, Any] = {"banner": BANNER, "source": self.a.dialect}
        report["server"] = self.a.server_info()
        report["sampling"] = {
            "chunks": self.s.chunks, "chunk_rows": self.s.chunk_rows,
            "timeline_probes": self.s.timeline_probes, "window_hours": self.s.window_hours,
            "since": self.s.since.isoformat() if self.s.since else None,
            "until": self.s.until.isoformat() if self.s.until else None,
        }
        tables = self._tables()
        report["tables"] = {}
        ranges = {}
        for t in tables:
            self.log(f"  profiling {t.name}")
            out, _ = self.profile_table(t)
            report["tables"][t.name] = out
            if out["first_ts"] and out["last_ts"]:
                ranges[t.name] = (int(datetime.fromisoformat(out["first_ts"]).timestamp() * 1e6),
                                  int(datetime.fromisoformat(out["last_ts"]).timestamp() * 1e6))
        self.log("  correlations")
        report["correlations"] = self.correlations(tables, ranges)
        report["statements_run"] = len(self.a.statements)
        return report
