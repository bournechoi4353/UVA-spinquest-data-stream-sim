"""A simulated LIVE ptdb: rows of a SIMULATED run are inserted into a SQLite file as time passes.

The real logger has been off since 2026-08-17, so there is no live data to watch. This stands in
for it. It writes the same tables, columns and quirks as ptdb (via ``sim.db_writer``):
partial rows with NULLs, a few late and duplicate rows, measured feed periods. Planted anomalies
with known answers go in the live part. The monitor follows the file exactly as it would follow
ptdb (``run_monitor.py --sqlite FILE --follow``), so the whole live path can be tested:
polling, reordering, the state store rewriting summary.txt, and later the Discord bot.

* History: rows before ``live_start`` are inserted at once, so the monitor has healthy data
  to calibrate on.
* Live: each row is inserted when the simulated clock reaches its insert time. The clock runs at
  ``speed`` times wall time (1 = real time).
* A late row is inserted when the rows around it are, so its timestamp is earlier than rows that
  are already in the table, as with real late rows.

Ground truth (labels.json) is written next to the database for checking afterwards. The monitor
never reads it. ``score_alarms`` compares the monitor's alarms.jsonl with it.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .db_writer import _q, _rows_with_stress, _stress_rng
from .layout_scenario import LayoutRun, datetime_to_us, sql_to_us, us_to_sql

UTC = timezone.utc


@dataclass
class PendingRow:
    insert_us: int
    table: str
    row: tuple


@dataclass
class LiveWriter:
    """Inserts the rows of ``run`` into ``path`` as the simulated clock passes their insert times."""

    run: LayoutRun
    path: Path
    seed: int = 0
    dup_fraction: float = 0.01
    late_fraction: float = 0.01
    pending: list[PendingRow] = field(default_factory=list)
    written: int = 0

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(f"{self.path} exists; remove it or pick another --out")
        con = sqlite3.connect(self.path)
        try:
            con.execute("PRAGMA journal_mode=WAL")      # readers can read while rows are written
            for fi, (name, feed) in enumerate(self.run.feeds.items()):
                cols = ", ".join(f"{_q(c)} REAL" for c in feed.columns)
                con.execute(f"CREATE TABLE {_q(name)} (id INTEGER PRIMARY KEY, "
                            f"{_q(feed.time_column)} TEXT NOT NULL, {cols})")
                con.execute(f"CREATE INDEX {_q(name + '_' + feed.time_column)} "
                            f"ON {_q(name)} ({_q(feed.time_column)})")
                rows, _ = _rows_with_stress(feed, _stress_rng(self.seed, fi), 0.0,
                                            self.dup_fraction, self.late_fraction)
                newest = None
                for r in rows:
                    t = sql_to_us(r[0])
                    newest = t if newest is None else max(newest, t)
                    self.pending.append(PendingRow(newest, name, r))
            con.commit()
        finally:
            con.close()
        order = sorted(range(len(self.pending)), key=lambda i: self.pending[i].insert_us)
        self.pending = [self.pending[i] for i in order]
        self._cols = {name: [feed.time_column, *feed.columns] for name, feed in self.run.feeds.items()}

    @property
    def done(self) -> bool:
        return self.written >= len(self.pending)

    def write_until(self, t_us: int) -> int:
        """Insert every row whose insert time is at or before ``t_us``. Returns how many were written."""
        start = self.written
        end = start
        while end < len(self.pending) and self.pending[end].insert_us <= t_us:
            end += 1
        if end == start:
            return 0
        con = sqlite3.connect(self.path, timeout=30)
        try:
            for p in self.pending[start:end]:
                cols = self._cols[p.table]
                con.execute(f"INSERT INTO {_q(p.table)} ({', '.join(_q(c) for c in cols)}) "
                            f"VALUES ({', '.join('?' * len(cols))})", p.row)
            con.commit()
        finally:
            con.close()
        self.written = end
        return end - start


def run_live(writer: LiveWriter, live_start_us: int, speed: float, tick_s: float = 0.5,
             announce: Callable[[str], None] | None = None, labels: list[dict[str, Any]] | None = None,
             sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> None:
    """Write the history at once, then the live part at ``speed`` times wall time."""
    n = writer.write_until(live_start_us)
    if announce:
        announce(f"history written: {n} rows up to {us_to_sql(live_start_us)} UTC; live from now at {speed:g}x")
    todo = sorted(((datetime_to_us(_t(lb["start"])), lb) for lb in (labels or [])), key=lambda x: x[0])
    todo = [(t, lb) for t, lb in todo if t >= live_start_us]
    wall0 = clock()
    while not writer.done:
        now_us = live_start_us + int((clock() - wall0) * speed * 1e6)
        writer.write_until(now_us)
        while todo and todo[0][0] <= now_us:
            _, lb = todo.pop(0)
            if announce:
                announce(f"[sim {us_to_sql(now_us - now_us % 1_000_000)} UTC] planted now: {lb['type']} on {lb['channel']}")
        if not writer.done:
            sleep(tick_s)
    if announce:
        announce(f"simulation finished: {writer.written} rows written; the last row is at "
                 f"{us_to_sql(writer.pending[-1].insert_us)} UTC")


def _t(text: str | None) -> datetime | None:
    if not text:
        return None
    t = datetime.fromisoformat(text)
    return t if t.tzinfo else t.replace(tzinfo=UTC)


def score_alarms(alarms: list[dict[str, Any]], labels: list[dict[str, Any]], live_start: datetime,
                 end: datetime | None = None) -> dict[str, Any]:
    """Planted anomalies found or missed, and false alarms, from run_monitor.py's alarms.jsonl.

    Alarms raised before ``live_start`` (the history the monitor calibrated on) are ignored.
    A detection matches a label on the same channel (or feed) raised between the label's start and
    GRACE_S after its end, preferring a compatible kind, as in the end-to-end test.
    """
    from monitor.scoring import COMPATIBLE, GRACE_S

    live = [a for a in alarms if _t(a["detected_at"]) >= live_start]
    found: dict[str, dict[str, Any]] = {}
    false: list[dict[str, Any]] = []
    for a in live:
        at = _t(a["detected_at"])
        hits = [lb for lb in labels if lb["channel"] == a["subject"] and _t(lb["start"]) <= at
                and (at - (_t(lb.get("end")) or end or at)).total_seconds() <= GRACE_S]
        good = [lb for lb in hits if lb["type"] in COMPATIBLE.get(a["kind"], set())] or hits
        if good:
            lb = good[0]
            if lb["id"] not in found:
                fd = _t(lb.get("first_detectable"))
                found[lb["id"]] = {"type": lb["type"], "channel": lb["channel"], "kind": a["kind"],
                                   "delay_s": None if fd is None else round((at - fd).total_seconds(), 1)}
        else:
            false.append({"subject": a["subject"], "kind": a["kind"], "at": a["detected_at"]})
    planted = [lb for lb in labels if _t(lb["start"]) >= live_start and (end is None or _t(lb["start"]) <= end)]
    missed = [{"type": lb["type"], "channel": lb["channel"], "start": lb["start"]}
              for lb in planted if lb["id"] not in found]
    return {"planted": len(planted), "found": [found[k] for k in sorted(found)], "missed": missed,
            "false_alarms": false}


def write_ground_truth(folder: Path, run: LayoutRun, live_start_us: int, info: dict[str, Any]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "labels.json").write_text(json.dumps(run.labels, indent=2) + "\n", encoding="utf-8")
    meta = {"synthetic": True, "live_start": us_to_sql(live_start_us), **info}
    (folder / "live_info.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def floor_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)

