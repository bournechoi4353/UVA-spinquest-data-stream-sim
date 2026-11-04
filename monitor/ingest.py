"""Live reader for ptdb-shaped tables (ING-3, ING-6, ING-12, ING-17 to ING-19).

The reader polls each included table by its ``id`` watermark through a
profiler adapter, so every query passes the same read-only guard:

    SELECT id, Timestamp, <channels> FROM <table> WHERE id > :last_id ORDER BY id LIMIT :n

and yields ``(timestamp, {"<table>.<db_column>": value})`` events, the same
shape as ``sim.replay`` (ING-3). Timestamps are aware UTC datetimes; the
session time zone is UTC (ING-5, ING-17).

Handling of messy rows (ING-6):
* NULL values are left out of the event (a missing sample, which does not
  reset detector state).
* Duplicate rows (same table, same time, same values as a recent row) are
  dropped and counted.
* Late rows: rows are held in a reorder buffer and released in time order once
  they are ``reorder_window_s`` older than the newest row seen. A row that
  arrives after rows with later times were already released is dropped and
  counted as too late.

Ordering and paging are always by ``id`` (ING-19). The real database is only
ever read by the person running the monitor, with ``MySQLAdapter``.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator

from sim.profiler.adapters import Adapter
from sim.registry import Registry, Table

UTC = timezone.utc


@dataclass
class ReaderStats:
    rows: int = 0
    emitted: int = 0
    duplicates: int = 0
    too_late: int = 0
    nulls: int = 0
    polls: int = 0
    last_id: dict[str, int] = field(default_factory=dict)


class PtdbReader:
    def __init__(self, adapter: Adapter, registry: Registry, tables: list[str] | None = None,
                 batch_rows: int = 5000, reorder_window_s: float = 60.0, dedup_memory: int = 2000,
                 start_ids: dict[str, int] | None = None):
        self.a = adapter
        self.tables: list[Table] = ([registry.table(n) for n in tables] if tables
                                    else registry.included_tables())
        self.batch_rows = batch_rows
        self.window_us = int(reorder_window_s * 1e6)
        self.stats = ReaderStats(last_id={t.name: int((start_ids or {}).get(t.name, 0)) for t in self.tables})
        self._recent: dict[str, deque] = {t.name: deque(maxlen=dedup_memory) for t in self.tables}
        self._recent_set: dict[str, set] = {t.name: set() for t in self.tables}
        self._heap: list[tuple[int, str, int, dict[str, float]]] = []
        self._newest_us: int | None = None
        self._released_us: int | None = None
        self._frontier: dict[str, int | None] = {t.name: None for t in self.tables}

    # ---- watermarks ------------------------------------------------------------------
    def start_at_time(self, when: datetime) -> None:
        """Set each table's watermark to just before the first row at or after ``when``."""
        text = when.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")
        for t in self.tables:
            rows = self.a.execute(f"SELECT MIN(id) FROM {self.a.quote(t.name)} "
                                  f"WHERE {self.a.quote(t.time_column)} >= {self.a.param}", (text,))
            first = rows[0][0] if rows and rows[0][0] is not None else None
            if first is None:
                rows = self.a.execute(f"SELECT MAX(id) FROM {self.a.quote(t.name)}")
                first = (rows[0][0] or 0) + 1
            self.stats.last_id[t.name] = int(first) - 1

    def start_at_end(self) -> None:
        """Only new rows from now on."""
        for t in self.tables:
            rows = self.a.execute(f"SELECT MAX(id) FROM {self.a.quote(t.name)}")
            self.stats.last_id[t.name] = int(rows[0][0] or 0)

    # ---- polling -------------------------------------------------------------------------
    def _fetch(self, t: Table) -> int:
        cols = ", ".join(self.a.quote(c) for c in t.select_columns())
        rows = self.a.execute(f"SELECT {cols} FROM {self.a.quote(t.name)} WHERE id > {self.a.param} "
                              f"ORDER BY id LIMIT {int(self.batch_rows)}", (self.stats.last_id[t.name],))
        names = [c.db_column for c in t.channels]
        for r in rows:
            rid, ts = int(r[0]), self.a.to_us(r[1])
            self.stats.rows += 1
            self.stats.last_id[t.name] = rid
            values = tuple(None if v is None else float(v) for v in r[2:])
            sig = (ts, values)
            if sig in self._recent_set[t.name]:
                self.stats.duplicates += 1
                continue
            rec = self._recent[t.name]
            if len(rec) == rec.maxlen:
                self._recent_set[t.name].discard(rec[0])
            rec.append(sig)
            self._recent_set[t.name].add(sig)
            if self._released_us is not None and ts < self._released_us:
                self.stats.too_late += 1
                continue
            event = {}
            for name, v in zip(names, values):
                if v is None:
                    self.stats.nulls += 1
                else:
                    event[f"{t.name}.{name}"] = v
            heapq.heappush(self._heap, (ts, t.name, rid, event))
            self._newest_us = ts if self._newest_us is None else max(self._newest_us, ts)
            f = self._frontier[t.name]
            self._frontier[t.name] = ts if f is None else max(f, ts)
        return len(rows)

    def _release(self, flush: bool, limit_us: int | None = None) -> Iterator[tuple[datetime, dict[str, float]]]:
        if limit_us is None:
            limit_us = (self._newest_us or 0) - self.window_us
        while self._heap and (flush or self._heap[0][0] <= limit_us):
            ts, _, _, event = heapq.heappop(self._heap)
            self._released_us = ts
            self.stats.emitted += 1
            yield datetime.fromtimestamp(ts / 1e6, tz=UTC), event

    def poll(self, flush: bool = False) -> list[tuple[datetime, dict[str, float]]]:
        """Read every table up to its newest row (in batches) and return released events."""
        self.stats.polls += 1
        for t in self.tables:
            while self._fetch(t) == self.batch_rows:
                pass
        return list(self._release(flush))

    def read_all(self) -> Iterator[tuple[datetime, dict[str, float]]]:
        """Backfill: read everything after the watermarks, in batches by id, then flush.

        Tables cover time at different rates per batch, so the table that is
        furthest behind in time is read next, and rows are released only up to
        the slowest unfinished table's newest time (minus the reorder window).
        """
        active = {t.name: t for t in self.tables}
        while active:
            name = min(active, key=lambda n: (self._frontier[n] is not None, self._frontier[n] or 0))
            if self._fetch(active[name]) < self.batch_rows:
                del active[name]
            fronts = [self._frontier[n] for n in active if self._frontier[n] is not None]
            if active and len(fronts) < len(active):
                continue
            limit = (min(fronts) if fronts else (self._newest_us or 0)) - self.window_us
            yield from self._release(False, limit)
        yield from self._release(True)
