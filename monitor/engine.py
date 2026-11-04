"""The engine: events in, detectors run, state store updated.

    engine = Engine.from_healthy(healthy_events, feeds=..., couplings=...)
    for t, values in events:
        engine.step(t, values)
    print(build_summary(engine.store))

Calibration needs healthy data. ``Engine.from_healthy`` takes the first part
of a stream (or a separate healthy run) as the calibration window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Iterator

import numpy as np

from .detectors import (Calibration, ChannelMonitor, Condition, CouplingWatch, DetectorSettings, FeedWatch,
                        adapt_thresholds, calibrate)
from .state import StateStore


@dataclass
class Coupling:
    channel: str
    source: str
    gain: float


class Engine:
    def __init__(self, cals: dict[str, Calibration], feeds: dict[str, list[str]] | None = None,
                 couplings: list[Coupling] | None = None, settings: DetectorSettings | None = None,
                 display: dict[str, tuple[str, str | None]] | None = None):
        self.s = settings or DetectorSettings()
        self.store = StateStore()
        self.monitors = {k: ChannelMonitor(k, c, self.s) for k, c in cals.items()}
        self.feed_of: dict[str, str] = {}
        self.feed_members: dict[str, list[str]] = {f: list(keys) for f, keys in (feeds or {}).items()}
        self.feed_watch: dict[str, FeedWatch] = {}
        for feed, keys in (feeds or {}).items():
            period = float(np.mean([cals[k].period_s for k in keys if k in cals])) if keys else 1.0
            self.feed_watch[feed] = FeedWatch(feed, period, self.s)
            self.store.add_feed(feed, period)
            for k in keys:
                self.feed_of[k] = feed
        display = display or {}
        for k, c in cals.items():
            name, unit = display.get(k, (k, None))
            self.store.add_channel(k, c, name, unit, self.feed_of.get(k))
        self.couplings: dict[str, CouplingWatch] = {}
        self.sources: dict[str, list[CouplingWatch]] = {}
        for cp in couplings or []:
            if cp.channel in cals and cp.source in cals:
                w = CouplingWatch(cp.channel, cp.source, cp.gain, cals[cp.channel], cals[cp.source], self.s)
                self.couplings[cp.channel] = w
                self.sources.setdefault(cp.source, []).append(w)
        self.raised: list[Condition] = []
        self.last_value_at: dict[str, datetime] = {}
        self.missing_wide: dict[str, Condition] = {}

    @classmethod
    def from_healthy(cls, events: Iterable[tuple[datetime, dict[str, float]]], adaptive: bool = True,
                     **kw) -> "Engine":
        """Calibrate every channel from healthy events.

        ``adaptive=True`` (the default) sets per-channel CUSUM and noise
        thresholds from the healthy data; ``adaptive=False`` uses the plain
        Phase 5 thresholds (5 sigma CUSUM) for every channel.
        """
        settings = kw.get("settings") or DetectorSettings()
        vals: dict[str, list[float]] = {}
        times: dict[str, list[float]] = {}
        healthy = list(events)
        for t, ev in healthy:
            ts = t.timestamp()
            for k, v in ev.items():
                vals.setdefault(k, []).append(v)
                times.setdefault(k, []).append(ts)
        cals = {k: calibrate(np.array(vals[k]), np.array(times[k])) for k in vals if len(vals[k]) >= 10}
        if adaptive:
            for k, c in cals.items():
                adapt_thresholds(np.array(vals[k]), c, settings)
        engine = cls(cals, **kw)
        if adaptive:
            engine._calibrate_couplings(healthy)
        engine._calibrate_missing(healthy)
        return engine

    def _calibrate_missing(self, healthy: list[tuple[datetime, dict[str, float]]]) -> None:
        """How long a wide-feed channel may go without a value before it counts as missing.

        Real feeds have partial rows: a channel is absent from a share q of its feed's rows even
        when healthy. A run of n absent rows has probability about q^n, so n is chosen to give about
        one false alarm per channel per FALSE_ALARM_DAYS, times a safety factor of 1.5, and never
        less than the gap-based limit from calibrate().
        """
        from .detectors import FALSE_ALARM_DAYS
        for feed, keys in self.feed_members.items():
            rows = 0
            present = dict.fromkeys(keys, 0)
            for _, ev in healthy:
                if any(k in ev for k in keys):
                    rows += 1
                    for k in keys:
                        if k in ev:
                            present[k] += 1
            if rows < 100:
                continue
            for k in keys:
                m = self.monitors.get(k)
                if m is None:
                    continue
                q = min(max(1.0 - present[k] / rows, 1e-6), 0.99)
                per_window = FALSE_ALARM_DAYS * 86400.0 / max(m.cal.period_s, 1e-3)
                n = 1.5 * math.log(per_window) / -math.log(q)
                limit = (n + 1) * m.cal.period_s
                m.cal.missing_after_s = max(m.cal.missing_after_s or 0.0, limit)

    def _calibrate_couplings(self, healthy: list[tuple[datetime, dict[str, float]]]) -> None:
        """Residual sigma and threshold of every coupling watch from healthy events."""
        for w in self.couplings.values():
            x = None
            resid = []
            for _, ev in healthy:
                if w.source in ev and ev[w.source] == ev[w.source]:
                    x = ev[w.source]
                y = ev.get(w.channel)
                if x is not None and y is not None and y == y:
                    resid.append((y - w.cy.target) - w.gain * (x - w.cx.target))
            if len(resid) < 20:
                continue
            r = np.array(resid)
            w.offset = float(np.median(r))
            w.resid_sigma = max(1.4826 * float(np.median(np.abs(r - w.offset))), 1e-12)
            up = dn = best = 0.0
            for v in r:
                z = max(-3.0, min(3.0, (v - w.offset) / w.resid_sigma))
                up = max(0.0, up + z - 0.5)
                dn = max(0.0, dn - z - 0.5)
                best = max(best, up, dn)
            w.threshold = max(self.s.decouple_threshold, 1.5 * best)

    def _emit(self, conds: list[Condition]) -> None:
        for c in conds:
            self.store.apply(c)
            if c.active or c.kind == "spike":
                self.raised.append(c)

    def step(self, t: datetime, values: dict[str, float]) -> None:
        """Feed one event. Keys not in the calibration are ignored."""
        for feed, w in self.feed_watch.items():
            self._emit(w.tick(t))
        rows_seen = set()
        for k, v in values.items():
            m = self.monitors.get(k)
            if m is None:
                continue
            self.store.observe(k, t, v)
            self._emit(m.update(t, v))
            for w in self.sources.get(k, []):
                w.source_value(v)
            if k in self.couplings:
                self._emit(self.couplings[k].update(t, v))
            f = self.feed_of.get(k)
            if f is not None and f not in rows_seen:
                rows_seen.add(f)
                ended = self.feed_watch[f].row(t)
                if ended:   # the feed is back after an outage: restart the channels' gap timers
                    for key in self.feed_members.get(f, ()):
                        self.last_value_at[key] = t
                self._emit(ended)
                self.store.feeds[f].last_row = t
        # In a wide feed a NULL value is simply absent from the row's event, and real feeds have
        # many partial rows (more than one writer per table). So "missing" is time-based here: a
        # channel is missing when its feed keeps sending rows but the channel has had no value for
        # longer than its healthy gaps allow (Calibration.missing_after_s).
        for f in rows_seen:
            for k in self.feed_members.get(f, ()):
                m = self.monitors.get(k)
                if m is None:
                    continue
                if k in values:
                    self.last_value_at[k] = t
                    if k in self.missing_wide:
                        cond = self.missing_wide.pop(k)
                        cond.active, cond.ended_at = False, t
                        self._emit([cond])
                    continue
                last = self.last_value_at.get(k)
                limit = m.cal.missing_after_s or 10 * m.cal.period_s
                if last is not None and k not in self.missing_wide and (t - last).total_seconds() > limit:
                    cond = Condition(k, "missing", last, t, detail={"gap_s": round((t - last).total_seconds(), 1)})
                    self.missing_wide[k] = cond
                    self._emit([cond])
        self.store.tick(t)

    def run(self, events: Iterable[tuple[datetime, dict[str, float]]]) -> Iterator[datetime]:
        for t, ev in events:
            self.step(t, ev)
            yield t
