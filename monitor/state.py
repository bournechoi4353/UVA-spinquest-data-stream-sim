"""The state store: what the detectors currently conclude, per channel and per feed.

The store is the only thing the summary builder reads. It holds:

* channels: key -> ChannelState (display name, unit, calibration, last value
  and time, and the list of active conditions)
* feeds: name -> FeedState (last row time, outage condition if any)
* history: every condition that was raised, most recent last, capped at
  HISTORY_LIMIT so a long run cannot grow without bound

It serializes to plain JSON (``to_dict`` / ``save``) so another process (the
bot) can read it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .detectors import Calibration, Condition

HISTORY_LIMIT = 500


@dataclass
class ChannelState:
    key: str
    display_name: str
    unit: str | None
    cal: Calibration
    feed: str | None = None
    last_value: float | None = None
    last_time: datetime | None = None
    active: dict[str, Condition] = field(default_factory=dict)   # kind -> condition


@dataclass
class FeedState:
    name: str
    period_s: float
    last_row: datetime | None = None
    outage: Condition | None = None


class StateStore:
    def __init__(self) -> None:
        self.channels: dict[str, ChannelState] = {}
        self.feeds: dict[str, FeedState] = {}
        self.history: list[Condition] = []
        self.now: datetime | None = None

    def add_channel(self, key: str, cal: Calibration, display_name: str | None = None,
                    unit: str | None = None, feed: str | None = None) -> None:
        self.channels[key] = ChannelState(key, display_name or key, unit, cal, feed)

    def add_feed(self, name: str, period_s: float) -> None:
        self.feeds[name] = FeedState(name, period_s)

    def observe(self, key: str, t: datetime, value: float) -> None:
        ch = self.channels[key]
        if value == value:
            ch.last_value, ch.last_time = value, t
        self.now = t if self.now is None or t > self.now else self.now

    def apply(self, cond: Condition) -> None:
        """Record a condition that was raised or ended."""
        if cond.kind == "feed_outage":
            fs = self.feeds[cond.subject]
            fs.outage = cond if cond.active else None
        else:
            ch = self.channels[cond.subject]
            if cond.active:
                ch.active[cond.kind] = cond
            else:
                ch.active.pop(cond.kind, None)
        if cond.active or cond.kind == "spike":
            if not self.history or self.history[-1] is not cond:
                self.history.append(cond)
                del self.history[:-HISTORY_LIMIT]

    def tick(self, now: datetime) -> None:
        self.now = now if self.now is None or now > self.now else self.now

    # ---- queries used by the summary ------------------------------------------------
    def abnormal_channels(self) -> list[ChannelState]:
        return [c for c in self.channels.values() if c.active]

    def recent_spikes(self, within_s: float) -> list[Condition]:
        if self.now is None:
            return []
        return [c for c in self.history if c.kind == "spike"
                and (self.now - c.detected_at).total_seconds() <= within_s]

    def to_dict(self) -> dict[str, Any]:
        def iso(t):
            return None if t is None else t.isoformat()
        return {
            "now": iso(self.now),
            "channels": {k: {"display_name": c.display_name, "unit": c.unit, "feed": c.feed,
                             "target": c.cal.target, "sigma": c.cal.sigma, "period_s": c.cal.period_s,
                             "last_value": c.last_value, "last_time": iso(c.last_time),
                             "active": [x.to_dict() for x in c.active.values()]}
                         for k, c in self.channels.items()},
            "feeds": {k: {"period_s": f.period_s, "last_row": iso(f.last_row),
                          "outage": f.outage.to_dict() if f.outage else None} for k, f in self.feeds.items()},
            "history": [c.to_dict() for c in self.history[-50:]],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
