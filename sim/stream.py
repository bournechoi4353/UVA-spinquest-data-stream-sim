"""Generator and replay interface.

* `generate(config)` builds the whole run as arrays, plus ground truth labels.
  This is the fast path used for batch generation.
* `replay(source, speed=None)` yields `(timestamp, {channel: value})` events in
  time order, merging channels of different sample rates. It never exposes the
  labels, so a detector built on it cannot cheat.

Determinism: each channel draws its noise from its own child of
`SeedSequence(seed)`, chosen by the channel's position in the config. Adding or
removing an anomaly therefore never changes the healthy noise of any channel.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .anomalies import FAULT_INJECTORS, PHYSICAL_INJECTORS, Injection
from .channels import ChannelWorkspace, build_workspace
from .labels import make_label
from .scenario import (
    MEASUREMENT_FAULTS,
    AnomalyConfig,
    ChannelConfig,
    ScenarioConfig,
    load_scenario,
    parse_scenario,
)


@dataclass
class ChannelData:
    """The samples of one channel: integer-microsecond times and values (NaN = dropout)."""

    t_us: np.ndarray
    values: np.ndarray


@dataclass
class StreamResult:
    """A complete generated run."""

    config: ScenarioConfig
    channels: dict[str, ChannelData]
    labels: list[dict[str, Any]] = field(default_factory=list)

    def timestamp(self, t_us: int) -> datetime:
        return self.config.start_time + timedelta(microseconds=int(t_us))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _processing_order(cfg: ScenarioConfig) -> list[ChannelConfig]:
    """Order channels so every source channel is built before channels coupled to it."""
    done: list[ChannelConfig] = []
    names: set[str] = set()
    pending = list(cfg.channels)
    while pending:
        for ch in pending:
            if ch.couples_to is None or ch.couples_to in names:
                done.append(ch)
                names.add(ch.name)
                pending.remove(ch)
                break
        else:  # pragma: no cover (validation rejects coupling loops)
            raise RuntimeError("coupling loop")
    return done


def generate(config: ScenarioConfig | dict[str, Any]) -> StreamResult:
    """Generate the full run: data arrays for every channel plus ground truth labels."""
    cfg = parse_scenario(config) if isinstance(config, dict) else config
    child_seeds = np.random.SeedSequence(cfg.seed).spawn(len(cfg.channels))
    index_of = {ch.name: i for i, ch in enumerate(cfg.channels)}

    channels: dict[str, ChannelData] = {}
    physical_dev: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # name -> (t_us, deviation)
    injections: list[tuple[int, AnomalyConfig, Injection]] = []  # (config order, anomaly, injection)

    for ch in _processing_order(cfg):
        rng = np.random.default_rng(child_seeds[index_of[ch.name]])
        if ch.couples_to is not None:
            src_t, src_dev = physical_dev[ch.couples_to]
            ws = build_workspace(cfg, ch, rng, src_t, src_dev, cfg.channel(ch.couples_to))
        else:
            ws = build_workspace(cfg, ch, rng)

        mine = [(i, a) for i, a in enumerate(cfg.anomalies) if a.channel == ch.name]
        for i, anomaly in mine:
            if anomaly.type not in MEASUREMENT_FAULTS:
                injections.append((i, anomaly, PHYSICAL_INJECTORS[anomaly.type](ws, anomaly)))

        physical = ws.values()
        physical_dev[ch.name] = (ws.t_us, ws.deviation())

        observed = physical.copy()
        for fault_type in FAULT_INJECTORS:  # stuck_sensor first, then dropout
            for i, anomaly in mine:
                if anomaly.type == fault_type:
                    injections.append((i, anomaly, FAULT_INJECTORS[fault_type](observed, physical, ws, anomaly)))
        channels[ch.name] = ChannelData(t_us=ws.t_us, values=observed)

    # Label ids follow time order (ties broken by position in the config).
    injections.sort(key=lambda item: (item[2].start_us, item[0]))
    labels = [
        make_label(f"ev_{n:04d}", cfg.channel(anomaly.channel), anomaly, injection, cfg.start_time)
        for n, (_, anomaly, injection) in enumerate(injections, start=1)
    ]
    # Keep channels in config order for stable output.
    channels = {ch.name: channels[ch.name] for ch in cfg.channels}
    return StreamResult(config=cfg, channels=channels, labels=labels)


# ---------------------------------------------------------------------------
# Merging channels by timestamp
# ---------------------------------------------------------------------------
def merge_samples(channels: dict[str, ChannelData]) -> Iterator[tuple[int, dict[str, float]]]:
    """Yield (t_us, {channel: value}) in time order.

    A channel appears in an event only when it has a sample at that time. A
    dropout sample is present with the value NaN (the sample exists, the
    reading is missing).
    """
    names = list(channels)
    all_t = np.unique(np.concatenate([channels[n].t_us for n in names]))
    pos = dict.fromkeys(names, 0)
    for t in all_t:
        event: dict[str, float] = {}
        for name in names:
            data = channels[name]
            p = pos[name]
            if p < len(data.t_us) and data.t_us[p] == t:
                event[name] = float(data.values[p])
                pos[name] = p + 1
        yield int(t), event


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
def _events_from_csv(path: Path) -> Iterator[tuple[datetime, dict[str, float]]]:
    from .io import read_data_csv  # local import: io.py imports from this module

    yield from read_data_csv(path)


def replay(
    source: ScenarioConfig | dict[str, Any] | str | Path,
    speed: float | None = None,
) -> Iterator[tuple[datetime, dict[str, float]]]:
    """Replay a run as a stream of `(timestamp, {channel: value})` events.

    `source` may be a ScenarioConfig, a config dict, a path to a scenario YAML
    (the data is generated), or a path to a run folder or its data.csv (the
    data is read back; labels.json is never opened).

    `speed=None` yields as fast as possible. A float plays in real time times
    that factor (speed=60 means one simulated minute per real second), using
    time.sleep between events.

    Contract: events come in strictly increasing time order; a channel appears
    only when it has a sample at that time; a missing reading is NaN; no label
    information is ever exposed.
    """
    if speed is not None and not speed > 0:
        raise ValueError(f"speed must be a positive number or None, got {speed!r}")

    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix in (".yaml", ".yml"):
            events = _events_from_config(load_scenario(path))
        else:
            data_path = path / "data.csv" if path.is_dir() else path
            events = _events_from_csv(data_path)
    else:
        events = _events_from_config(parse_scenario(source) if isinstance(source, dict) else source)

    previous: datetime | None = None
    for timestamp, event in events:
        if speed is not None and previous is not None:
            time.sleep((timestamp - previous).total_seconds() / speed)
        previous = timestamp
        yield timestamp, event


def _events_from_config(cfg: ScenarioConfig) -> Iterator[tuple[datetime, dict[str, float]]]:
    result = generate(cfg)  # labels stay inside this function and are never yielded
    for t_us, event in merge_samples(result.channels):
        yield result.timestamp(t_us), event
