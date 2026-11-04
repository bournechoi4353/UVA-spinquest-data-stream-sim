"""Wide feeds shaped like the real ptdb tables (Phase R2).

A *feed* is one database table: all of its channels share one timestamp series,
and one row carries every channel's value at that time. This module builds a
simulated run over the included feeds of the registry
(``configs/ptdb_layout.yaml``).

How it works
------------
1. Every included channel becomes a channel of the existing simulator, named
   by its key ``<table>.<db_column>`` and sampled at its feed's nominal period.
   All channels of a feed therefore share the same nominal sample times, and
   every existing anomaly type (and cross-feed coupling) works unchanged.
2. Realistic, healthy irregularity is then applied per feed, from its own seed
   stream: a small positive insert-time jitter (``jitter_s``), flooring to
   ``timestamp_resolution_s`` (1 s in ptdb, so close rows can share a
   timestamp), and occasional single-row gaps (``gap_fraction``).
3. ``feed_outage`` anomalies remove every row of a feed inside a window. The
   rows are absent, not NaN.
4. Values are rounded to single precision, because ptdb stores ``float``.
   A dropout (NaN) becomes a NULL.
5. Labels are moved onto the realized row times: a label time becomes the
   realized time of the first surviving row at or after it.

Rows are numbered by ``id`` in time order. ``Timestamp`` is the insert time.

Config format (YAML or dict)::

    seed: 1
    start_time: "2026-01-01T00:00:00"     # UTC
    duration_seconds: 21600
    layout: configs/ptdb_layout.yaml      # optional, this is the default
    default_channel: {baseline: 100.0, sigma: 1.0}
    feeds:                                # optional per-feed overrides
      Labjack: {nominal_period_s: 1.79, jitter_s: 0.09, gap_fraction: 0.0005}
    channels:                             # optional per-channel overrides
      Labjack.buffer_pressure: {baseline: 50.0, sigma: 0.2, ar1_phi: 0.5, null_fraction: 0.5}
    couplings:                            # optional, default none
      - {channel: Labjack.magnet_pressure, source: Flow_Rates.magnet_flow, gain: 0.1}
    anomalies:
      - {channel: Labjack.buffer_pressure, type: spike, start: 600, magnitude: 8}
      - {feed: Flow_Rates, type: feed_outage, start: 3600, duration: 900}

Every run made here is synthetic and is marked ``synthetic: true``.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml

from .registry import Registry, Table, load_registry
from .scenario import ScenarioError, parse_scenario
from .stream import generate

DEFAULT_LAYOUT = Path(__file__).resolve().parent.parent / "configs" / "ptdb_layout.yaml"
DEFAULT_BASELINE = 100.0
DEFAULT_SIGMA = 1.0
DEFAULT_JITTER_FRACTION = 0.05     # jitter_s default: 5 percent of the period
MAX_JITTER_FRACTION = 0.5          # jitter must stay below half a period so rows keep their order
DEFAULT_GAP_FRACTION = 0.0005      # share of rows dropped as healthy single-row gaps
OUTAGE_DETECT_PERIODS = 3.0        # a feed outage is detectable once the gap exceeds 3 periods

_TOP_KEYS = {"seed", "start_time", "duration_seconds", "layout", "default_channel", "feeds",
             "channels", "couplings", "anomalies", "synthetic", "notes"}
_CHANNEL_KEYS = {"baseline", "sigma", "ar1_phi", "slow_wander", "null_fraction"}
_FEED_KEYS = {"nominal_period_s", "jitter_s", "gap_fraction", "timestamp_resolution_s"}
_COUPLING_KEYS = {"channel", "source", "gain"}
_OUTAGE_KEYS = {"feed", "type", "start", "duration"}

UTC = timezone.utc


@dataclass
class FeedSettings:
    """Timing settings of one feed after defaults and overrides."""

    period_s: float
    jitter_s: float
    gap_fraction: float
    resolution_s: float | None


@dataclass
class Feed:
    """The rows of one table. Times are integer microseconds since the Unix epoch (UTC)."""

    table: str
    time_column: str
    columns: list[str]                  # value columns, in registry order
    ids: np.ndarray                     # int64, 1..n
    t_us: np.ndarray                    # realized insert time of each row
    values: dict[str, np.ndarray]       # db_column -> float64 (float32-rounded), NaN = NULL
    settings: FeedSettings

    @property
    def n_rows(self) -> int:
        return len(self.ids)

    def timestamps(self) -> list[datetime]:
        return [us_to_datetime(t) for t in self.t_us]


@dataclass
class LayoutRun:
    """A complete simulated run over the included feeds."""

    config: dict[str, Any]              # the config as given (after defaults are filled in)
    feeds: dict[str, Feed]
    labels: list[dict[str, Any]]
    channel_models: dict[str, dict[str, Any]]   # key -> healthy model used (baseline, sigma, ...)
    start_us: int
    end_us: int
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def datetime_to_us(dt: datetime) -> int:
    """UTC microseconds since the epoch. A naive datetime is taken as UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    delta = dt - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def us_to_datetime(t_us: int) -> datetime:
    """Inverse of datetime_to_us: an aware UTC datetime."""
    return _EPOCH + timedelta(microseconds=int(t_us))


def us_to_iso(t_us: int | None) -> str | None:
    return None if t_us is None else us_to_datetime(t_us).isoformat()


def us_to_sql(t_us: int) -> str:
    """'YYYY-MM-DD HH:MM:SS' (with fractional seconds only when present), as MySQL shows UTC."""
    dt = us_to_datetime(t_us)
    text = dt.strftime("%Y-%m-%d %H:%M:%S")
    return text if dt.microsecond == 0 else f"{text}.{dt.microsecond:06d}"


def sql_to_us(text: str) -> int:
    """Parse 'YYYY-MM-DD HH:MM:SS[.ffffff]' as UTC."""
    return datetime_to_us(datetime.fromisoformat(text).replace(tzinfo=UTC))


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------
def _check_keys(d: Any, allowed: set[str], where: str) -> None:
    if not isinstance(d, dict):
        raise ScenarioError(f"{where}: expected a mapping, got {type(d).__name__}")
    unknown = sorted(set(d) - allowed)
    if unknown:
        raise ScenarioError(f"{where}: unknown key(s) {unknown}. Allowed keys: {sorted(allowed)}")


def _number(value: Any, where: str, minimum: float | None = None, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ScenarioError(f"{where}: expected a finite number, got {value!r}")
    if minimum is not None and (value < minimum or (strict and value == minimum)):
        raise ScenarioError(f"{where}: must be {'>' if strict else '>='} {minimum}, got {value!r}")
    return float(value)


def feed_settings(table: Table, override: dict[str, Any] | None) -> FeedSettings:
    """Defaults from the registry, then the config's per-feed overrides."""
    o = dict(override or {})
    where = f"feeds.{table.name}"
    _check_keys(o, _FEED_KEYS, where)
    period = _number(o.get("nominal_period_s", table.nominal_period_s), where + ".nominal_period_s",
                     0.0, strict=True)
    jitter = _number(o.get("jitter_s", DEFAULT_JITTER_FRACTION * period), where + ".jitter_s", 0.0)
    if jitter >= MAX_JITTER_FRACTION * period:
        raise ScenarioError(f"{where}.jitter_s: {jitter:g} s must be below half the period "
                            f"({MAX_JITTER_FRACTION * period:g} s) so rows keep their order")
    gap = _number(o.get("gap_fraction", DEFAULT_GAP_FRACTION), where + ".gap_fraction", 0.0)
    if gap >= 0.5:
        raise ScenarioError(f"{where}.gap_fraction: must be below 0.5, got {gap:g}")
    res = o.get("timestamp_resolution_s", table.timestamp_resolution_s)
    if res is not None:
        res = _number(res, where + ".timestamp_resolution_s", 0.0, strict=True)
    return FeedSettings(period, jitter, gap, res)


def load_layout_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_inner_scenario(cfg: dict[str, Any], reg: Registry) -> tuple[dict, dict, list, dict]:
    """Translate a layout config into a plain simulator scenario.

    Returns (scenario dict, feed settings by table, feed_outage anomalies, channel models).
    """
    _check_keys(cfg, _TOP_KEYS, "layout scenario")
    for key in ("seed", "start_time", "duration_seconds"):
        if key not in cfg:
            raise ScenarioError(f"layout scenario: missing required key '{key}'")
    if cfg.get("synthetic", True) is not True:
        raise ScenarioError("layout scenario: 'synthetic' must be true (simulated data with real names)")

    included = reg.included_tables()
    keys = {c.key for t in included for c in t.channels}
    feeds_cfg = cfg.get("feeds") or {}
    _check_keys(feeds_cfg, {t.name for t in included}, "feeds")
    settings = {t.name: feed_settings(t, feeds_cfg.get(t.name)) for t in included}

    default = dict(cfg.get("default_channel") or {})
    _check_keys(default, _CHANNEL_KEYS, "default_channel")
    overrides = cfg.get("channels") or {}
    _check_keys(overrides, keys, "channels")

    couplings = cfg.get("couplings") or []
    coupled: dict[str, dict[str, Any]] = {}
    for i, c in enumerate(couplings):
        _check_keys(c, _COUPLING_KEYS, f"couplings[{i}]")
        if c.get("channel") not in keys or c.get("source") not in keys:
            raise ScenarioError(f"couplings[{i}]: channel and source must be included channel keys")
        if c["channel"] in coupled:
            raise ScenarioError(f"couplings[{i}]: {c['channel']} already has a coupling")
        coupled[c["channel"]] = {"channel": c["source"], **({"gain": c["gain"]} if "gain" in c else {})}

    channels, models = [], {}
    for t in included:
        rate = 1.0 / settings[t.name].period_s
        for c in t.channels:
            model = {"baseline": DEFAULT_BASELINE, "sigma": DEFAULT_SIGMA, **copy.deepcopy(default),
                     **copy.deepcopy(overrides.get(c.key) or {})}
            _check_keys(model, _CHANNEL_KEYS, f"channels.{c.key}")
            nf = model.get("null_fraction", 0.0)
            _number(nf, f"channels.{c.key}.null_fraction", 0.0)
            if nf >= 1.0:
                raise ScenarioError(f"channels.{c.key}.null_fraction must be below 1, got {nf}")
            entry = {"name": c.key, **{k: v for k, v in model.items() if k != "null_fraction"},
                     "sample_rate_hz": rate}
            if c.key in coupled:
                entry["couples_to"] = coupled[c.key]
            channels.append(entry)
            models[c.key] = model

    inner_anoms, outages = [], []
    for i, a in enumerate(cfg.get("anomalies") or []):
        if not isinstance(a, dict):
            raise ScenarioError(f"anomalies[{i}]: expected a mapping")
        if a.get("type") == "feed_outage":
            _check_keys(a, _OUTAGE_KEYS, f"anomalies[{i}] (feed_outage)")
            if a.get("feed") not in settings:
                raise ScenarioError(f"anomalies[{i}]: feed must be one of {sorted(settings)}")
            start = _number(a.get("start"), f"anomalies[{i}].start", 0.0)
            dur = _number(a.get("duration"), f"anomalies[{i}].duration", 0.0, strict=True)
            if start + dur > cfg["duration_seconds"] + 1e-9:
                raise ScenarioError(f"anomalies[{i}]: feed_outage ends after the run")
            outages.append({"feed": a["feed"], "start": start, "duration": dur, "index": i})
        else:
            if "feed" in a:
                raise ScenarioError(f"anomalies[{i}]: 'feed' is only used by feed_outage")
            inner_anoms.append(copy.deepcopy(a))

    start = cfg["start_time"]
    start_dt = start if isinstance(start, datetime) else datetime.fromisoformat(str(start))
    if start_dt.tzinfo is not None:
        start_dt = start_dt.astimezone(UTC).replace(tzinfo=None)
    scenario = {
        "seed": cfg["seed"],
        "start_time": start_dt.isoformat(),
        "duration_seconds": cfg["duration_seconds"],
        "channels": channels,
        "anomalies": inner_anoms,
    }
    return scenario, settings, outages, models


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _feed_rng(seed: int, feed_index: int) -> np.random.Generator:
    """A seed stream per feed that never collides with the per-channel noise streams."""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1_000_000 + feed_index,)))


def _floor(t_us: np.ndarray, resolution_s: float | None) -> np.ndarray:
    if resolution_s is None:
        return t_us
    step = int(round(resolution_s * 1e6))
    return (t_us // step) * step


def _f32(values: np.ndarray) -> np.ndarray:
    """Round to single precision (ptdb columns are float), kept as float64."""
    return values.astype(np.float32).astype(np.float64)


def _map_time(nominal_rel: np.ndarray, realized_abs: np.ndarray, t_rel: int, start_us: int) -> int:
    """Realized time of the first surviving row at or after nominal offset t_rel."""
    i = int(np.searchsorted(nominal_rel, t_rel, side="left"))
    if i < len(nominal_rel):
        return int(realized_abs[i])
    return start_us + t_rel


def generate_layout(config: dict[str, Any] | str | Path, registry: Registry | None = None) -> LayoutRun:
    """Generate a wide-feed run from a layout config (dict or YAML path)."""
    cfg = load_layout_config(config) if isinstance(config, (str, Path)) else copy.deepcopy(config)
    reg = registry or load_registry(cfg.get("layout") or DEFAULT_LAYOUT)
    scenario, settings, outages, models = build_inner_scenario(cfg, reg)
    result = generate(parse_scenario(scenario))
    start_us = datetime_to_us(result.config.start_time)
    end_us = start_us + int(round(result.config.duration_seconds * 1e6))

    feeds: dict[str, Feed] = {}
    nominal_by_feed: dict[str, np.ndarray] = {}
    realized_by_feed: dict[str, np.ndarray] = {}
    for fi, table in enumerate(reg.included_tables()):
        st = settings[table.name]
        rng = _feed_rng(cfg["seed"], fi)
        first_key = table.channels[0].key
        nominal = result.channels[first_key].t_us              # relative to start, shared by the feed
        n = len(nominal)
        jitter = np.round(rng.uniform(0.0, st.jitter_s, n) * 1e6).astype(np.int64)
        keep = rng.random(n) >= st.gap_fraction
        for o in outages:
            if o["feed"] == table.name:
                lo = int(round(o["start"] * 1e6))
                hi = int(round((o["start"] + o["duration"]) * 1e6))
                keep &= ~((nominal >= lo) & (nominal < hi))
        realized = _floor(start_us + nominal + jitter, st.resolution_s)
        nominal_by_feed[table.name] = nominal[keep]
        realized_by_feed[table.name] = realized[keep]
        values = {c.db_column: _f32(result.channels[c.key].values[keep]) for c in table.channels}
        # Healthy NULLs (real feeds have many partial rows): each channel's own seed stream.
        for ci, c in enumerate(table.channels):
            nf = float(models[c.key].get("null_fraction", 0.0))
            if nf > 0:
                crng = np.random.default_rng(np.random.SeedSequence(cfg["seed"],
                                                                    spawn_key=(4_000_000 + 100 * fi + ci,)))
                mask = crng.random(len(values[c.db_column])) < nf
                # A planted spike is one sample: never hide it behind a healthy NULL.
                for lb in result.labels:
                    if lb["channel"] == c.key and lb["type"] in ("spike", "spike_burst"):
                        lo = datetime_to_us(datetime.fromisoformat(lb["start"])) - start_us
                        hi = datetime_to_us(datetime.fromisoformat(lb["end"])) - start_us
                        mask &= ~((nominal[keep] >= lo) & (nominal[keep] < hi))
                values[c.db_column][mask] = np.nan
        feeds[table.name] = Feed(
            table=table.name, time_column=table.time_column,
            columns=[c.db_column for c in table.channels],
            ids=np.arange(1, int(keep.sum()) + 1, dtype=np.int64),
            t_us=realized[keep], values=values, settings=st,
        )

    labels = []
    for lb in result.labels:
        table = lb["channel"].split(".", 1)[0]
        nom, real = nominal_by_feed[table], realized_by_feed[table]
        new = dict(lb)
        for key in ("start", "end", "first_detectable"):
            if lb[key] is not None:
                rel = datetime_to_us(datetime.fromisoformat(lb[key])) - start_us
                new[key] = us_to_iso(_map_time(nom, real, rel, start_us))
        labels.append(new)
    for o in outages:
        labels.append(_outage_label(o, feeds[o["feed"]], start_us))

    labels.sort(key=lambda lb: (lb["start"], lb["channel"]))
    for k, lb in enumerate(labels, start=1):
        lb["id"] = f"ev_{k:04d}"

    meta = {
        "synthetic": True,
        "seed": cfg["seed"],
        "layout_version": reg.version,
        "database_shape": reg.database,
        "start": us_to_iso(start_us),
        "end": us_to_iso(end_us),
        "feeds": {name: {"rows": f.n_rows, "nominal_period_s": f.settings.period_s,
                         "jitter_s": f.settings.jitter_s, "gap_fraction": f.settings.gap_fraction,
                         "timestamp_resolution_s": f.settings.resolution_s}
                  for name, f in feeds.items()},
        "n_labels": len(labels),
    }
    return LayoutRun(cfg, feeds, labels, models, start_us, end_us, meta)


def _outage_label(o: dict[str, Any], feed: Feed, start_us: int) -> dict[str, Any]:
    """Label for a feed outage. Detectable once the gap since the last row exceeds 3 periods."""
    lo = start_us + int(round(o["start"] * 1e6))
    hi = start_us + int(round((o["start"] + o["duration"]) * 1e6))
    period_us = int(round(feed.settings.period_s * 1e6))
    before = feed.t_us[feed.t_us < lo]
    after = feed.t_us[feed.t_us >= hi]
    last = int(before[-1]) if len(before) else lo
    detect = last + int(OUTAGE_DETECT_PERIODS * period_us)
    next_row = int(after[0]) if len(after) else None
    if next_row is not None and detect >= next_row:
        detect = None
    return {
        "id": "",
        "channel": feed.table,
        "type": "feed_outage",
        "start": us_to_iso(lo),
        "end": us_to_iso(hi),
        "first_detectable": us_to_iso(detect),
        "magnitude": None,
        "magnitude_sigma": None,
        "params": {"start_offset_seconds": o["start"], "duration": o["duration"], "feed": feed.table},
        "severity": "high",
    }


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
def merge_feed_events(feeds: dict[str, Feed]) -> Iterator[tuple[datetime, dict[str, float]]]:
    """One event per feed row, merged by time (ties: table name, then id).

    A NULL value is left out of the event. Keys are ``<table>.<db_column>``.
    """
    order = []
    for name in sorted(feeds):
        f = feeds[name]
        order.append(np.rec.fromarrays([f.t_us, np.full(f.n_rows, len(order)), np.arange(f.n_rows)],
                                       names="t,feed,row"))
    if not order:
        return
    allrows = np.concatenate(order)
    allrows.sort(order=["t", "feed", "row"])
    names = sorted(feeds)
    for t, fi, row in allrows:
        f = feeds[names[fi]]
        event = {}
        for col in f.columns:
            v = f.values[col][row]
            if v == v:
                event[f"{f.table}.{col}"] = float(v)
        yield us_to_datetime(int(t)), event


def replay_layout(source: LayoutRun | dict[str, Any] | str | Path) -> Iterator[tuple[datetime, dict[str, float]]]:
    """Replay a wide-feed run as ``(timestamp, {channel_key: value})`` events.

    ``source`` is a LayoutRun, a layout config dict, a layout config YAML, or a
    run folder written by ``write_layout_run`` (labels.json is never opened).
    Timestamps are aware UTC datetimes.
    """
    if isinstance(source, LayoutRun):
        feeds = source.feeds
    elif isinstance(source, dict):
        feeds = generate_layout(source).feeds
    else:
        path = Path(source)
        if path.suffix in (".yaml", ".yml"):
            feeds = generate_layout(path).feeds
        else:
            from .layout_io import read_feeds  # local import to avoid a cycle
            feeds = read_feeds(path)
    yield from merge_feed_events(feeds)
