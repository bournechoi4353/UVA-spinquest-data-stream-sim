"""Batch generation of many randomized scenarios (training data).

Design
------
* One master seed drives everything. `SeedSequence(master)` is split into two
  independent streams: one for the batch plan (categories and splits) and one
  that is spawned into a child per scenario. Each scenario child is split again
  into a config stream (what the scenario looks like) and a seed stream (the
  scenario's own `seed` field, which drives the simulator's noise).
* Scenario categories are assigned with EXACT counts, not coin flips:
    clean   no anomalies                      (--clean-frac of all runs)
    multi   several overlapping anomalies     (--multi-frac of all runs)
    hard    a hard-case recipe, see below     (--hard-frac of the NON-CLEAN runs)
    single  one injected event                (everything else)
* Primary anomaly types are handed out round-robin inside each category, so
  every type shows up about equally often before the split.
* Train/val/test are split at the scenario level: each scenario lands in
  exactly one split, and every scenario has its own distinct seed (asserted).

Hard cases
----------
    slow_drift         a 3 to 5 sigma drift spread over 40 to 70 percent of the run
    noise_like_spikes  2 or 3 spikes of only 3.5 to 5 sigma
    spike_then_step    a spike, then a step shift a few seconds later
    stuck              a stuck sensor

A decoupling event is always accompanied by a "driver" anomaly on the source
channel (a held drift or step). Without it the decoupling would be invisible,
because the coupled contribution is small while the source sits at baseline.
The driver counts toward `n_anomalies` but not toward the single/multi
distinction: a decoupling plus its driver is one event.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from .io import write_run
from .scenario import parse_scenario
from .stream import generate

SPLITS = ("train", "val", "test")

DEFAULT_CHANNELS: dict[str, dict[str, float]] = {
    "polarization": {"baseline": 41.0, "sigma": 0.15, "sample_rate_hz": 1.0},
    "beam_current": {"baseline": 5.0, "sigma": 0.05, "sample_rate_hz": 1.0},
    "rf_power": {"baseline": 100.0, "sigma": 0.4, "sample_rate_hz": 1.0},
    "target_temp": {"baseline": 1.0, "sigma": 0.01, "sample_rate_hz": 1.0},
    "nmr_signal": {"baseline": 2.5, "sigma": 0.03, "sample_rate_hz": 0.2},
}
SOURCE_CHANNEL, COUPLED_CHANNEL = "polarization", "nmr_signal"

ALL_TYPES = [
    "spike", "spike_burst", "step_shift", "linear_drift", "exponential_drift",
    "noise_growth", "transient_decay", "dropout", "stuck_sensor", "decoupling",
]
# Types that last long enough for other anomalies to overlap with them.
WINDOWED_TYPES = [
    "step_shift", "linear_drift", "exponential_drift", "noise_growth",
    "stuck_sensor", "decoupling", "transient_decay", "dropout",
]
EXTRA_TYPES = [t for t in ALL_TYPES if t != "decoupling"]
HARD_KINDS = ["slow_drift", "noise_like_spikes", "spike_then_step", "stuck"]


# ---------------------------------------------------------------------------
# Plan: which scenario is which category, which split
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PlanEntry:
    index: int
    category: str  # clean | single | multi | hard
    primary: str | None  # anomaly type (single, multi) or hard kind (hard)


def make_plan(n: int, rng: np.random.Generator, clean_frac: float, hard_frac: float,
              multi_frac: float) -> list[PlanEntry]:
    """Assign a category (with exact counts) and a primary anomaly to every scenario."""
    n_clean = round(clean_frac * n)
    n_non_clean = n - n_clean
    n_hard = round(hard_frac * n_non_clean)
    n_multi = min(round(multi_frac * n), n_non_clean - n_hard)
    n_single = n_non_clean - n_hard - n_multi
    cats = ["clean"] * n_clean + ["hard"] * n_hard + ["multi"] * n_multi + ["single"] * n_single
    cats = [cats[j] for j in rng.permutation(n)]

    counters = {"single": 0, "multi": 0, "hard": 0}
    plan = []
    for i, cat in enumerate(cats):
        primary = None
        if cat != "clean":
            j = counters[cat]
            counters[cat] += 1
            pool = {"single": ALL_TYPES, "multi": WINDOWED_TYPES, "hard": HARD_KINDS}[cat]
            primary = pool[j % len(pool)]
        plan.append(PlanEntry(i, cat, primary))
    return plan


def assign_splits(n: int, rng: np.random.Generator, fractions: tuple[float, float, float]) -> list[str]:
    """Randomly assign each scenario index to a split (exact counts, at least 1 each when n >= 3)."""
    total = sum(fractions)
    n_val = round(n * fractions[1] / total)
    n_test = round(n * fractions[2] / total)
    if n >= 3:
        n_val, n_test = max(n_val, 1), max(n_test, 1)
    n_train = n - n_val - n_test
    labels = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    return [labels[j] for j in rng.permutation(n)]


# ---------------------------------------------------------------------------
# Small random helpers (always return plain Python values, safe for YAML/JSON)
# ---------------------------------------------------------------------------
def _uniform(rng, a, b) -> float:
    return float(rng.uniform(a, b))


def _signed(rng, a, b) -> float:
    sign = 1.0 if rng.random() < 0.5 else -1.0
    return round(sign * _uniform(rng, a, b), 3)


def _flag(rng, p=0.5) -> bool:
    return bool(rng.random() < p)


def _pick(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def _grid(x: float, period: float) -> float:
    """Round DOWN to a multiple of the sample period (so windows always contain a sample)."""
    return round(math.floor(x / period + 1e-9) * period, 6)


def _grid_up(x: float, period: float) -> float:
    """Round UP to a multiple of the sample period."""
    return round(math.ceil(x / period - 1e-9) * period, 6)


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------
def _make_channels(rng, required: set[str]) -> list[dict[str, Any]]:
    names = list(DEFAULT_CHANNELS)
    k = int(rng.integers(2, len(names) + 1))
    chosen = {names[j] for j in rng.permutation(len(names))[:k]} | required
    if COUPLED_CHANNEL in chosen:
        chosen.add(SOURCE_CHANNEL)

    channels: dict[str, dict[str, Any]] = {}
    for name in names:
        if name not in chosen:
            continue
        base = DEFAULT_CHANNELS[name]
        sigma = base["sigma"] * math.exp(_uniform(rng, math.log(0.5), math.log(2.0)))
        ch: dict[str, Any] = {
            "name": name,
            "baseline": round(base["baseline"] * _uniform(rng, 0.8, 1.2), 6),
            "sigma": round(sigma, 6),
            "sample_rate_hz": base["sample_rate_hz"],
        }
        if _flag(rng, 0.4):
            ch["ar1_phi"] = round(_uniform(rng, 0.2, 0.8), 3)
        if _flag(rng, 0.5):
            ch["slow_wander"] = {
                "amplitude": round(sigma * _uniform(rng, 0.1, 0.5), 6),
                "period_seconds": round(_uniform(rng, 1800, 10800)),
            }
        channels[name] = ch

    if COUPLED_CHANNEL in channels:
        src, dst = channels[SOURCE_CHANNEL], channels[COUPLED_CHANNEL]
        # Keep gain * sigma_source safely below the coupled channel's total sigma.
        gain = min(_uniform(rng, 0.08, 0.2), 0.8 * dst["sigma"] / src["sigma"])
        dst["couples_to"] = {"channel": SOURCE_CHANNEL, "gain": round(gain, 4)}
    return list(channels.values())


# ---------------------------------------------------------------------------
# Anomaly samplers
# ---------------------------------------------------------------------------
def _sample_anomaly(rng, atype: str, ch: dict[str, Any], D: float, lo: float,
                    start_range: tuple[float, float] | None = None, anchor: bool = False) -> dict[str, Any]:
    """Draw one anomaly of `atype` on channel `ch` with random parameters.

    All start times and durations are multiples of the channel's sample period,
    so every window contains at least one sample. If `start_range` is given the
    start is drawn from it (used to make anomalies overlap); otherwise from
    anywhere between `lo` and the latest time at which the anomaly still fits.
    `anchor=True` marks the anomaly other anomalies will overlap with, so a dropout
    is made at least 30 s long.
    """
    P = 1.0 / ch["sample_rate_hz"]

    def place(dur: float) -> tuple[float, float]:
        if start_range is None:
            dur = min(dur, 0.9 * (D - lo))
            s = _uniform(rng, lo, D - dur)
        else:
            s = _uniform(rng, *start_range)
        # Round into the allowed range: up when overlapping a given window, down otherwise.
        s = _grid(s, P) if start_range is None else _grid_up(s, P)
        dur = max(P, _grid(min(dur, D - s), P))
        return min(s, _grid(D - dur, P)), dur

    a: dict[str, Any] = {"channel": ch["name"], "type": atype}
    if atype == "spike":
        a["start"], _ = place(P)
        a["magnitude"] = _signed(rng, 6, 15)
    elif atype == "spike_burst":
        count = int(rng.integers(3, 9))
        spacing = P * int(rng.integers(1, int(math.ceil(20 / P)) + 1))
        a["start"], _ = place(count * spacing + P)
        a.update(magnitude=round(_uniform(rng, 5, 10), 3), count=count, spacing=round(spacing, 6),
                 alternate_sign=_flag(rng))
    elif atype == "step_shift":
        if _flag(rng, 0.4):
            a["start"], a["duration"] = place(_uniform(rng, 300, 5400))
        else:
            a["start"], _ = place(0.15 * D)
        a["magnitude"] = _signed(rng, 3, 10)
    elif atype in ("linear_drift", "exponential_drift"):
        a["start"], a["duration"] = place(_uniform(rng, 1200, 10800))
        a["magnitude"] = _signed(rng, 5, 25)
        a["hold_after"] = _flag(rng)
        if atype == "exponential_drift":
            a["time_constant"] = round(a["duration"] / _uniform(rng, 2, 5), 3)
    elif atype == "noise_growth":
        a["start"], a["duration"] = place(_uniform(rng, 1200, 7200))
        a.update(factor=round(_uniform(rng, 2, 5), 3), hold_after=_flag(rng))
    elif atype == "transient_decay":
        tau = float(int(rng.integers(30, 601)))
        a["start"], _ = place(5 * tau)
        a.update(magnitude=_signed(rng, 6, 15), time_constant=tau)
    elif atype == "dropout":
        if _flag(rng) and not anchor:
            dur = P * int(rng.integers(1, int(math.ceil(4 / P)) + 1))  # short: under 5 s where possible
        else:
            dur = _uniform(rng, 30 if anchor else 5, 600)
        a["start"], a["duration"] = place(dur)
    elif atype == "stuck_sensor":
        a["start"], a["duration"] = place(_uniform(rng, 60, 1800))
    elif atype == "decoupling":
        a["start"], a["duration"] = place(_uniform(rng, 900, 7200))
        a["hold_after"] = _flag(rng)
        if _flag(rng, 0.2):
            a["end_gain_fraction"] = round(_uniform(rng, 0.05, 0.3), 3)
    else:  # pragma: no cover
        raise ValueError(atype)
    return a


def _anomaly_end(a: dict[str, Any], D: float) -> float:
    """When an anomaly's effect is over, for choosing overlapping neighbours."""
    kind = a["type"]
    if kind == "spike":
        return a["start"]
    if kind == "spike_burst":
        return a["start"] + (a["count"] - 1) * a["spacing"]
    if kind == "transient_decay":
        return min(a["start"] + 5 * a["time_constant"], D)  # same rule as the label end
    if a.get("hold_after") or a.get("duration") is None:
        return D
    return a["start"] + a["duration"]


def _decoupling_driver(rng, source: dict[str, Any], dec: dict[str, Any], D: float, lo: float) -> dict[str, Any]:
    """A held drift or step on the source channel, around the time decoupling starts."""
    if _flag(rng):
        dur = _grid(_uniform(rng, 600, 3600), 1.0)
        start = _grid(max(lo, dec["start"] - dur * _uniform(rng, 0.0, 0.8)), 1.0)
        dur = min(dur, _grid(D - start, 1.0))
        return {"channel": source["name"], "type": "linear_drift", "start": start, "duration": dur,
                "magnitude": _signed(rng, 8, 25), "hold_after": True}
    start = _grid(max(lo, dec["start"] - _uniform(rng, 0, 300)), 1.0)
    return {"channel": source["name"], "type": "step_shift", "start": start, "magnitude": _signed(rng, 8, 25)}


def _hard_anomalies(rng, kind: str, channels: list[dict[str, Any]], D: float, lo: float) -> list[dict[str, Any]]:
    ch = _pick(rng, channels)
    P = 1.0 / ch["sample_rate_hz"]
    if kind == "slow_drift":
        dur = _grid(_uniform(rng, 0.4, 0.7) * D, P)
        return [{"channel": ch["name"], "type": "linear_drift", "start": _grid(_uniform(rng, lo, D - dur), P),
                 "duration": dur, "magnitude": _signed(rng, 3, 5), "hold_after": _flag(rng)}]
    if kind == "noise_like_spikes":
        count = int(rng.integers(2, 4))
        slots = sorted(_grid(_uniform(rng, lo, D - 2 * P), P) for _ in range(count))
        slots = [s for k, s in enumerate(slots) if k == 0 or s - slots[k - 1] >= 60]  # keep them apart
        return [{"channel": ch["name"], "type": "spike", "start": s, "magnitude": _signed(rng, 3.5, 5)}
                for s in slots]
    if kind == "spike_then_step":
        gap = P * int(rng.integers(1, int(math.ceil(30 / P)) + 1))
        t0 = _grid(_uniform(rng, lo, D - gap - 2 * P), P)
        return [
            {"channel": ch["name"], "type": "spike", "start": t0, "magnitude": _signed(rng, 6, 10)},
            {"channel": ch["name"], "type": "step_shift", "start": round(t0 + gap, 6), "magnitude": _signed(rng, 3, 6)},
        ]
    if kind == "stuck":
        return [_sample_anomaly(rng, "stuck_sensor", ch, D, lo)]
    raise ValueError(kind)  # pragma: no cover


# ---------------------------------------------------------------------------
# One scenario
# ---------------------------------------------------------------------------
def build_scenario(rng: np.random.Generator, seed: int, entry: PlanEntry,
                   duration_range_hours: tuple[float, float]) -> dict[str, Any]:
    """Build one randomized scenario config (a plain dict ready for parse_scenario)."""
    D = float(max(60, round(_uniform(rng, *duration_range_hours) * 3600 / 60) * 60))
    lo = 0.1 * D  # keep the start of every run healthy (calibration room)
    start_time = datetime(2026, 1, 1) + timedelta(minutes=int(rng.integers(0, 365 * 24 * 60)))

    needs_coupling = entry.primary == "decoupling"
    required = {SOURCE_CHANNEL, COUPLED_CHANNEL} if needs_coupling else set()
    channels = _make_channels(rng, required)
    by_name = {c["name"]: c for c in channels}

    anomalies: list[dict[str, Any]] = []
    if entry.category == "hard":
        anomalies = _hard_anomalies(rng, entry.primary, channels, D, lo)
    elif entry.category in ("single", "multi"):
        if entry.primary == "decoupling":
            primary = _sample_anomaly(rng, "decoupling", by_name[COUPLED_CHANNEL], D, lo)
            anomalies = [primary, _decoupling_driver(rng, by_name[SOURCE_CHANNEL], primary, D, lo)]
        else:
            primary = _sample_anomaly(rng, entry.primary, _pick(rng, channels), D, lo,
                                      anchor=entry.category == "multi")
            anomalies = [primary]
        if entry.category == "multi":
            a_start, a_end = primary["start"], _anomaly_end(primary, D)
            for _ in range(int(rng.integers(1, 3))):
                hi = max(a_start, min(a_end, D - 60))
                anomalies.append(_sample_anomaly(rng, _pick(rng, EXTRA_TYPES), _pick(rng, channels), D, lo,
                                                 start_range=(a_start, hi)))
    anomalies.sort(key=lambda a: a["start"])

    return {
        "seed": seed,
        "start_time": start_time.isoformat(),
        "duration_seconds": D,
        "channels": channels,
        "anomalies": anomalies,
    }


# ---------------------------------------------------------------------------
# The whole batch
# ---------------------------------------------------------------------------
def scenario_seeds(master_seed: int, n: int):
    """Derive the plan stream, split stream, and (config stream, seed) per scenario."""
    top = np.random.SeedSequence(master_seed)
    plan_ss, scenarios_ss = top.spawn(2)
    plan_rng = np.random.default_rng(plan_ss)
    children = scenarios_ss.spawn(n)
    out = []
    for child in children:
        cfg_ss, seed_ss = child.spawn(2)
        seed = int(seed_ss.generate_state(1, dtype=np.uint64)[0] & np.uint64((1 << 63) - 1))
        out.append((np.random.default_rng(cfg_ss), seed))
    return plan_rng, out


def generate_batch(
    n: int,
    master_seed: int,
    out_dir: str | Path,
    clean_frac: float = 0.25,
    hard_frac: float = 0.2,
    multi_frac: float = 0.25,
    split: tuple[float, float, float] = (80, 10, 10),
    duration_range_hours: tuple[float, float] = (1.0, 12.0),
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate `n` scenarios, write them under `out_dir/<split>/run_XXXX/`, and write manifest.json.

    Returns the manifest dict.
    """
    if n < 3:
        raise ValueError("n must be at least 3 (one scenario per split)")
    for name, v in (("clean_frac", clean_frac), ("hard_frac", hard_frac), ("multi_frac", multi_frac)):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1, got {v}")
    if clean_frac + multi_frac > 1.0:
        raise ValueError("clean_frac + multi_frac cannot exceed 1")
    if len(split) != 3 or min(split) < 0 or sum(split) <= 0:
        raise ValueError(f"split must be three non-negative numbers, got {split}")
    lo_h, hi_h = duration_range_hours
    if not 0 < lo_h <= hi_h:
        raise ValueError(f"duration_range_hours must satisfy 0 < min <= max, got {duration_range_hours}")

    out_dir = Path(out_dir)
    _prepare_output(out_dir, overwrite)

    plan_rng, scenario_streams = scenario_seeds(master_seed, n)
    seeds = [seed for _, seed in scenario_streams]
    if len(set(seeds)) != n:
        raise RuntimeError("scenario seeds are not unique; choose a different master seed")
    plan = make_plan(n, plan_rng, clean_frac, hard_frac, multi_frac)
    splits = assign_splits(n, plan_rng, split)

    width = max(4, len(str(n - 1)))
    runs = []
    for entry, split_name, (cfg_rng, seed) in zip(plan, splits, scenario_streams):
        config = build_scenario(cfg_rng, seed, entry, (lo_h, hi_h))
        result = generate(parse_scenario(config))
        run_name = f"run_{entry.index:0{width}d}"
        write_run(result, out_dir / split_name / run_name)
        counts: dict[str, int] = {}
        for label in result.labels:
            counts[label["type"]] = counts.get(label["type"], 0) + 1
        runs.append({
            "run": run_name,
            "path": f"{split_name}/{run_name}",
            "split": split_name,
            "seed": seed,
            "category": entry.category,
            "hard_kind": entry.primary if entry.category == "hard" else None,
            "channels": [c["name"] for c in config["channels"]],
            "duration_seconds": config["duration_seconds"],
            "n_anomalies": len(result.labels),
            "anomaly_types": sorted(counts),
            "anomaly_counts": dict(sorted(counts.items())),
        })

    manifest = {
        "master_seed": master_seed,
        "n": n,
        "params": {
            "clean_frac": clean_frac, "hard_frac": hard_frac, "multi_frac": multi_frac,
            "split": list(split), "duration_range_hours": [lo_h, hi_h],
        },
        "split_counts": {s: sum(1 for r in runs if r["split"] == s) for s in SPLITS},
        "summary": summarize(runs),
        "runs": runs,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _prepare_output(out_dir: Path, overwrite: bool) -> None:
    """Refuse to mix a new batch into an old one. With overwrite, remove only what a batch creates."""
    owned = [out_dir / "manifest.json", *(out_dir / s for s in SPLITS)]
    existing = [p for p in owned if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"{out_dir} already contains batch output ({', '.join(p.name for p in existing)}). "
            f"Use --overwrite to replace it or choose another --out."
        )
    for p in existing:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts per anomaly type per split, per category per split, and types missing from a split."""
    events = {t: {s: 0 for s in SPLITS} for t in ALL_TYPES}
    categories = {c: {s: 0 for s in SPLITS} for c in ("clean", "single", "multi", "hard")}
    for r in runs:
        categories[r["category"]][r["split"]] += 1
        for t, c in r["anomaly_counts"].items():
            events[t][r["split"]] += c
    for row in events.values():
        row["total"] = sum(row[s] for s in SPLITS)
    for row in categories.values():
        row["total"] = sum(row[s] for s in SPLITS)
    missing = {s: [t for t in ALL_TYPES if events[t][s] == 0] for s in SPLITS}
    return {"events_per_type": events, "runs_per_category": categories, "types_missing_from_split": missing}


def format_summary(manifest: dict[str, Any]) -> str:
    """A plain-text table of the summary, for printing."""
    s = manifest["summary"]
    lines = []
    header = f"{'':<20}{'train':>7}{'val':>7}{'test':>7}{'total':>8}"
    lines.append(f"Batch: {manifest['n']} scenarios, master seed {manifest['master_seed']}")
    lines.append("")
    lines.append("Runs per category")
    lines.append(header)
    for cat, row in s["runs_per_category"].items():
        lines.append(f"{cat:<20}{row['train']:>7}{row['val']:>7}{row['test']:>7}{row['total']:>8}")
    lines.append("")
    lines.append("Labeled anomaly events per type (includes decoupling drivers)")
    lines.append(header)
    for t, row in s["events_per_type"].items():
        flag = "  <- missing from a split" if 0 in (row["train"], row["val"], row["test"]) else ""
        lines.append(f"{t:<20}{row['train']:>7}{row['val']:>7}{row['test']:>7}{row['total']:>8}{flag}")
    lines.append("")
    missing = s["types_missing_from_split"]
    if any(missing.values()):
        lines.append("WARNING: some anomaly types do not appear in every split:")
        for split_name, types in missing.items():
            if types:
                lines.append(f"  {split_name}: {', '.join(types)}")
    else:
        lines.append("Every anomaly type appears in train, val, and test.")
    return "\n".join(lines)
