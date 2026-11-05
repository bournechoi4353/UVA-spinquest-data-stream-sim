"""Randomized stress test of the detectors on realistic SIMULATED ptdb data.

The end-to-end test plants the same seven large anomalies every time, so "7 of 7" says little about
sensitivity. Here every seed gets its own random plan:

* 8 anomalies per run (every CLEAN_EVERY-th seed is a clean run, for the false alarm rate), with
  types drawn uniformly from ten kinds: spike, spike_burst, step_shift, linear_drift,
  exponential_drift, noise_growth, transient_decay, dropout, stuck_sensor and feed_outage;
* random channels on all three feeds, random start times, random signs;
* sizes log-uniform from 2 to 10 sigma (noise growth 1.5x to 5x), random durations, including
  short dropouts and outages below the detectors' limits;
* anomalies on different channels may overlap in time; anomalies on one channel, or on a channel
  during an outage of its feed, do not, so every label stays observable.

Each run is calibrated on its own separate healthy run, both written to a ptdb-shaped SQLite file
with 1% duplicate and 1% late rows and read back through the live reader, exactly like real data.

``by_design`` marks labels the detectors are not meant to catch, reported separately:
* spikes and spike bursts below the channel's spike threshold, in the detector's own sigma;
* dropouts shorter than the channel's missing limit, outages shorter than the feed's outage limit;
* stuck sensors on channels that hold values when healthy (stuck detection is off there);
* changes smaller than the 32-bit float resolution of the stored value (ptdb stores float), which
  no detector can see.
Drifts, steps, transients and noise growth that are small compared with the channel's healthy
spread stay in scope and count as misses. ``eff_sigma`` gives every sized anomaly in the detector's
own sigma (the planted size is in units of the channel's short-term simulated noise, and slow
wander can make the healthy spread much larger), and the report has a table by that size too.
"""

from __future__ import annotations

import math
import random

import numpy as np
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .engine import Engine
from .ingest import PtdbReader

UTC = timezone.utc
TYPES = ["spike", "spike_burst", "step_shift", "linear_drift", "exponential_drift", "noise_growth",
         "transient_decay", "dropout", "stuck_sensor", "feed_outage"]
SIZED = {"spike", "spike_burst", "step_shift", "linear_drift", "exponential_drift", "transient_decay"}
CLEAN_EVERY = 5
PER_RUN = 8


def random_plan(reg, hours: float, rng: random.Random, n: int = PER_RUN) -> list[dict[str, Any]]:
    """n random anomalies over ``hours``, no two on one channel at once, none hidden by an outage."""
    D = hours * 3600.0
    channels = [c.key for c in reg.included_channels()]
    feed_of = {c.key: t.name for t in reg.included_tables() for c in t.channels}
    taken: list[tuple[str, float, float]] = []      # (channel or feed, start, end)

    def free(subject: str, a: float, b: float) -> bool:
        for s, x, y in taken:
            same = s == subject or feed_of.get(subject) == s or feed_of.get(s) == subject
            if same and a < y + 300 and x < b + 300:
                return False
        return True

    plan: list[dict[str, Any]] = []
    tries = 0
    while len(plan) < n and tries < 500:
        tries += 1
        kind = rng.choice(TYPES)
        mag = math.exp(rng.uniform(math.log(2.0), math.log(10.0))) * rng.choice([-1, 1])
        start = rng.uniform(0.05, 0.85) * D
        a: dict[str, Any] = {"type": kind, "start": round(start, 1)}
        if kind == "spike":
            span = 10.0
            a["magnitude"] = round(mag, 2)
        elif kind == "spike_burst":
            a.update(magnitude=round(mag, 2), count=rng.randint(2, 5), spacing=round(rng.uniform(6, 60), 1))
            span = a["count"] * a["spacing"]
        elif kind == "step_shift":
            span = rng.uniform(600, 5400)
            a.update(magnitude=round(mag, 2), duration=round(span, 1))
        elif kind in ("linear_drift", "exponential_drift"):
            span = rng.uniform(1800, 7200)
            a.update(magnitude=round(mag, 2), duration=round(span, 1))
        elif kind == "noise_growth":
            span = rng.uniform(1800, 7200)
            a.update(factor=round(rng.uniform(1.5, 5.0), 2), duration=round(span, 1))
        elif kind == "transient_decay":
            tau = rng.uniform(60, 900)
            span = 5 * tau
            a.update(magnitude=round(mag, 2), time_constant=round(tau, 1))
        elif kind == "dropout":
            span = math.exp(rng.uniform(math.log(20), math.log(1800)))
            a["duration"] = round(span, 1)
        elif kind == "stuck_sensor":
            span = rng.uniform(120, 3600)
            a["duration"] = round(span, 1)
        else:   # feed_outage
            span = math.exp(rng.uniform(math.log(60), math.log(3600)))
            a["duration"] = round(span, 1)
        if start + span > 0.95 * D:
            continue
        if kind == "feed_outage":
            subject = rng.choice(sorted({feed_of[c] for c in channels}))
            a["feed"] = subject
        else:
            subject = rng.choice(channels)
            a["channel"] = subject
        if not free(subject, start, start + span):
            continue
        taken.append((subject, start, start + span))
        plan.append(a)
    return sorted(plan, key=lambda x: x["start"])


def _events(cfg, reg, path: Path):
    from sim.db_writer import write_fixture
    from sim.layout_scenario import generate_layout
    from sim.profiler.adapters import SQLiteAdapter
    run = generate_layout(cfg, reg)
    write_fixture(run, path, dup_fraction=0.01, late_fraction=0.01)
    a = SQLiteAdapter(path)
    try:
        events = list(PtdbReader(a, reg).read_all())
    finally:
        a.close()
    return run, events


def run_seed(seed: int, hours: float = 12.0, calibrate_hours: float = 24.0,
             profile: str | None = "out/profile/ptdb_profile.json",
             layout: str = "configs/ptdb_layout.yaml") -> dict[str, Any]:
    """One stress run. Returns the labels with detection results and the false alarms."""
    from sim.realistic import realistic_config
    from sim.registry import load_registry
    from .scoring import COMPATIBLE, GRACE_S

    reg = load_registry(layout)
    rng = random.Random(seed * 7_777 + 13)
    clean = seed % CLEAN_EVERY == 0
    cfg = realistic_config(reg, Path(profile) if profile else None, hours, seed)
    healthy_cfg = {**cfg, "seed": seed + 7919, "duration_seconds": calibrate_hours * 3600}
    test_cfg = {**cfg, "anomalies": [] if clean else random_plan(reg, hours, rng)}
    with tempfile.TemporaryDirectory() as tmp:
        _, healthy = _events(healthy_cfg, reg, Path(tmp) / "healthy.sqlite")
        run, events = _events(test_cfg, reg, Path(tmp) / "test.sqlite")
    feeds = {t.name: [c.key for c in t.channels] for t in reg.included_tables()}
    eng = Engine.from_healthy(healthy, feeds=feeds)
    for t, ev in events:
        eng.step(t, ev)
    end = events[-1][0] if events else None

    def t(iso):
        if iso is None:
            return None
        x = datetime.fromisoformat(iso)
        return x if x.tzinfo else x.replace(tzinfo=UTC)

    labels = []
    for lb in run.labels:
        p = lb.get("params", {})
        kind, subj = lb["type"], lb["channel"]
        dur = p.get("duration")
        model = run.channel_models.get(subj, {})
        mon = eng.monitors.get(subj)
        mag = p.get("magnitude") if kind in SIZED else None
        eff = None
        if mag is not None and mon is not None and model.get("sigma"):
            eff = abs(mag) * float(model["sigma"]) / mon.cal.sigma
        by_design = None
        if model.get("sigma") is not None:
            change = float(model["sigma"]) * max(abs(mag or 0.0), float(p.get("factor") or 0.0), 1.0)
            if change < 2 * float(np.spacing(np.float32(abs(float(model.get("baseline", 0.0)))))):
                by_design = "smaller than the 32-bit float resolution of the stored value"
        if by_design:
            pass
        elif (kind in ("spike", "spike_burst") and mon is not None and eff is not None
              and eff < (mon.cal.spike_z or eng.s.spike_z)):
            by_design = f"below the channel's {mon.cal.spike_z or eng.s.spike_z:.3g} sigma spike threshold"
        elif kind == "stuck_sensor" and mon is not None and mon.cal.stuck_normal:
            by_design = "stuck detection is off: the channel holds values when healthy"
        elif kind == "dropout" and subj in eng.monitors:
            lim = eng.monitors[subj].cal.missing_after_s or 0.0
            if dur is not None and dur < lim:
                by_design = f"shorter than the {lim:.0f} s missing limit"
        elif kind == "feed_outage" and subj in eng.feed_watch:
            lim = eng.feed_watch[subj].limit()
            if dur is not None and dur < lim:
                by_design = f"shorter than the {lim:.0f} s outage limit"
        labels.append({"seed": seed, "id": lb["id"], "type": kind, "channel": subj,
                       "magnitude_sigma": mag, "eff_sigma": None if eff is None else round(eff, 3),
                       "factor": p.get("factor"), "duration_s": dur, "start": lb["start"], "end": lb.get("end"),
                       "first_detectable": lb.get("first_detectable"), "by_design": by_design,
                       "detected": False, "compatible": False, "delay_s": None})
    by_id = {lb["id"]: lb for lb in labels}
    false = []
    for c in eng.raised:
        at = c.detected_at
        hits = [lb for lb in labels if lb["channel"] == c.subject and t(lb["start"]) <= at
                and (at - (t(lb["end"]) or end or at)).total_seconds() <= GRACE_S]
        good = [lb for lb in hits if lb["type"] in COMPATIBLE.get(c.kind, set())]
        pick = (good or hits)
        if not pick:
            false.append({"seed": seed, "subject": c.subject, "kind": c.kind, "at": at.isoformat()})
            continue
        lb = by_id[pick[0]["id"]]
        if not lb["detected"] or (good and not lb["compatible"]):
            lb["detected"], lb["compatible"] = True, bool(good)
            fd = t(lb["first_detectable"])
            lb["delay_s"] = None if fd is None else round((at - fd).total_seconds(), 1)
    test_hours = (events[-1][0] - events[0][0]).total_seconds() / 3600 if events else 0.0
    return {"seed": seed, "clean": clean, "hours": test_hours, "labels": labels, "false_alarms": false}


# ---- statistics ---------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def poisson_rate_ci(k: int, exposure: float) -> tuple[float, float]:
    """Approximate 95% interval for a Poisson rate k/exposure (exact upper bound 3/exposure at k=0)."""
    if exposure <= 0:
        return (float("nan"), float("nan"))
    if k == 0:
        return (0.0, 3.0 / exposure)
    lo = k * (1 - 1 / (9 * k) - 1.96 / (3 * math.sqrt(k))) ** 3
    k1 = k + 1
    hi = k1 * (1 - 1 / (9 * k1) + 1.96 / (3 * math.sqrt(k1))) ** 3
    return (lo / exposure, hi / exposure)


def size_bin(lb: dict[str, Any]) -> str:
    if lb["magnitude_sigma"] is not None:
        m = abs(lb["magnitude_sigma"])
        return "2-3 sigma" if m < 3 else "3-5 sigma" if m < 5 else "5-7 sigma" if m < 7 else "7-10 sigma"
    if lb["factor"] is not None:
        f = lb["factor"]
        return "1.5-2x" if f < 2 else "2-3x" if f < 3 else "3-5x"
    d = lb["duration_s"] or 0
    return "<1 min" if d < 60 else "1-5 min" if d < 300 else "5-30 min" if d < 1800 else ">30 min"


EFF_ORDER = ["<1", "1-2", "2-3", "3-5", "5-10", ">10"]


def eff_bin(lb: dict[str, Any]) -> str | None:
    e = lb.get("eff_sigma")
    if e is None:
        return None
    return ("<1" if e < 1 else "1-2" if e < 2 else "2-3" if e < 3 else "3-5" if e < 5 else "5-10" if e < 10
            else ">10") + " detector sigma"


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [lb for r in results for lb in r["labels"]]
    fa = [a for r in results for a in r["false_alarms"]]
    hours = sum(r["hours"] for r in results)
    clean = [r for r in results if r["clean"]]
    clean_h = sum(r["hours"] for r in clean)
    clean_fa = sum(len(r["false_alarms"]) for r in clean)

    def rate(rows):
        k = sum(1 for x in rows if x["detected"])
        lo, hi = wilson(k, len(rows))
        delays = sorted(x["delay_s"] for x in rows if x["detected"] and x["delay_s"] is not None)
        med = delays[len(delays) // 2] if delays else None
        return {"n": len(rows), "detected": k, "rate": k / len(rows) if rows else float("nan"),
                "ci": (lo, hi), "compatible": sum(1 for x in rows if x["compatible"]), "median_delay_s": med}

    in_scope = [lb for lb in labels if not lb["by_design"]]
    out = {"runs": len(results), "clean_runs": len(clean), "test_hours": hours, "labels": len(labels),
           "in_scope": rate(in_scope), "by_design": rate([lb for lb in labels if lb["by_design"]]),
           "per_type": {}, "per_type_size": {}, "per_type_eff": {}, "by_design_reasons": {},
           "false_alarms": {"total": len(fa), "per_hour": len(fa) / hours if hours else float("nan"),
                            "ci": poisson_rate_ci(len(fa), hours)},
           "clean_false_alarms": {"total": clean_fa, "hours": clean_h,
                                  "per_hour": clean_fa / clean_h if clean_h else float("nan"),
                                  "ci": poisson_rate_ci(clean_fa, clean_h)},
           "false_alarm_kinds": {}}
    for ty in TYPES:
        rows = [lb for lb in in_scope if lb["type"] == ty]
        if rows:
            out["per_type"][ty] = rate(rows)
            bins: dict[str, list] = {}
            for lb in rows:
                bins.setdefault(size_bin(lb), []).append(lb)
            out["per_type_size"][ty] = {b: rate(v) for b, v in sorted(bins.items())}
            eb: dict[str, list] = {}
            for lb in rows:
                if eff_bin(lb):
                    eb.setdefault(eff_bin(lb), []).append(lb)
            if eb:
                out["per_type_eff"][ty] = {b: rate(eb[b]) for b in
                                           sorted(eb, key=lambda x: EFF_ORDER.index(x.split()[0]))}
    for lb in labels:
        if lb["by_design"]:
            key = lb["by_design"].split(":")[0]
            if "spike threshold" in key:
                key = "spike or burst below the spike threshold"
            elif "missing limit" in key:
                key = "dropout shorter than the missing limit"
            elif "outage limit" in key:
                key = "outage shorter than the outage limit"
            out["by_design_reasons"][key] = out["by_design_reasons"].get(key, 0) + 1
    for a in fa:
        key = f"{a['kind']}"
        out["false_alarm_kinds"][key] = out["false_alarm_kinds"].get(key, 0) + 1
    return out
