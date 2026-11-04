"""Score the detectors against ground truth labels (Phase 5).

Detection runs through ``replay`` only, so labels are never seen while
detecting. Calibration uses a separate healthy run of the same system: the
run's own config with every anomaly removed and a different seed.

Matching rules:
* A label is detected when any condition on the label's channel (or on the
  feed, for feed_outage) is raised between the label start and its end (or the
  run end) plus GRACE_S.
* Detection delay = time raised minus ``first_detectable`` (negative means the
  detector fired before the effect passed 1 sigma).
* A raised condition that matches no label is a false alarm, except on a
  channel coupled to a channel with a label (it is following its source, which
  is correct behavior).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from sim.io import read_labels
from sim.stream import replay

from .detectors import DetectorSettings
from .engine import Coupling, Engine

GRACE_S = 120.0

# Which detector kinds can explain which label types. A condition is matched to a
# compatible label first, and only to an incompatible one when no compatible label fits.
COMPATIBLE = {
    "spike": {"spike", "spike_burst", "transient_decay"},
    "drift_up": {"step_shift", "linear_drift", "exponential_drift", "transient_decay"},
    "drift_down": {"step_shift", "linear_drift", "exponential_drift", "transient_decay"},
    "stuck": {"stuck_sensor"},
    "missing": {"dropout"},
    "noisy": {"noise_growth"},
    "decoupled": {"decoupling"},
    "feed_outage": {"feed_outage"},
}
CALIBRATION_HOURS = 6.0


@dataclass
class RunScore:
    run: str
    hours: float
    labels: list[dict[str, Any]]
    matched: dict[str, dict[str, Any]] = field(default_factory=dict)   # label id -> {delay_s, kind}
    masked: set[str] = field(default_factory=set)   # labels on a channel that was already in alarm
    false_alarms: list[dict[str, Any]] = field(default_factory=list)


def healthy_config(raw: dict[str, Any], hours: float = CALIBRATION_HOURS) -> dict[str, Any]:
    cfg = copy.deepcopy(raw)
    cfg["anomalies"] = []
    cfg["seed"] = int(raw["seed"]) + 7919
    cfg["duration_seconds"] = hours * 3600.0
    return cfg


def couplings_of(raw: dict[str, Any]) -> list[Coupling]:
    out = []
    for ch in raw["channels"]:
        c = ch.get("couples_to")
        if c:
            out.append(Coupling(ch["name"], c["channel"], float(c.get("gain", 0.15))))
    return out


def _t(iso: str | None) -> datetime | None:
    return None if iso is None else datetime.fromisoformat(iso)


def score_run(run_dir: str | Path, settings: DetectorSettings | None = None, adaptive: bool = True) -> RunScore:
    run_dir = Path(run_dir)
    raw = yaml.safe_load((run_dir / "config_used.yaml").read_text())
    couplings = couplings_of(raw)
    engine = Engine.from_healthy(replay(healthy_config(raw)), adaptive=adaptive, couplings=couplings,
                                 settings=settings)
    last = None
    for t, ev in replay(run_dir):           # data.csv only; labels.json is not opened here
        engine.step(t, ev)
        last = t
    labels = read_labels(run_dir / "labels.json")
    hours = float(raw["duration_seconds"]) / 3600.0
    score = RunScore(run_dir.name, hours, labels)

    followers = {c.channel: c.source for c in couplings}

    def in_window(lb, t):
        start, end = _t(lb["start"]), _t(lb["end"]) or last
        return start <= t and (t - end).total_seconds() <= GRACE_S

    for cond in engine.raised:
        cands = [lb for lb in labels if lb["channel"] == cond.subject and in_window(lb, cond.detected_at)]
        good = [lb for lb in cands if lb["type"] in COMPATIBLE.get(cond.kind, set())]
        free = [lb for lb in good if lb["id"] not in score.matched] or good or                [lb for lb in cands if lb["id"] not in score.matched] or cands
        hit = free[0] if free else None
        if hit is not None:
            if hit["id"] not in score.matched:
                fd = _t(hit["first_detectable"])
                delay = None if fd is None else (cond.detected_at - fd).total_seconds()
                score.matched[hit["id"]] = {"delay_s": delay, "kind": cond.kind}
            continue
        src = followers.get(cond.subject)
        if src is not None and any(lb["channel"] == src for lb in labels):
            continue
        score.false_alarms.append({"subject": cond.subject, "kind": cond.kind,
                                   "at": cond.detected_at.isoformat()})

    # A missed label is "masked" when its channel already had an active condition at the label start
    # (raised for an earlier anomaly): the channel is already reported abnormal, so no new alarm fires.
    for lb in labels:
        if lb["id"] in score.matched:
            continue
        start = _t(lb["start"])
        for cond in engine.raised:
            if cond.subject == lb["channel"] and cond.kind != "spike" and cond.detected_at < start and                     (cond.ended_at is None or cond.ended_at >= start):
                score.masked.add(lb["id"])
                break
    return score


def summarize(scores: list[RunScore], clean_runs: set[str]) -> dict[str, Any]:
    by_type: dict[str, dict[str, Any]] = {}
    for sc in scores:
        for lb in sc.labels:
            d = by_type.setdefault(lb["type"], {"n": 0, "detected": 0, "masked": 0, "delays": []})
            d["n"] += 1
            if lb["id"] in sc.masked:
                d["masked"] += 1
            m = sc.matched.get(lb["id"])
            if m is not None:
                d["detected"] += 1
                if m["delay_s"] is not None:
                    d["delays"].append(m["delay_s"])
    for d in by_type.values():
        d["rate"] = d["detected"] / d["n"] if d["n"] else None
        d["rate_incl_masked"] = (d["detected"] + d["masked"]) / d["n"] if d["n"] else None
        ds = sorted(d.pop("delays"))
        d["median_delay_s"] = ds[len(ds) // 2] if ds else None
    clean = [s for s in scores if s.run in clean_runs]
    clean_hours = sum(s.hours for s in clean)
    clean_fa = sum(len(s.false_alarms) for s in clean)
    all_hours = sum(s.hours for s in scores)
    all_fa = sum(len(s.false_alarms) for s in scores)
    return {
        "by_type": dict(sorted(by_type.items())),
        "clean_runs": len(clean), "clean_hours": clean_hours,
        "false_alarms_per_hour_clean": clean_fa / clean_hours if clean_hours else None,
        "false_alarms_per_hour_all": all_fa / all_hours if all_hours else None,
    }
