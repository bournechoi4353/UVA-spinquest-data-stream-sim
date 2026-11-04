"""Ready-made anomaly sets for wide-feed scenarios (Phase R2).

``mixed_anomalies`` draws one feed outage plus a handful of channel anomalies
over the included feeds. Everything comes from a seed stream derived from the
scenario seed, so the same seed always gives the same set.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .registry import Registry

_RECIPE = ["spike", "step_shift", "linear_drift", "stuck_sensor", "dropout", "noise_growth"]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(3_000_000,)))


def random_anomalies(cfg: dict[str, Any], reg: Registry, count: int) -> list[dict[str, Any]]:
    """``count`` anomalies drawn from the mixed recipe (a feed outage counts as one). 0 gives a clean run."""
    if count <= 0:
        return []
    full = mixed_anomalies(cfg, reg)
    rng = _rng(cfg["seed"] + 1)
    idx = sorted(rng.choice(len(full), size=min(count, len(full)), replace=False).tolist())
    return [full[i] for i in idx]


def mixed_anomalies(cfg: dict[str, Any], reg: Registry) -> list[dict[str, Any]]:
    """One feed outage and one each of six channel anomaly types, on distinct channels.

    Windows are placed in the middle 80 percent of the run so labels never
    touch the edges.
    """
    rng = _rng(cfg["seed"])
    D = float(cfg["duration_seconds"])
    lo, hi = 0.1 * D, 0.9 * D
    keys = [c.key for c in reg.included_channels()]
    chosen = rng.choice(len(keys), size=len(_RECIPE), replace=False)
    out: list[dict[str, Any]] = []
    for atype, k in zip(_RECIPE, chosen):
        ch = keys[int(k)]
        if atype == "spike":
            out.append({"channel": ch, "type": "spike", "start": round(float(rng.uniform(lo, hi)), 1),
                        "magnitude": float(rng.choice([-1, 1]) * rng.uniform(7, 10))})
            continue
        dur = round(float(rng.uniform(0.05, 0.15) * D), 1)
        start = round(float(rng.uniform(lo, hi - dur)), 1)
        a: dict[str, Any] = {"channel": ch, "type": atype, "start": start}
        if atype == "step_shift":
            a.update(magnitude=float(rng.choice([-1, 1]) * rng.uniform(4, 8)), duration=dur)
        elif atype == "linear_drift":
            a.update(magnitude=float(rng.choice([-1, 1]) * rng.uniform(5, 10)), duration=dur,
                     hold_after=True)
        elif atype == "stuck_sensor":
            a.update(duration=dur)
        elif atype == "dropout":
            a.update(duration=round(min(dur, float(rng.uniform(60, 600))), 1))
        elif atype == "noise_growth":
            a.update(factor=float(rng.uniform(3, 5)), duration=dur)
        out.append(a)

    feeds = [t.name for t in reg.included_tables()]
    feed = feeds[int(rng.integers(len(feeds)))]
    dur = round(float(rng.uniform(0.02, 0.06) * D), 1)
    out.append({"feed": feed, "type": "feed_outage", "start": round(float(rng.uniform(lo, hi - dur)), 1),
                "duration": dur})
    return out
