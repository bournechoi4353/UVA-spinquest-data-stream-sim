"""Realistic SIMULATED ptdb scenarios with known answers, shared by the end-to-end test and the live simulator.

``realistic_config`` builds the healthy model: fitted to the real profile where one exists
(out/profile/ptdb_profile.json), plausible defaults for channels without one, and the edge cases seen
in the real data (a sentinel-like constant, a channel that barely moves, partial rows with NULLs).

``planted_anomalies`` places seven known problems on real channel names, mostly on QT (the system
in active use for the purifier tests): a pressure drift, a flow spike, a 30 minute Flow_Rates outage,
a temperature step, a stuck temperature, a 15 minute pressure dropout, and a Labjack temperature step.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .from_profile import fit_config, load_profile
from .registry import Registry

START = "2026-05-01T00:00:00"


def realistic_config(reg: Registry, profile_path: Path | None, hours: float, seed: int,
                     start: str = START) -> dict[str, Any]:
    """Healthy model: fitted to the real profile where available, realistic defaults elsewhere."""
    cfg: dict[str, Any] = {"seed": seed, "start_time": start, "duration_seconds": hours * 3600,
                           "synthetic": True, "feeds": {}, "channels": {}}
    if profile_path is not None and Path(profile_path).exists():
        fit = fit_config(load_profile(profile_path), reg)
        cfg["feeds"].update(fit["feeds"])
        cfg["channels"].update(fit["channels"])
    rng = random.Random(seed)
    for c in reg.included_channels():
        if c.key in cfg["channels"]:
            continue
        # Channels without a profile: plausible levels, modest noise, some NULLs.
        base = round(10 ** rng.uniform(-1, 2.5), 3)
        cfg["channels"][c.key] = {"baseline": base, "sigma": round(base * rng.uniform(0.002, 0.01), 6),
                                  "ar1_phi": rng.choice([0.0, 0.3, 0.6]), "null_fraction": 0.3}
    # Edge cases seen in the real data, made explicit so they are always present:
    cfg["channels"]["Labjack.root_exhaust_pressure"] = {"baseline": -41030.7, "sigma": 0.0001,
                                                        "null_fraction": 0.53}      # sentinel-like constant
    cfg["channels"]["Flow_Rates.heat_exchanger_flow"] = {"baseline": 0.00408, "sigma": 1e-8,
                                                         "null_fraction": 0.4}      # barely moves
    return cfg


def planted_anomalies(reg: Registry, hours: float, offset_s: float = 0.0) -> list[dict[str, Any]]:
    """Known problems with known answers, placed in time order through ``hours`` starting at ``offset_s``."""
    D = hours * 3600
    keys = {c.key for c in reg.included_channels()}
    plan = [
        {"channel": "QT.pt501_ai", "type": "linear_drift", "start": 0.10 * D, "duration": 0.15 * D,
         "magnitude": 12, "hold_after": True},
        {"channel": "QT.fc501_ai", "type": "spike", "start": 0.20 * D, "magnitude": 12},
        {"feed": "Flow_Rates", "type": "feed_outage", "start": 0.30 * D, "duration": 1800},
        {"channel": "QT.ti501_ai", "type": "step_shift", "start": 0.40 * D, "duration": 0.15 * D, "magnitude": -10},
        {"channel": "QT.ti502_ai", "type": "stuck_sensor", "start": 0.55 * D, "duration": 1200},
        {"channel": "QT.pt502_ai", "type": "dropout", "start": 0.65 * D, "duration": 900},
        {"channel": "Labjack.magnet_top_temperature", "type": "step_shift", "start": 0.75 * D,
         "duration": 0.1 * D, "magnitude": 15},
    ]
    out = []
    for a in plan:
        if a.get("feed") or a["channel"] in keys:
            a = dict(a)
            a["start"] = round(offset_s + a["start"], 1)
            out.append(a)
    return out
