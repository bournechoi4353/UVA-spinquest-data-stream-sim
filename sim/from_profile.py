"""Fit the wide-feed simulator to a profiler report (Phase R4).

Input: a profile JSON written by ``scripts/profile_db.py`` (or a trimmed copy
with only the parts that were approved for sharing). It is validated against
the same allowlist schema first. Nothing here connects to a database.

Output: a plain layout-scenario fragment (``feeds`` and ``channels`` sections)
that ``generate_layout`` already understands, plus a fit report.

Mapping
-------
Per channel:
    baseline          the profile median
    sigma, ar1_phi    from noise_mad_sigma and noise_lag1, corrected with a
                      calibration curve (see below)
    slow_wander       amplitude from slow_share, period WANDER_PERIOD_S (the
                      profile does not measure a period)
Per feed:
    nominal_period_s  mean_spacing_s
    jitter_s          from the p5 to p95 spacing spread when timestamps are not
                      floored; otherwise the default (5 percent of the period),
                      because whole-second flooring hides the jitter
    gap_fraction      from long_step_share (steps over 1.5 periods and over the largest normal
                      step). With whole-second timestamps only part of the single missing rows
                      show up as long steps, so this is a rough, low estimate.
Outages (rows absent for a long time) are not part of a healthy run. Their rate
and typical length are put in the fit report, for use when placing labeled
feed_outage anomalies.

Couplings are copied only from a list given by the caller (chosen by a person
from the profile's correlation lists). The fitter never adds one by itself.

Calibration curve
-----------------
The profiler's noise estimate is the MAD sigma of the residual after a 61
sample rolling median, and its lag-1 autocorrelation. For AR(1) noise the
rolling median removes part of the slow noise, so both numbers come out low.
``calibration()`` runs the same estimator on long simulated AR(1) series for a
grid of phi values (fixed seed, so it is deterministic) and inverts the curve:
measured lag-1 gives phi, and the measured sigma is divided by the matching
sigma ratio.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .channels import ar1_noise
from .profiler.schema import validate_profile
from .profiler.stats import mad_sigma, residual
from .registry import Registry

WANDER_PERIOD_S = 7200.0
MAX_PHI = 0.95
CAL_PHIS = np.round(np.arange(0.0, MAX_PHI + 1e-9, 0.05), 2)
CAL_SAMPLES = 40_000
MIN_SIGMA_FRACTION = 1e-6


def load_profile(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_profile(data)


@lru_cache(maxsize=1)
def calibration() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(phi grid, measured lag-1 at each phi, measured sigma / true sigma at each phi)."""
    rng = np.random.default_rng(np.random.SeedSequence(20260101))
    lags, ratios = [], []
    for phi in CAL_PHIS:
        x = ar1_noise(rng.standard_normal(CAL_SAMPLES), 1.0, float(phi))
        r = residual(x)
        ok = np.isfinite(r)
        a, b = r[:-1], r[1:]
        both = ok[:-1] & ok[1:]
        lags.append(float(np.corrcoef(a[both], b[both])[0, 1]))
        ratios.append(mad_sigma(r[ok]))
    lags_arr = np.maximum.accumulate(np.array(lags))   # keep the curve monotonic for inversion
    return CAL_PHIS.astype(float), lags_arr, np.array(ratios)


def invert_noise(measured_sigma: float, measured_lag1: float) -> tuple[float, float]:
    """True (sigma, phi) from the profiler's residual sigma and residual lag-1."""
    phis, lags, ratios = calibration()
    lag = float(np.clip(measured_lag1 if math.isfinite(measured_lag1) else 0.0, lags[0], lags[-1]))
    phi = float(np.interp(lag, lags, phis))
    ratio = float(np.interp(phi, phis, ratios))
    return measured_sigma / ratio, phi


def _num(d: dict, key: str) -> float | None:
    v = d.get(key)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def fit_channel(stats: dict[str, Any]) -> dict[str, Any] | None:
    """Healthy channel model from one channel's profile stats, or None if unusable."""
    median, noise, lag1 = _num(stats, "median"), _num(stats, "noise_mad_sigma"), _num(stats, "noise_lag1")
    if median is None or noise is None:
        return None
    if noise <= 0:
        # Held, quantized or constant channels have no measurable short-term noise. Use a tiny sigma
        # (one millionth of the level) so the channel is simulated as nearly flat, as measured.
        noise, lag1 = 1e-6 * max(abs(median), 1.0), 0.0
    sigma, phi = invert_noise(noise, lag1 if lag1 is not None else 0.0)
    sigma = max(sigma, MIN_SIGMA_FRACTION * max(abs(median), 1.0))
    model: dict[str, Any] = {"baseline": median, "sigma": float(f"{sigma:.6g}"), "ar1_phi": round(phi, 3)}
    nf = _num(stats, "null_fraction")
    if nf is not None and 0 < nf < 0.95:
        model["null_fraction"] = round(nf, 4)
    slow = _num(stats, "slow_share")
    if slow is not None and 0 < slow < 0.99:
        _, _, ratios = calibration()
        r = float(np.interp(phi, CAL_PHIS, ratios))
        # within-chunk variance = sigma^2 + A^2 / 2; residual variance = (r * sigma)^2
        a2 = 2.0 * sigma ** 2 * ((r ** 2) / (1.0 - slow) - 1.0)
        if a2 > 0:
            model["slow_wander"] = {"amplitude": float(f"{math.sqrt(a2):.6g}"),
                                    "period_seconds": WANDER_PERIOD_S}
    return model


def fit_feed(table_stats: dict[str, Any], resolution_s: float | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    period = _num(table_stats, "mean_spacing_s")
    if period and period > 0:
        out["nominal_period_s"] = float(f"{period:.6g}")
        sp = table_stats.get("spacing_s") or {}
        p5, p95 = _num(sp, "p5"), _num(sp, "p95")
        if resolution_s is None and p5 is not None and p95 is not None:
            # spacing = period + (j2 - j1) with j uniform on [0, jitter]: p95 - p5 = 2 * jitter * (1 - sqrt(0.1))
            jitter = (p95 - p5) / (2.0 * (1.0 - math.sqrt(0.1)))
            out["jitter_s"] = float(f"{min(max(jitter, 0.0), 0.45 * period):.6g}")
    long_share = _num(table_stats, "long_step_share")
    if long_share is not None:
        out["gap_fraction"] = float(f"{min(long_share, 0.2):.6g}")
    return out


def fit_config(profile: dict[str, Any], registry: Registry,
               couplings: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Return {'feeds': ..., 'channels': ..., 'couplings': ...} for a layout scenario config."""
    feeds, channels = {}, {}
    included = {t.name: t for t in registry.included_tables()}
    for name, t in (profile.get("tables") or {}).items():
        if name not in included:
            continue
        f = fit_feed(t, included[name].timestamp_resolution_s)
        if f:
            feeds[name] = f
        for col, stats in (t.get("channels") or {}).items():
            model = fit_channel(stats)
            if model is not None:
                channels[f"{name}.{col}"] = model
    out: dict[str, Any] = {"feeds": feeds, "channels": channels}
    if couplings:
        out["couplings"] = [dict(c) for c in couplings]
    return out


def fit_report(profile: dict[str, Any], registry: Registry) -> dict[str, Any]:
    """What was fitted, what was not, and outage statistics for placing feed_outage labels."""
    fitted = fit_config(profile, registry)
    keys = {c.key for c in registry.included_channels()}
    not_fitted = sorted(keys - set(fitted["channels"]))
    outages = {}
    for name, t in (profile.get("tables") or {}).items():
        gaps = [g for g in t.get("largest_gaps") or [] if g.get("length_s")]
        first, last = t.get("first_ts"), t.get("last_ts")
        days = None
        if first and last:
            from datetime import datetime
            days = (datetime.fromisoformat(last) - datetime.fromisoformat(first)).total_seconds() / 86400
        outages[name] = {
            "listed_gaps": len(gaps),
            "median_gap_s": float(np.median([g["length_s"] for g in gaps])) if gaps else None,
            "estimated_downtime_s": (t.get("timeline") or {}).get("estimated_downtime_s"),
            "days_covered": days,
        }
    return {"fitted_channels": len(fitted["channels"]), "not_fitted": not_fitted,
            "null_fraction": {f"{n}.{c}": s.get("null_fraction")
                              for n, t in (profile.get("tables") or {}).items()
                              for c, s in (t.get("channels") or {}).items()},
            "outages": outages}


NOT_CAPTURED = (
    "The fit only matches aggregates. It does not capture: regime changes and operating modes "
    "(cooldown, warmup, beam on or off), bursty or heavy-tailed noise, slow wander shape and "
    "period (a fixed sinusoid is used), daily or weekly cycles, correlations between channels "
    "unless couplings are listed by hand, value-dependent noise, quantization of the readout, "
    "sentinel values for failed reads, and anything that happens only in the unsampled rows."
)
