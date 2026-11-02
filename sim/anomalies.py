"""Anomaly injectors.

Each injector takes a ChannelWorkspace and an AnomalyConfig, edits the
workspace (or the assembled array, for measurement faults), and returns an
`Injection` describing what it did. labels.py turns that into a label record.

Two kinds of injector:

* Physical anomalies change the real quantity: spike, spike_burst, step_shift,
  linear_drift, exponential_drift, noise_growth, transient_decay, decoupling.
  Channels coupled to this one DO see them.
* Measurement faults corrupt only what is reported: stuck_sensor, dropout.
  They are applied last and are invisible to coupled channels.

Anomalies compose: injectors only add to (or multiply into) the workspace, so
any number of them can be active on one channel at the same time.

Window conventions (all times are on the sample grid):
* Ramps (linear_drift, exponential_drift, noise_growth, decoupling) use the
  closed window [start, end]: the sample AT `end` has the full effect.
* Everything else uses the half-open window [start, end).
* With `hold_after: true` a ramp keeps its final effect after `end`, and the
  label `end` is null because the effect never goes away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .channels import ChannelWorkspace
from .scenario import AnomalyConfig, ScenarioError


@dataclass
class Injection:
    """What an injector did, in the units labels.py needs.

    Times are integer microseconds since the scenario start. `end_us` is None
    when the effect never ends. `first_detectable_us` is the first sample at
    which the injected effect exceeds 1 sigma, or None if it never does.
    """

    start_us: int
    end_us: int | None
    first_detectable_us: int | None
    magnitude: float | None  # in channel units, None when not meaningful
    magnitude_is_sigma_scaled: bool = True  # False for noise_growth (a plain factor)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _abs_magnitude(ws: ChannelWorkspace, params: dict) -> float:
    """Convert the configured magnitude to channel units."""
    if params.get("magnitude_unit", "sigma") == "sigma":
        return params["magnitude"] * ws.channel.sigma
    return params["magnitude"]


def _start_us(anomaly: AnomalyConfig) -> int:
    return round(anomaly.start * 1e6)


def _end_us(anomaly: AnomalyConfig) -> int | None:
    duration = anomaly.params.get("duration")
    return None if duration is None else round((anomaly.start + duration) * 1e6)


def _describe(anomaly: AnomalyConfig) -> str:
    return f"{anomaly.type} on '{anomaly.channel}' at {anomaly.start:g} s"


def _check_has_samples(mask: np.ndarray, anomaly: AnomalyConfig) -> None:
    if not mask.any():
        raise ScenarioError(
            f"{_describe(anomaly)}: the window contains no samples of this channel "
            f"(it is shorter than the sample period); lengthen it"
        )


def _first_exceeding(effect: np.ndarray, t_us: np.ndarray, sigma: float, mask: np.ndarray | None = None):
    """Time of the first sample where |effect| > sigma, or None."""
    over = np.abs(effect) > sigma
    if mask is not None:
        over &= mask
    if not over.any():
        return None
    return int(t_us[np.argmax(over)])


def _ramp_progress(ws: ChannelWorkspace, anomaly: AnomalyConfig):
    """Return (start_us, end_us, in_window, progress, after) for a ramp anomaly.

    `progress` runs from 0 at start to 1 at end (only meaningful in the window).
    `after` marks samples strictly after the window.
    """
    start_us, end_us = _start_us(anomaly), _end_us(anomaly)
    in_window = (ws.t_us >= start_us) & (ws.t_us <= end_us)
    _check_has_samples(in_window, anomaly)
    progress = np.clip((ws.t_us - start_us) / (end_us - start_us), 0.0, 1.0)
    after = ws.t_us > end_us
    return start_us, end_us, in_window, progress, after


def _add_ramp_effect(ws, anomaly, shape: Callable[[np.ndarray], np.ndarray]) -> Injection:
    """Shared logic for linear and exponential drift."""
    mag = _abs_magnitude(ws, anomaly.params)
    start_us, end_us, in_window, progress, after = _ramp_progress(ws, anomaly)
    hold = anomaly.params["hold_after"]
    effect = np.where(in_window, mag * shape(progress), 0.0)
    if hold:
        effect = np.where(after, mag, effect)
    ws.additive += effect
    return Injection(
        start_us=start_us,
        end_us=None if hold else end_us,
        first_detectable_us=_first_exceeding(effect, ws.t_us, ws.channel.sigma),
        magnitude=mag,
    )


# ---------------------------------------------------------------------------
# Physical anomalies
# ---------------------------------------------------------------------------
def _spike_indices(ws: ChannelWorkspace, anomaly: AnomalyConfig, times_s: list[float]) -> list[int]:
    rate = ws.channel.sample_rate_hz
    return [min(int(round(t * rate)), ws.n - 1) for t in times_s]


def inject_spike(ws: ChannelWorkspace, anomaly: AnomalyConfig) -> Injection:
    """A single-sample jump. Lands on the sample nearest to `start`."""
    mag = _abs_magnitude(ws, anomaly.params)
    (i,) = _spike_indices(ws, anomaly, [anomaly.start])
    ws.additive[i] += mag
    t = int(ws.t_us[i])
    return Injection(
        start_us=t,
        end_us=t + ws.channel.sample_period_us,
        first_detectable_us=t if abs(mag) > ws.channel.sigma else None,
        magnitude=mag,
    )


def inject_spike_burst(ws: ChannelWorkspace, anomaly: AnomalyConfig) -> Injection:
    """`count` spikes, `spacing` seconds apart, starting at `start`."""
    p = anomaly.params
    mag = _abs_magnitude(ws, p)
    times = [anomaly.start + k * p["spacing"] for k in range(p["count"])]
    idx = _spike_indices(ws, anomaly, times)
    for k, i in enumerate(idx):
        sign = -1.0 if (p["alternate_sign"] and k % 2 == 1) else 1.0
        ws.additive[i] += sign * mag
    return Injection(
        start_us=int(ws.t_us[idx[0]]),
        end_us=int(ws.t_us[idx[-1]]) + ws.channel.sample_period_us,
        first_detectable_us=int(ws.t_us[idx[0]]) if abs(mag) > ws.channel.sigma else None,
        magnitude=mag,
    )


def inject_step_shift(ws: ChannelWorkspace, anomaly: AnomalyConfig) -> Injection:
    """The level jumps by `magnitude` at `start` and stays until `end` (or forever)."""
    mag = _abs_magnitude(ws, anomaly.params)
    start_us, end_us = _start_us(anomaly), _end_us(anomaly)
    mask = ws.t_us >= start_us
    if end_us is not None:
        mask &= ws.t_us < end_us
    _check_has_samples(mask, anomaly)
    effect = np.where(mask, mag, 0.0)
    ws.additive += effect
    return Injection(
        start_us=start_us,
        end_us=end_us,
        first_detectable_us=_first_exceeding(effect, ws.t_us, ws.channel.sigma),
        magnitude=mag,
    )


def inject_linear_drift(ws: ChannelWorkspace, anomaly: AnomalyConfig) -> Injection:
    """The level moves linearly by `magnitude` over `duration`."""
    return _add_ramp_effect(ws, anomaly, lambda s: s)


def inject_exponential_drift(ws: ChannelWorkspace, anomaly: AnomalyConfig) -> Injection:
    """Slow start, accelerating growth, reaching `magnitude` at `end`.

    The shape is (exp(D*s/tau) - 1) / (exp(D/tau) - 1) for progress s in [0, 1],
    written in an overflow-safe form. A small `time_constant` (relative to the
    duration) gives a late, sharp rise.
    """
    ratio = anomaly.params["duration"] / anomaly.params["time_constant"]

    def shape(s: np.ndarray) -> np.ndarray:
        return (np.exp((s - 1.0) * ratio) - np.exp(-ratio)) / (1.0 - np.exp(-ratio))

    return _add_ramp_effect(ws, anomaly, shape)


def inject_noise_growth(ws: ChannelWorkspace, anomaly: AnomalyConfig) -> Injection:
    """The channel's own noise sigma grows linearly from 1x to `factor`x.

    The mean is untouched: the factor multiplies only the noise part. For the
    label, "detectable" means the EXTRA noise std, own_sigma * sqrt(f^2 - 1),
    exceeds 1 channel sigma.
    """
    factor = anomaly.params["factor"]
    start_us, end_us, in_window, progress, after = _ramp_progress(ws, anomaly)
    hold = anomaly.params["hold_after"]
    f = np.where(in_window, 1.0 + (factor - 1.0) * progress, 1.0)
    if hold:
        f = np.where(after, factor, f)
    ws.noise_scale *= f
    extra_std = ws.own_sigma * np.sqrt(np.maximum(f**2 - 1.0, 0.0))
    return Injection(
        start_us=start_us,
        end_us=None if hold else end_us,
        first_detectable_us=_first_exceeding(extra_std, ws.t_us, ws.channel.sigma),
        magnitude=factor,
        magnitude_is_sigma_scaled=False,
    )


def inject_transient_decay(ws: ChannelWorkspace, anomaly: AnomalyConfig) -> Injection:
    """A jump of `magnitude` at `start` that decays as exp(-t / time_constant).

    The decay is not cut off. The label `end` is start + 5 time constants (by
    then under 0.7 percent of the jump is left), capped at the scenario end.
    """
    mag = _abs_magnitude(ws, anomaly.params)
    tau_us = anomaly.params["time_constant"] * 1e6
    start_us = _start_us(anomaly)
    after_start = ws.t_us >= start_us
    _check_has_samples(after_start, anomaly)
    effect = np.where(after_start, mag * np.exp(-(ws.t_us - start_us) / tau_us), 0.0)
    ws.additive += effect
    return Injection(
        start_us=start_us,
        end_us=min(int(start_us + 5 * tau_us), ws.duration_us),
        first_detectable_us=_first_exceeding(effect, ws.t_us, ws.channel.sigma),
        magnitude=mag,
    )


def inject_decoupling(ws: ChannelWorkspace, anomaly: AnomalyConfig) -> Injection:
    """The coupling gain falls linearly toward `end_gain_fraction` x gain.

    Each channel still looks fine on its own; only the relationship to the
    source channel breaks. `first_detectable` is the first time the missing
    coupled contribution, (gain - gain(t)) * source deviation, exceeds 1 sigma.
    """
    end_frac = anomaly.params["end_gain_fraction"]
    start_us, end_us, in_window, progress, after = _ramp_progress(ws, anomaly)
    hold = anomaly.params["hold_after"]
    frac = np.where(in_window, 1.0 - (1.0 - end_frac) * progress, 1.0)
    if hold:
        frac = np.where(after, end_frac, frac)
    ws.gain_t *= frac
    missing = ws.channel.gain * (1.0 - frac) * ws.source
    return Injection(
        start_us=start_us,
        end_us=None if hold else end_us,
        first_detectable_us=_first_exceeding(missing, ws.t_us, ws.channel.sigma),
        magnitude=None,
    )


# ---------------------------------------------------------------------------
# Measurement faults (applied to the assembled array)
# ---------------------------------------------------------------------------
def _fault_mask(ws: ChannelWorkspace, anomaly: AnomalyConfig):
    start_us, end_us = _start_us(anomaly), _end_us(anomaly)
    mask = ws.t_us >= start_us
    if end_us is not None:
        mask &= ws.t_us < end_us
    _check_has_samples(mask, anomaly)
    return start_us, end_us, mask


def inject_stuck_sensor(
    observed: np.ndarray, physical: np.ndarray, ws: ChannelWorkspace, anomaly: AnomalyConfig
) -> Injection:
    """The reported value freezes at its last good value (zero noise).

    `magnitude` is the frozen value's deviation from the baseline.
    `first_detectable` is the first sample where the frozen value differs from
    the true (physical) value by more than 1 sigma.
    """
    start_us, end_us, mask = _fault_mask(ws, anomaly)
    first = int(np.argmax(mask))
    candidates = observed[:first][np.isfinite(observed[:first])]
    frozen = float(candidates[-1]) if len(candidates) else float(physical[first])
    observed[mask] = frozen
    return Injection(
        start_us=start_us,
        end_us=end_us,
        first_detectable_us=_first_exceeding(frozen - physical, ws.t_us, ws.channel.sigma, mask),
        magnitude=frozen - ws.channel.baseline,
    )


def inject_dropout(
    observed: np.ndarray, physical: np.ndarray, ws: ChannelWorkspace, anomaly: AnomalyConfig
) -> Injection:
    """Samples inside the window become NaN (missing data)."""
    start_us, end_us, mask = _fault_mask(ws, anomaly)
    observed[mask] = np.nan
    return Injection(
        start_us=start_us,
        end_us=end_us,
        first_detectable_us=int(ws.t_us[np.argmax(mask)]),
        magnitude=None,
    )


PHYSICAL_INJECTORS: dict[str, Callable[[ChannelWorkspace, AnomalyConfig], Injection]] = {
    "spike": inject_spike,
    "spike_burst": inject_spike_burst,
    "step_shift": inject_step_shift,
    "linear_drift": inject_linear_drift,
    "exponential_drift": inject_exponential_drift,
    "noise_growth": inject_noise_growth,
    "transient_decay": inject_transient_decay,
    "decoupling": inject_decoupling,
}

# Order matters: stuck sensors freeze first, then dropouts blank samples.
FAULT_INJECTORS = {
    "stuck_sensor": inject_stuck_sensor,
    "dropout": inject_dropout,
}
