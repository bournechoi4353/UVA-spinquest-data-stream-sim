"""Healthy baseline signal model for one channel.

A channel's healthy value at sample k is::

    value = baseline
          + wander(t)                      (harmless slow sinusoid)
          + own_noise(k)                   (white or AR(1) noise)
          + gain * source_deviation(t)     (only for coupled channels)

Anomalies are layered on top of these parts (see anomalies.py), so the
ChannelWorkspace below keeps the parts separate until the final assembly.

Time is handled as integer microseconds since the scenario start, which avoids
floating point trouble when channels with different sample rates are merged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .scenario import ChannelConfig, ScenarioConfig


def sample_times_us(cfg: ScenarioConfig, ch: ChannelConfig) -> np.ndarray:
    """Sample times of a channel in integer microseconds since scenario start.

    Sample k is taken at k / sample_rate_hz seconds.
    """
    k = np.arange(cfg.n_samples(ch), dtype=np.float64)
    return np.round(k * 1e6 / ch.sample_rate_hz).astype(np.int64)


def own_noise_sigma(ch: ChannelConfig, source: ChannelConfig | None) -> float:
    """Standard deviation of the channel's OWN noise.

    `sigma` in the config is the channel's TOTAL healthy standard deviation.
    For a coupled channel part of that variance comes from the source channel,
    so the own-noise sigma is sqrt(sigma^2 - (gain * sigma_source)^2).
    Config validation guarantees this is positive.
    """
    if source is None:
        return ch.sigma
    return float(np.sqrt(ch.sigma**2 - (ch.gain * source.sigma) ** 2))


def ar1_noise(z: np.ndarray, sigma: float, phi: float) -> np.ndarray:
    """Turn standard normal draws `z` into noise with stationary std `sigma`.

    The AR(1) recursion is  x[t] = phi * x[t-1] + e[t].  The innovation e has
    std sigma * sqrt(1 - phi^2), which makes the stationary std equal `sigma`.
    The first sample is drawn directly from the stationary distribution.

    With phi = 0 this is plain white noise.

    Implementation note: the recursion is computed with pandas' exponentially
    weighted mean, which evaluates  y[t] = phi * y[t-1] + (1 - phi) * e[t]
    quickly in compiled code. Dividing by (1 - phi) gives back x. The
    tests compare it against a plain Python loop.
    """
    if phi == 0.0:
        return sigma * z
    innovations = sigma * np.sqrt(1.0 - phi**2) * z
    # y[0] must equal (1 - phi) * x[0] so that x[0] = sigma * z[0].
    innovations[0] = sigma * z[0] * (1.0 - phi)
    y = pd.Series(innovations).ewm(alpha=1.0 - phi, adjust=False).mean().to_numpy()
    return y / (1.0 - phi)


@dataclass
class ChannelWorkspace:
    """All the separate ingredients of one channel, before final assembly.

    Anomaly injectors edit `additive`, `noise_scale` and `gain_t`. Measurement
    faults (dropout, stuck sensor) are applied afterwards to the assembled array.
    """

    channel: ChannelConfig
    t_us: np.ndarray  # sample times, integer microseconds
    duration_us: int  # scenario length
    own_sigma: float  # std of the channel's own noise
    wander: np.ndarray  # slow sinusoid (deviation units)
    noise: np.ndarray  # own noise, before any noise_growth
    source: np.ndarray | None  # source channel deviation at our sample times
    # Editable by anomalies:
    additive: np.ndarray  # extra deviation (spikes, steps, drifts, ...)
    noise_scale: np.ndarray  # multiplies `noise` (noise_growth)
    gain_t: np.ndarray  # coupling gain over time (decoupling)

    @property
    def t_s(self) -> np.ndarray:
        """Sample times in seconds."""
        return self.t_us / 1e6

    @property
    def n(self) -> int:
        return len(self.t_us)

    def deviation(self) -> np.ndarray:
        """Physical deviation from baseline: everything except measurement faults."""
        dev = self.wander + self.noise * self.noise_scale + self.additive
        if self.source is not None:
            dev = dev + self.gain_t * self.source
        return dev

    def values(self) -> np.ndarray:
        """Physical value: baseline plus deviation."""
        return self.channel.baseline + self.deviation()


def build_workspace(
    cfg: ScenarioConfig,
    ch: ChannelConfig,
    rng: np.random.Generator,
    source_t_us: np.ndarray | None = None,
    source_dev: np.ndarray | None = None,
    source_cfg: ChannelConfig | None = None,
) -> ChannelWorkspace:
    """Create the healthy parts of a channel.

    For a coupled channel, `source_t_us` and `source_dev` give the source
    channel's physical deviation; it is linearly interpolated onto this
    channel's sample times (the channels may have different sample rates).
    """
    t_us = sample_times_us(cfg, ch)
    t_s = t_us / 1e6
    n = len(t_us)

    own_sigma = own_noise_sigma(ch, source_cfg)
    noise = ar1_noise(rng.standard_normal(n), own_sigma, ch.ar1_phi)
    wander = ch.wander_amplitude * np.sin(2.0 * np.pi * t_s / ch.wander_period_seconds)

    source = None
    if ch.couples_to is not None:
        source = np.interp(t_us, source_t_us, source_dev)

    return ChannelWorkspace(
        channel=ch,
        t_us=t_us,
        duration_us=round(cfg.duration_seconds * 1e6),
        own_sigma=own_sigma,
        wander=wander,
        noise=noise,
        source=source,
        additive=np.zeros(n),
        noise_scale=np.ones(n),
        gain_t=np.full(n, ch.gain),
    )
