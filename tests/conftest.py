"""Shared helpers for the tests."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.stream import generate  # noqa: E402

START = "2026-01-01T08:00:00"


def make_config(channels=None, anomalies=None, duration=3600, seed=7):
    """A small config dict. Default: one white-noise channel, rate 1 Hz."""
    if channels is None:
        channels = [{"name": "a", "baseline": 10.0, "sigma": 0.5}]
    return {
        "seed": seed,
        "start_time": START,
        "duration_seconds": duration,
        "channels": copy.deepcopy(channels),
        "anomalies": copy.deepcopy(anomalies or []),
    }


COUPLED_CHANNELS = [
    {"name": "pol", "baseline": 41.0, "sigma": 0.15},
    {"name": "nmr", "baseline": 2.5, "sigma": 0.03, "sample_rate_hz": 0.2,
     "couples_to": {"channel": "pol", "gain": 0.15}},
]


def effect_of(anomalies, channel="a", channels=None, duration=3600, seed=7):
    """Return (effect, healthy_values, t_s) for one channel.

    Because each channel's noise comes from its own seed stream, running with and
    without the anomalies and subtracting gives the injected effect exactly.
    """
    base = generate(make_config(channels, [], duration, seed)).channels[channel]
    with_anoms = generate(make_config(channels, anomalies, duration, seed)).channels[channel]
    return with_anoms.values - base.values, base.values, base.t_us / 1e6


@pytest.fixture
def cfg_factory():
    return make_config
