"""Scenario configuration: loading, defaults, and validation.

A scenario is a YAML file (or a plain dict with the same shape) describing one
synthetic run: the seed, the start time, the duration, the channels, and the
anomalies to inject. Everything is validated up front so that a typo produces a
readable error instead of a confusing failure deep inside the generator.

Example::

    seed: 42
    start_time: "2026-01-01T08:00:00"
    duration_seconds: 21600
    channels:
      - {name: polarization, baseline: 41.0, sigma: 0.15}
    anomalies:
      - {channel: polarization, type: linear_drift, start: 7200,
         duration: 3600, magnitude: -3.5, magnitude_unit: absolute}
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

DEFAULT_COUPLING_GAIN = 0.15
MAGNITUDE_UNITS = ("sigma", "absolute")


class ScenarioError(ValueError):
    """Raised when a scenario config is invalid. The message says what to fix."""


# ---------------------------------------------------------------------------
# Anomaly parameter specs
# ---------------------------------------------------------------------------
# For each anomaly type: which keys are required and which are optional (with
# their defaults). "channel", "type" and "start" are common to every anomaly
# and handled separately. A default of None means "no default": the value is
# filled in later (see _fill_derived_defaults) or the key stays unset.

_MAG = {"magnitude_unit": "sigma"}

ANOMALY_SPECS: dict[str, dict[str, Any]] = {
    "spike": {
        "required": ["magnitude"],
        "optional": {**_MAG},
    },
    "spike_burst": {
        "required": ["magnitude", "count", "spacing"],
        "optional": {**_MAG, "alternate_sign": False},
    },
    "step_shift": {
        "required": ["magnitude"],
        "optional": {**_MAG, "duration": None},
    },
    "linear_drift": {
        "required": ["magnitude", "duration"],
        "optional": {**_MAG, "hold_after": False},
    },
    "exponential_drift": {
        "required": ["magnitude", "duration"],
        "optional": {**_MAG, "hold_after": False, "time_constant": None},
    },
    "noise_growth": {
        "required": ["factor", "duration"],
        "optional": {"hold_after": False},
    },
    "transient_decay": {
        "required": ["magnitude", "time_constant"],
        "optional": {**_MAG},
    },
    "dropout": {
        "required": ["duration"],
        "optional": {},
    },
    "stuck_sensor": {
        "required": [],
        "optional": {"duration": None},
    },
    "decoupling": {
        "required": ["duration"],
        "optional": {"end_gain_fraction": 0.0, "hold_after": False},
    },
}

# Anomalies that corrupt the measurement rather than the physical quantity.
# They are applied last and are NOT seen by channels coupled to this one.
MEASUREMENT_FAULTS = ("stuck_sensor", "dropout")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChannelConfig:
    """Healthy signal model for one channel (see sim/channels.py)."""

    name: str
    baseline: float
    sigma: float
    sample_rate_hz: float = 1.0
    ar1_phi: float = 0.0
    wander_amplitude: float = 0.0
    wander_period_seconds: float = 3600.0
    couples_to: str | None = None
    gain: float = DEFAULT_COUPLING_GAIN

    @property
    def sample_period_us(self) -> int:
        """Time between samples in integer microseconds."""
        return round(1e6 / self.sample_rate_hz)


@dataclass(frozen=True)
class AnomalyConfig:
    """One anomaly to inject. `params` holds every type-specific parameter,
    with defaults filled in, so the label can record "all parameters used"."""

    channel: str
    type: str
    start: float
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioConfig:
    """A fully validated scenario."""

    seed: int
    start_time: datetime
    duration_seconds: float
    channels: tuple[ChannelConfig, ...]
    anomalies: tuple[AnomalyConfig, ...]
    raw: dict[str, Any] = field(default_factory=dict, compare=False)

    def channel(self, name: str) -> ChannelConfig:
        for ch in self.channels:
            if ch.name == name:
                return ch
        raise KeyError(name)

    def n_samples(self, ch: ChannelConfig) -> int:
        """Number of samples a channel produces: sample k is at k / rate seconds."""
        return int(math.floor(self.duration_seconds * ch.sample_rate_hz + 1e-9))


# ---------------------------------------------------------------------------
# Small validation helpers
# ---------------------------------------------------------------------------
def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _require_number(value: Any, where: str, what: str) -> float:
    if not _is_number(value):
        raise ScenarioError(f"{where}: '{what}' must be a finite number, got {value!r}")
    return float(value)


def _check_keys(d: dict, allowed: set[str], where: str) -> None:
    unknown = sorted(set(d) - allowed)
    if unknown:
        raise ScenarioError(
            f"{where}: unknown key(s) {unknown}. Allowed keys: {sorted(allowed)}"
        )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _parse_channel(d: Any, index: int) -> ChannelConfig:
    where = f"channels[{index}]"
    if not isinstance(d, dict):
        raise ScenarioError(f"{where}: each channel must be a mapping, got {d!r}")
    name = d.get("name")
    if not isinstance(name, str) or not name:
        raise ScenarioError(f"{where}: 'name' must be a non-empty string")
    where = f"channels[{index}] ('{name}')"
    _check_keys(
        d,
        {"name", "baseline", "sigma", "sample_rate_hz", "ar1_phi", "slow_wander", "couples_to"},
        where,
    )
    for key in ("baseline", "sigma"):
        if key not in d:
            raise ScenarioError(f"{where}: missing required key '{key}'")
    baseline = _require_number(d["baseline"], where, "baseline")
    sigma = _require_number(d["sigma"], where, "sigma")
    if sigma <= 0:
        raise ScenarioError(f"{where}: 'sigma' must be positive, got {sigma}")
    rate = _require_number(d.get("sample_rate_hz", 1.0), where, "sample_rate_hz")
    if rate <= 0:
        raise ScenarioError(f"{where}: 'sample_rate_hz' must be positive, got {rate}")
    phi = _require_number(d.get("ar1_phi", 0.0), where, "ar1_phi")
    if not 0.0 <= phi < 1.0:
        raise ScenarioError(f"{where}: 'ar1_phi' must be in [0, 1), got {phi}")

    wander = d.get("slow_wander") or {}
    if not isinstance(wander, dict):
        raise ScenarioError(f"{where}: 'slow_wander' must be a mapping")
    _check_keys(wander, {"amplitude", "period_seconds"}, f"{where} slow_wander")
    amp = _require_number(wander.get("amplitude", 0.0), where, "slow_wander.amplitude")
    period = _require_number(wander.get("period_seconds", 3600.0), where, "slow_wander.period_seconds")
    if amp < 0:
        raise ScenarioError(f"{where}: 'slow_wander.amplitude' must be >= 0, got {amp}")
    if period <= 0:
        raise ScenarioError(f"{where}: 'slow_wander.period_seconds' must be positive, got {period}")

    couples_to = None
    gain = DEFAULT_COUPLING_GAIN
    coupling = d.get("couples_to")
    if coupling is not None:
        if not isinstance(coupling, dict):
            raise ScenarioError(f"{where}: 'couples_to' must be a mapping with 'channel' and 'gain'")
        _check_keys(coupling, {"channel", "gain"}, f"{where} couples_to")
        couples_to = coupling.get("channel")
        if not isinstance(couples_to, str) or not couples_to:
            raise ScenarioError(f"{where}: 'couples_to.channel' must be a channel name")
        gain = _require_number(coupling.get("gain", DEFAULT_COUPLING_GAIN), where, "couples_to.gain")

    return ChannelConfig(
        name=name,
        baseline=baseline,
        sigma=sigma,
        sample_rate_hz=rate,
        ar1_phi=phi,
        wander_amplitude=amp,
        wander_period_seconds=period,
        couples_to=couples_to,
        gain=gain,
    )


def _check_coupling(channels: tuple[ChannelConfig, ...]) -> None:
    """Coupling targets must exist, must not form a loop, and the gain must
    leave a positive amount of the channel's own noise."""
    by_name = {c.name: c for c in channels}
    for ch in channels:
        if ch.couples_to is None:
            continue
        if ch.couples_to not in by_name:
            raise ScenarioError(
                f"channel '{ch.name}': couples_to refers to unknown channel '{ch.couples_to}'. "
                f"Known channels: {sorted(by_name)}"
            )
        if ch.couples_to == ch.name:
            raise ScenarioError(f"channel '{ch.name}': a channel cannot couple to itself")
        other = by_name[ch.couples_to]
        coupled_var = (ch.gain * other.sigma) ** 2
        if coupled_var >= ch.sigma**2:
            max_gain = ch.sigma / other.sigma
            raise ScenarioError(
                f"channel '{ch.name}': gain {ch.gain} to '{other.name}' would explain "
                f"{coupled_var**0.5:.4g} of the total sigma {ch.sigma:.4g}, leaving no noise of its "
                f"own. |gain| must be below {max_gain:.4g} (= sigma / sigma of '{other.name}')"
            )
    # Cycle check: follow the couples_to chain from every channel.
    for ch in channels:
        seen = {ch.name}
        cur = ch
        while cur.couples_to is not None:
            if cur.couples_to in seen:
                raise ScenarioError(f"coupling loop detected involving channel '{ch.name}'")
            seen.add(cur.couples_to)
            cur = by_name[cur.couples_to]


def _parse_anomaly(
    d: Any, index: int, by_name: dict[str, ChannelConfig], duration_seconds: float
) -> AnomalyConfig:
    where = f"anomalies[{index}]"
    if not isinstance(d, dict):
        raise ScenarioError(f"{where}: each anomaly must be a mapping, got {d!r}")
    for key in ("channel", "type", "start"):
        if key not in d:
            raise ScenarioError(f"{where}: missing required key '{key}'")
    channel, atype = d["channel"], d["type"]
    if channel not in by_name:
        raise ScenarioError(f"{where}: unknown channel {channel!r}. Known channels: {sorted(by_name)}")
    if atype not in ANOMALY_SPECS:
        raise ScenarioError(f"{where}: unknown anomaly type {atype!r}. Known types: {sorted(ANOMALY_SPECS)}")
    where = f"anomalies[{index}] ({atype} on '{channel}')"
    spec = ANOMALY_SPECS[atype]
    _check_keys(d, {"channel", "type", "start", *spec["required"], *spec["optional"]}, where)
    for key in spec["required"]:
        if key not in d:
            raise ScenarioError(f"{where}: missing required parameter '{key}'")

    start = _require_number(d["start"], where, "start")
    if start < 0:
        raise ScenarioError(f"{where}: 'start' must be >= 0 seconds, got {start}")
    if start >= duration_seconds:
        raise ScenarioError(
            f"{where}: start {start:g} s is outside the scenario duration of {duration_seconds:g} s"
        )

    params: dict[str, Any] = dict(spec["optional"])
    params.update({k: v for k, v in d.items() if k not in ("channel", "type", "start")})

    for key in ("magnitude", "factor", "time_constant", "spacing", "end_gain_fraction"):
        if key in params and params[key] is not None:
            params[key] = _require_number(params[key], where, key)
    if params.get("duration") is not None:
        params["duration"] = _require_number(params["duration"], where, "duration")
        if params["duration"] <= 0:
            raise ScenarioError(f"{where}: 'duration' must be positive, got {params['duration']:g}")
        end = start + params["duration"]
        if end > duration_seconds + 1e-9:
            raise ScenarioError(
                f"{where}: window ends at {end:g} s, after the scenario duration of {duration_seconds:g} s"
            )

    if "magnitude_unit" in params and params["magnitude_unit"] not in MAGNITUDE_UNITS:
        raise ScenarioError(
            f"{where}: 'magnitude_unit' must be one of {list(MAGNITUDE_UNITS)}, got {params['magnitude_unit']!r}"
        )
    for key in ("hold_after", "alternate_sign"):
        if key in params and not isinstance(params[key], bool):
            raise ScenarioError(f"{where}: '{key}' must be true or false, got {params[key]!r}")
    if "magnitude" in params and params["magnitude"] == 0:
        raise ScenarioError(f"{where}: 'magnitude' must be non-zero")
    if params.get("time_constant") is not None and params["time_constant"] <= 0:
        raise ScenarioError(f"{where}: 'time_constant' must be positive, got {params['time_constant']:g}")

    if atype == "spike_burst":
        count = params["count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 2:
            raise ScenarioError(f"{where}: 'count' must be an integer >= 2, got {count!r}")
        if params["spacing"] <= 0:
            raise ScenarioError(f"{where}: 'spacing' must be positive, got {params['spacing']:g}")
        period = 1.0 / by_name[channel].sample_rate_hz
        if params["spacing"] < period - 1e-9:
            raise ScenarioError(
                f"{where}: 'spacing' {params['spacing']:g} s is shorter than the sample period "
                f"{period:g} s of '{channel}', so spikes would land on the same sample"
            )
        last = start + (count - 1) * params["spacing"]
        if last >= duration_seconds:
            raise ScenarioError(
                f"{where}: the last spike would be at {last:g} s, outside the scenario duration "
                f"of {duration_seconds:g} s"
            )
    if atype == "noise_growth" and params["factor"] <= 1:
        raise ScenarioError(f"{where}: 'factor' must be greater than 1, got {params['factor']:g}")
    if atype == "decoupling":
        if by_name[channel].couples_to is None:
            raise ScenarioError(
                f"{where}: channel '{channel}' has no couples_to setting, so there is nothing to decouple"
            )
        if not 0.0 <= params["end_gain_fraction"] < 1.0:
            raise ScenarioError(
                f"{where}: 'end_gain_fraction' must be in [0, 1), got {params['end_gain_fraction']:g}"
            )
    if atype == "exponential_drift" and params["time_constant"] is None:
        params["time_constant"] = params["duration"] / 3.0

    return AnomalyConfig(channel=channel, type=atype, start=start, params=params)


def parse_scenario(raw: dict[str, Any]) -> ScenarioConfig:
    """Validate a config dict and turn it into a ScenarioConfig."""
    if not isinstance(raw, dict):
        raise ScenarioError("the scenario must be a mapping at the top level")
    _check_keys(raw, {"seed", "start_time", "duration_seconds", "channels", "anomalies"}, "scenario")
    for key in ("seed", "start_time", "duration_seconds", "channels"):
        if key not in raw:
            raise ScenarioError(f"scenario: missing required key '{key}'")

    seed = raw["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ScenarioError(f"scenario: 'seed' must be a non-negative integer, got {seed!r}")

    start_raw = raw["start_time"]
    try:
        start_time = start_raw if isinstance(start_raw, datetime) else datetime.fromisoformat(str(start_raw))
    except ValueError:
        raise ScenarioError(
            f"scenario: 'start_time' must be an ISO string like 2026-01-01T08:00:00, got {start_raw!r}"
        ) from None
    if start_time.tzinfo is not None:
        raise ScenarioError("scenario: 'start_time' must not include a timezone offset (use a naive time)")

    duration = _require_number(raw["duration_seconds"], "scenario", "duration_seconds")
    if duration <= 0:
        raise ScenarioError(f"scenario: 'duration_seconds' must be positive, got {duration:g}")

    channels_raw = raw["channels"]
    if not isinstance(channels_raw, list) or not channels_raw:
        raise ScenarioError("scenario: 'channels' must be a non-empty list")
    channels = tuple(_parse_channel(c, i) for i, c in enumerate(channels_raw))
    names = [c.name for c in channels]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ScenarioError(f"scenario: duplicate channel name(s) {dupes}")
    _check_coupling(channels)
    cfg_tmp = ScenarioConfig(seed, start_time, duration, channels, (), {})
    for ch in channels:
        if cfg_tmp.n_samples(ch) < 2:
            raise ScenarioError(
                f"channel '{ch.name}': a {duration:g} s scenario at {ch.sample_rate_hz:g} Hz gives fewer than 2 samples"
            )

    by_name = {c.name: c for c in channels}
    anomalies_raw = raw.get("anomalies") or []
    if not isinstance(anomalies_raw, list):
        raise ScenarioError("scenario: 'anomalies' must be a list")
    anomalies = tuple(_parse_anomaly(a, i, by_name, duration) for i, a in enumerate(anomalies_raw))

    return ScenarioConfig(
        seed=seed,
        start_time=start_time,
        duration_seconds=duration,
        channels=channels,
        anomalies=anomalies,
        raw=copy.deepcopy(raw),
    )


def load_scenario(path: str | Path) -> ScenarioConfig:
    """Read and validate a scenario YAML file."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except OSError as exc:
        raise ScenarioError(f"cannot read config file {path}: {exc}") from None
    except yaml.YAMLError as exc:
        raise ScenarioError(f"{path} is not valid YAML: {exc}") from None
    try:
        return parse_scenario(raw)
    except ScenarioError as exc:
        raise ScenarioError(f"{path}: {exc}") from None
