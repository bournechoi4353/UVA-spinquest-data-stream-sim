"""Ground truth label records.

Every injected anomaly produces one label. Labels are computed from the
injected model only (never from any detector). The replay interface never
exposes them; they are available only through StreamResult.labels or by
reading labels.json.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from .anomalies import Injection
from .scenario import AnomalyConfig, ChannelConfig

HIGH_SEVERITY_TYPES = {
    "linear_drift",
    "exponential_drift",
    "step_shift",
    "noise_growth",
    "decoupling",
    "stuck_sensor",
    "spike_burst",
    "transient_decay",
}
SHORT_DROPOUT_SECONDS = 5.0


def severity_for(anomaly: AnomalyConfig) -> str:
    """"high" or "low" following the project rules.

    Low: an isolated spike, or a dropout shorter than 5 s. Everything else is high.
    """
    if anomaly.type == "spike":
        return "low"
    if anomaly.type == "dropout":
        return "low" if anomaly.params["duration"] < SHORT_DROPOUT_SECONDS else "high"
    return "high"


def _iso(start_time: datetime, offset_us: int | None) -> str | None:
    if offset_us is None:
        return None
    return (start_time + timedelta(microseconds=int(offset_us))).isoformat()


def make_label(
    event_id: str,
    channel: ChannelConfig,
    anomaly: AnomalyConfig,
    injection: Injection,
    start_time: datetime,
) -> dict[str, Any]:
    """Build the label record (a plain dict, ready for JSON)."""
    magnitude = injection.magnitude
    magnitude_sigma = None
    if magnitude is not None:
        if injection.magnitude_is_sigma_scaled:
            magnitude_sigma = round(magnitude / channel.sigma, 6)
        magnitude = round(magnitude, 9)
    params = {"start_offset_seconds": anomaly.start, **anomaly.params}
    return {
        "id": event_id,
        "channel": anomaly.channel,
        "type": anomaly.type,
        "start": _iso(start_time, injection.start_us),
        "end": _iso(start_time, injection.end_us),
        "first_detectable": _iso(start_time, injection.first_detectable_us),
        "magnitude": magnitude,
        "magnitude_sigma": magnitude_sigma,
        "params": params,
        "severity": severity_for(anomaly),
    }
