"""Synthetic multi-channel data stream simulator with injected anomalies and exact ground truth."""

from .scenario import ScenarioConfig, ScenarioError, load_scenario, parse_scenario
from .stream import StreamResult, generate, replay

__all__ = [
    "ScenarioConfig",
    "ScenarioError",
    "StreamResult",
    "generate",
    "load_scenario",
    "parse_scenario",
    "replay",
]
