"""Allowlist schema for the profile output.

The report is checked against this schema before anything is written. A key
outside the schema, a wrong type, or a list longer than allowed raises
SchemaViolation, so a coding mistake cannot leak raw rows into the file.

Lists: at most 20 items, except the percentile mapping (a fixed dict) and the
two 30-pair correlation lists.
"""

from __future__ import annotations

from typing import Any

NUM = (int, float, type(None))
STR = (str, type(None))
BOOL = (bool, type(None))
MAX_LIST = 20
MAX_CORR = 30


class SchemaViolation(ValueError):
    pass


CHANNEL = {k: NUM for k in ("n", "null_fraction", "min", "max", "mean", "std", "median", "mad_sigma",
                            "noise_mad_sigma", "noise_lag1", "slow_share", "repeat_share",
                            "longest_repeat")}
GAP = {"start": STR, "end": STR, "length_s": NUM, "exact": BOOL}
TABLE = {
    "id_min": NUM, "id_max": NUM, "row_estimate": NUM, "id_min_is_1": BOOL,
    "first_ts": STR, "last_ts": STR,
    "id_step_1_share": NUM, "backward_ts_share": NUM, "duplicate_ts_share": NUM,
    "spacing_s": {k: NUM for k in ("p1", "p5", "p50", "p95", "p99", "max")},
    "spacing_share": {k: NUM for k in ("0", "1", "2", "gt2")},
    "mean_spacing_s": NUM, "gaps_gt_5x_median": NUM, "long_step_share": NUM, "sampled_hours": NUM,
    "timeline": {"probes": NUM, "estimated_downtime_s": NUM},
    "largest_gaps": ("list", GAP, MAX_LIST),
    "channels": ("map", CHANNEL),
}
PAIR = {"a": STR, "b": STR, "r": NUM}
SCHEMA = {
    "banner": STR,
    "source": STR,
    "server": {"version": STR, "system_time_zone": STR, "now": STR, "utc_now": STR},
    "sampling": {"chunks": NUM, "chunk_rows": NUM, "timeline_probes": NUM, "window_hours": NUM,
                 "since": STR, "until": STR},
    "tables": ("map", TABLE),
    "correlations": {"windows": ("list", {"start": STR, "end": STR}, 3), "grid_seconds": NUM,
                     "diff_top30": ("list", PAIR, MAX_CORR), "level_top30": ("list", PAIR, MAX_CORR)},
    "statements_run": NUM,
}
REQUIRED_TOP = {"banner", "tables"}


def _check(value: Any, spec: Any, where: str) -> None:
    if isinstance(spec, tuple) and spec and spec[0] == "list":
        _, item, limit = spec
        if not isinstance(value, list):
            raise SchemaViolation(f"{where}: expected a list")
        if len(value) > limit:
            raise SchemaViolation(f"{where}: {len(value)} items, at most {limit} allowed")
        for i, v in enumerate(value):
            _check(v, item, f"{where}[{i}]")
    elif isinstance(spec, tuple) and spec and spec[0] == "map":
        if not isinstance(value, dict):
            raise SchemaViolation(f"{where}: expected a mapping")
        for k, v in value.items():
            if not isinstance(k, str):
                raise SchemaViolation(f"{where}: keys must be strings")
            _check(v, spec[1], f"{where}.{k}")
    elif isinstance(spec, dict):
        if not isinstance(value, dict):
            raise SchemaViolation(f"{where}: expected a mapping")
        extra = sorted(set(value) - set(spec))
        if extra:
            raise SchemaViolation(f"{where}: key(s) {extra} are not in the allowlist")
        for k, v in value.items():
            _check(v, spec[k], f"{where}.{k}")
    else:
        if isinstance(value, bool) and spec is NUM:
            raise SchemaViolation(f"{where}: expected a number, got a boolean")
        if not isinstance(value, spec):
            raise SchemaViolation(f"{where}: unexpected type {type(value).__name__}")
        if isinstance(value, str) and len(value) > 200:
            raise SchemaViolation(f"{where}: string longer than 200 characters")


def validate_profile(report: dict) -> dict:
    """Raise SchemaViolation unless ``report`` fits the allowlist. Returns it unchanged."""
    _check(report, SCHEMA, "profile")
    missing = REQUIRED_TOP - set(report)
    if missing:
        raise SchemaViolation(f"profile: missing {sorted(missing)}")
    return report
