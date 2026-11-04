"""Channel registry for the real ptdb layout: loading and validation.

The registry is the single source of truth for table names, column names, and
channel metadata. Later phases (the simulator's wide feeds, the SQLite fixture,
the profiler) read names from here instead of typing them by hand.

A channel's key is ``<table>.<db_column>``, for example
``Labjack.fridge_vapor_pressure``. Display names are for people only and are
never used as keys.

Example::

    from sim.registry import load_registry
    reg = load_registry("configs/ptdb_layout.yaml")
    for t in reg.included_tables():
        print(t.name, t.nominal_period_s, [c.key for c in t.channels])

Every check runs inside ``load_registry`` (or ``parse_registry`` for a dict), so a
bad layout fails at load time with a ``RegistryError`` that names the table and
column at fault.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .ptdb_schema import EXPECTED_SCHEMA, ID_COLUMN

TIME_TYPES = ("timestamp", "datetime")
GROUPS = ("temperature", "pressure", "flow", "other")
ROLES = ("sensor", "controller_output")
STATUSES = ("named", "inferred", "unnamed")
SQL_TYPES = ("float", "int", "text")

_TOP_KEYS = {"version", "database", "tables"}
_TABLE_REQUIRED = {"name", "id_column", "time_column", "time_type", "time_zone",
                   "time_zone_assumed", "timestamp_resolution_s", "include",
                   "nominal_period_s", "channels"}
_TABLE_KEYS = _TABLE_REQUIRED | {"reason", "notes"}
_CHANNEL_REQUIRED = {"db_column", "display_name", "unit", "group", "group_inferred",
                     "role", "registry_status"}
_CHANNEL_KEYS = _CHANNEL_REQUIRED | {"sql_type", "notes"}


class RegistryError(ValueError):
    """Raised when the layout file is invalid. The message says what to fix."""


@dataclass(frozen=True)
class Channel:
    """One value column of one table."""

    table: str
    db_column: str
    display_name: str
    unit: str | None
    group: str
    group_inferred: bool
    role: str
    registry_status: str
    sql_type: str = "float"
    notes: str | None = None

    @property
    def key(self) -> str:
        """The channel's name inside the system: ``<table>.<db_column>``."""
        return f"{self.table}.{self.db_column}"


@dataclass(frozen=True)
class Table:
    """One database table. An included table is a feed: its channels share one time series."""

    name: str
    id_column: str
    time_column: str
    time_type: str
    time_zone: str
    time_zone_assumed: bool
    timestamp_resolution_s: float | None
    include: bool
    nominal_period_s: float | None
    channels: tuple[Channel, ...]
    reason: str | None = None
    notes: str | None = None

    @property
    def column_count(self) -> int:
        """id + time column + value columns."""
        return 2 + len(self.channels)

    def select_columns(self) -> list[str]:
        """The exact column list any SQL query for this table should use, in order."""
        return [self.id_column, self.time_column] + [c.db_column for c in self.channels]


@dataclass(frozen=True)
class Registry:
    """All tables of the layout, in file order."""

    version: int
    database: str
    tables: tuple[Table, ...]

    def table(self, name: str) -> Table:
        """Look up a table by its exact name."""
        for t in self.tables:
            if t.name == name:
                return t
        raise KeyError(f"no table named {name!r}")

    def included_tables(self) -> list[Table]:
        """The feeds used by the first version (``include: true``)."""
        return [t for t in self.tables if t.include]

    def channels(self, table: str) -> list[Channel]:
        """The channels of one table, in column order."""
        return list(self.table(table).channels)

    def included_channels(self) -> list[Channel]:
        """Every channel of every included table."""
        return [c for t in self.included_tables() for c in t.channels]

    def channel(self, key: str) -> Channel:
        """Look up one channel by its ``<table>.<db_column>`` key."""
        table, _, column = key.partition(".")
        for c in self.table(table).channels:
            if c.db_column == column:
                return c
        raise KeyError(f"no channel {key!r}")

    def select_columns(self, table: str) -> list[str]:
        """Shortcut for ``registry.table(name).select_columns()``."""
        return self.table(table).select_columns()

    @property
    def total_columns(self) -> int:
        """Columns across all tables, including id and time columns."""
        return sum(t.column_count for t in self.tables)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _check_keys(d: Any, required: set[str], allowed: set[str], where: str) -> None:
    if not isinstance(d, dict):
        raise RegistryError(f"{where}: expected a mapping, got {type(d).__name__}")
    unknown = sorted(set(d) - allowed)
    if unknown:
        raise RegistryError(f"{where}: unknown key(s) {unknown}. Allowed keys: {sorted(allowed)}")
    missing = sorted(required - set(d))
    if missing:
        raise RegistryError(f"{where}: missing key(s) {missing}")


def _choice(value: Any, options: tuple[str, ...], where: str) -> str:
    if value not in options:
        raise RegistryError(f"{where}: {value!r} is not one of {list(options)}")
    return value


def _bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise RegistryError(f"{where}: expected true or false, got {value!r}")
    return value


def _text(value: Any, where: str, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RegistryError(f"{where}: expected a nonempty string, got {value!r}")
    return value


def _positive_or_none(value: Any, where: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value) or value <= 0:
        raise RegistryError(f"{where}: expected a positive number or null, got {value!r}")
    return float(value)


def _parse_channel(d: Any, table: str, index: int) -> Channel:
    where = f"table {table}, channel #{index}"
    _check_keys(d, _CHANNEL_REQUIRED, _CHANNEL_KEYS, where)
    col = _text(d["db_column"], where + ", db_column")
    where = f"table {table}, channel {col}"
    return Channel(
        table=table,
        db_column=col,
        display_name=_text(d["display_name"], where + ", display_name"),
        unit=_text(d["unit"], where + ", unit", allow_none=True),
        group=_choice(d["group"], GROUPS, where + ", group"),
        group_inferred=_bool(d["group_inferred"], where + ", group_inferred"),
        role=_choice(d["role"], ROLES, where + ", role"),
        registry_status=_choice(d["registry_status"], STATUSES, where + ", registry_status"),
        sql_type=_choice(d.get("sql_type", "float"), SQL_TYPES, where + ", sql_type"),
        notes=_text(d.get("notes"), where + ", notes", allow_none=True),
    )


def _parse_table(d: Any, index: int) -> Table:
    _check_keys(d, _TABLE_REQUIRED, _TABLE_KEYS, f"table #{index}")
    name = _text(d["name"], f"table #{index}, name")
    where = f"table {name}"

    time_column = _text(d["time_column"], where + ", time_column", allow_none=True)
    include = _bool(d["include"], where + ", include")
    period = _positive_or_none(d["nominal_period_s"], where + ", nominal_period_s")
    reason = _text(d.get("reason"), where + ", reason", allow_none=True)

    if include and time_column is None:
        raise RegistryError(f"{where}: an included feed needs a time_column")
    if include and period is None:
        raise RegistryError(f"{where}: an included feed needs a nominal_period_s")
    if not include and reason is None:
        raise RegistryError(f"{where}: a table with include: false needs a reason")

    raw_channels = d["channels"]
    if not isinstance(raw_channels, list) or not raw_channels:
        raise RegistryError(f"{where}: channels must be a nonempty list")
    channels = tuple(_parse_channel(c, name, i) for i, c in enumerate(raw_channels))

    id_column = _text(d["id_column"], where + ", id_column")
    seen: set[str] = {id_column, time_column or ""}
    for c in channels:
        if c.db_column in seen:
            raise RegistryError(f"{where}: duplicate column {c.db_column!r}")
        seen.add(c.db_column)

    return Table(
        name=name,
        id_column=id_column,
        time_column=time_column,
        time_type=_choice(d["time_type"], TIME_TYPES, where + ", time_type"),
        time_zone=_text(d["time_zone"], where + ", time_zone"),
        time_zone_assumed=_bool(d["time_zone_assumed"], where + ", time_zone_assumed"),
        timestamp_resolution_s=_positive_or_none(d["timestamp_resolution_s"],
                                                 where + ", timestamp_resolution_s"),
        include=include,
        nominal_period_s=period,
        channels=channels,
        reason=reason,
        notes=_text(d.get("notes"), where + ", notes", allow_none=True),
    )


def _check_against_schema(tables: tuple[Table, ...], schema: dict) -> None:
    """Every table and column of the real schema must be listed, and nothing else."""
    problems: list[str] = []
    names = {t.name for t in tables}
    for missing in sorted(set(schema) - names):
        problems.append(f"table {missing} is missing from the layout")
    for extra in sorted(names - set(schema)):
        problems.append(f"table {extra} is not in the real schema")

    for t in tables:
        if t.name not in schema:
            continue
        time_col, time_type, cols = schema[t.name]
        if t.id_column != ID_COLUMN:
            problems.append(f"table {t.name}: id_column should be {ID_COLUMN!r}")
        if t.time_column != time_col:
            problems.append(f"table {t.name}: time_column should be {time_col!r}")
        if t.time_type != time_type:
            problems.append(f"table {t.name}: time_type should be {time_type!r}")
        expected = dict(cols)
        listed = {c.db_column: c.sql_type for c in t.channels}
        for col in sorted(set(expected) - set(listed)):
            problems.append(f"table {t.name}: column {col} is missing")
        for col in sorted(set(listed) - set(expected)):
            problems.append(f"table {t.name}: column {col} is not in the real schema")
        for col in sorted(set(expected) & set(listed)):
            if expected[col] != listed[col]:
                problems.append(f"table {t.name}: column {col} should have sql_type "
                                f"{expected[col]!r}, not {listed[col]!r}")
    if problems:
        raise RegistryError("layout does not match the real schema:\n  " + "\n  ".join(problems))


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def parse_registry(d: Any, schema: dict | None = EXPECTED_SCHEMA) -> Registry:
    """Validate a layout dict and return a Registry.

    ``schema`` is the reference the layout must match exactly. Pass ``None`` to
    skip that check (useful for small test layouts).
    """
    _check_keys(d, _TOP_KEYS, _TOP_KEYS, "layout")
    version = d["version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise RegistryError(f"layout, version: expected an integer, got {version!r}")
    if not isinstance(d["tables"], list) or not d["tables"]:
        raise RegistryError("layout, tables: must be a nonempty list")

    tables = tuple(_parse_table(t, i) for i, t in enumerate(d["tables"]))
    seen: set[str] = set()
    for t in tables:
        if t.name in seen:
            raise RegistryError(f"duplicate table name {t.name!r}")
        seen.add(t.name)

    if schema is not None:
        _check_against_schema(tables, schema)
    return Registry(version=version, database=_text(d["database"], "layout, database"),
                    tables=tables)


def load_registry(path: str | Path, schema: dict | None = EXPECTED_SCHEMA) -> Registry:
    """Load and validate a layout YAML file."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return parse_registry(data, schema)
