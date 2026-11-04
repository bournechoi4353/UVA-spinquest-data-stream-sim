"""Phase R1: the ptdb layout file and the registry validator."""

import copy
from pathlib import Path

import pytest
import yaml

from sim.ptdb_schema import EXPECTED_SCHEMA, EXPECTED_TOTAL_COLUMNS
from sim.registry import RegistryError, load_registry, parse_registry

ROOT = Path(__file__).resolve().parent.parent
LAYOUT = ROOT / "configs" / "ptdb_layout.yaml"

UNNAMED = (
    [f"Lakeshore_Magnet_Temp.magnet_channel_{i}" for i in range(1, 9)]
    + [f"MaxiGauge.maxigauge_channel_{i}" for i in range(3, 7)]
    + [f"stress_test_data.col{i}" for i in range(1, 6)]
)


@pytest.fixture(scope="module")
def reg():
    return load_registry(LAYOUT)


@pytest.fixture
def raw():
    """A fresh, mutable copy of the layout dict for the bad-case tests."""
    with open(LAYOUT, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _table(raw, name):
    return next(t for t in raw["tables"] if t["name"] == name)


# ---------------------------------------------------------------------------
# The real layout
# ---------------------------------------------------------------------------

def test_layout_has_109_columns(reg):
    assert EXPECTED_TOTAL_COLUMNS == 109
    assert reg.total_columns == 109


def test_layout_lists_every_table_of_the_schema(reg):
    assert {t.name for t in reg.tables} == set(EXPECTED_SCHEMA)


def test_included_feeds_follow_the_mentor(reg):
    # 2026-11-03: Lakeshore_Fridge_Temp sensors are not connected yet; QT is the place to start.
    assert [t.name for t in reg.included_tables()] == ["Flow_Rates", "Labjack", "QT"]
    assert len(reg.included_channels()) == 29
    assert [len(t.channels) for t in reg.included_tables()] == [5, 8, 16]
    assert "not connected" in reg.table("Lakeshore_Fridge_Temp").reason


def test_included_feeds_have_period_time_column_and_resolution(reg):
    periods = {t.name: t.nominal_period_s for t in reg.included_tables()}
    assert periods == {"Flow_Rates": 1.354, "Labjack": 1.105, "QT": 1.878}   # all measured while running
    for t in reg.included_tables():
        assert t.time_column == "Timestamp"
        assert t.timestamp_resolution_s == 1


def test_every_excluded_table_has_a_reason(reg):
    for t in reg.tables:
        if not t.include:
            assert t.reason


def test_included_channels_have_display_names(reg):
    for c in reg.included_channels():
        assert c.display_name.strip()


def test_unnamed_channels_are_flagged(reg):
    flagged = [c.key for t in reg.tables for c in t.channels if c.registry_status == "unnamed"]
    assert sorted(flagged) == sorted(UNNAMED)


def test_typos_kept_in_db_column_and_fixed_only_for_seperator(reg):
    c = reg.channel("Flow_Rates.seperator_flow")
    assert c.display_name == "Separator flow"
    buffle = reg.channel("Lakeshore_Target_Stick.target_stick_buffle_top_temperature")
    assert "buffle" in buffle.display_name.lower()
    assert "TODO" in buffle.notes


def test_units_are_unknown_and_groups_inferred(reg):
    for t in reg.tables:
        for c in t.channels:
            assert c.unit is None
            assert c.group_inferred is True


def test_qt_out_columns_are_controller_outputs(reg):
    outs = {c.key for c in reg.channels("QT") if c.role == "controller_output"}
    assert outs == {"QT.fc501_out", "QT.fc502_out"}


def test_datetime_tables_mark_their_zone_as_assumed(reg):
    for t in reg.tables:
        assert t.time_zone_assumed == (t.time_type == "datetime")
    assert reg.table("sensor_data").time_column == "timestamp"


def test_channel_key_and_select_columns(reg):
    c = reg.channel("Labjack.fridge_vapor_pressure")
    assert c.key == "Labjack.fridge_vapor_pressure"
    cols = reg.select_columns("Labjack")
    assert cols[:2] == ["id", "Timestamp"]
    assert len(cols) == 10


def test_nmr_keeps_non_float_types(reg):
    types = {c.db_column: c.sql_type for c in reg.channels("NMR")}
    assert types["measurement_type"] == "text"
    assert types["run_number"] == types["beam_on"] == types["nmr_channel"] == "int"


def test_unknown_lookup_raises(reg):
    with pytest.raises(KeyError):
        reg.channel("Labjack.nope")
    with pytest.raises(KeyError):
        reg.table("Nope")


# ---------------------------------------------------------------------------
# The validator rejects bad layouts
# ---------------------------------------------------------------------------

def _rejects(raw, match):
    with pytest.raises(RegistryError, match=match):
        parse_registry(raw)


def test_rejects_unknown_top_level_key(raw):
    raw["extra"] = 1
    _rejects(raw, "unknown key")


def test_rejects_unknown_table_key(raw):
    _table(raw, "Labjack")["sample_rate"] = 1
    _rejects(raw, "unknown key")


def test_rejects_unknown_channel_key(raw):
    _table(raw, "Labjack")["channels"][0]["colour"] = "red"
    _rejects(raw, "unknown key")


def test_rejects_duplicate_db_column(raw):
    chans = _table(raw, "Labjack")["channels"]
    chans.append(copy.deepcopy(chans[0]))
    _rejects(raw, "duplicate column")


def test_rejects_included_feed_without_time_column(raw):
    _table(raw, "Labjack")["time_column"] = None
    _rejects(raw, "needs a time_column")


def test_rejects_included_feed_without_period(raw):
    _table(raw, "Labjack")["nominal_period_s"] = None
    _rejects(raw, "needs a nominal_period_s")


def test_rejects_excluded_table_without_reason(raw):
    del _table(raw, "IVC")["reason"]
    _rejects(raw, "needs a reason")


def test_rejects_missing_schema_column(raw):
    _table(raw, "QT")["channels"].pop()
    _rejects(raw, "ti523_ai is missing")


def test_rejects_missing_table(raw):
    raw["tables"] = [t for t in raw["tables"] if t["name"] != "IVC"]
    _rejects(raw, "IVC is missing")


def test_rejects_corrected_spelling_in_db_column(raw):
    _table(raw, "Flow_Rates")["channels"][0]["db_column"] = "separator_flow"
    _rejects(raw, "seperator_flow is missing")


def test_rejects_wrong_sql_type(raw):
    nmr = _table(raw, "NMR")["channels"]
    next(c for c in nmr if c["db_column"] == "run_number")["sql_type"] = "float"
    _rejects(raw, "should have sql_type")


def test_rejects_bad_group(raw):
    _table(raw, "Labjack")["channels"][0]["group"] = "vacuum"
    _rejects(raw, "group")


def test_rejects_duplicate_table(raw):
    raw["tables"].append(copy.deepcopy(_table(raw, "IVC")))
    _rejects(raw, "duplicate table")


def test_schema_check_can_be_skipped_for_small_layouts(raw):
    raw["tables"] = [_table(raw, "Labjack")]
    reg = parse_registry(raw, schema=None)
    assert len(reg.included_channels()) == 8
