"""The real ptdb schema, as read from information_schema (section 3.2 of the prompt).

This is the reference the layout file is checked against. It is written out by
hand here, separately from ``configs/ptdb_layout.yaml``, so that a column that
is dropped or misspelled in the layout is caught by the validator instead of
silently disappearing. Spelling is kept exactly, typos included.

Each table maps to ``(time_column, time_type, [(value_column, sql_type), ...])``.
Every table also has an ``id`` primary key, which is not repeated below.
"""

from __future__ import annotations

ID_COLUMN = "id"

_F = "float"


def _floats(*names: str) -> list[tuple[str, str]]:
    return [(n, _F) for n in names]


EXPECTED_SCHEMA: dict[str, tuple[str, str, list[tuple[str, str]]]] = {
    "Flow_Rates": ("Timestamp", "timestamp", _floats(
        "seperator_flow", "magnet_flow", "main_flow", "microwave_flow", "heat_exchanger_flow",
    )),
    "IVC": ("Timestamp", "timestamp", _floats("ivc_pressure")),
    "Labjack": ("Timestamp", "timestamp", _floats(
        "root_exhaust_pressure", "buffer_pressure", "magnet_pressure", "purifier_inlet_pressure",
        "fridge_vapor_pressure", "thermocouple", "magnet_bottom_temperature", "magnet_top_temperature",
    )),
    "Lakeshore_Fridge_Temp": ("Timestamp", "timestamp", _floats(
        "fridge_target_top_up_temperature", "fridge_target_top_up_center_temperature",
        "fridge_target_top_down_temperature", "fridge_target_bottom_up_temperature",
        "fridge_target_bottom_up_center_temperature", "fridge_target_bottom_down_temperature",
        "fridge_target_top_cernox_temperature", "fridge_target_bottom_cernox_temperature",
    )),
    "Lakeshore_Magnet_Temp": ("Timestamp", "timestamp", _floats(
        *(f"magnet_channel_{i}" for i in range(1, 9)),
    )),
    "Lakeshore_Target_Stick": ("Timestamp", "timestamp", _floats(
        "target_stick_buffle_top_temperature", "target_stick_buffle_bottom_temperature",
        "target_stick_seperator_top_temperature", "target_stick_seperator_bottom_temperature",
        "target_stick_heat_exchanger_top_temperature", "target_stick_heat_exchanger_bottom_temperature",
        "target_stick_annealing_plate_bar_temperature", "target_stick_annealing_plate_top_temperature",
    )),
    "MaxiGauge": ("Timestamp", "timestamp", _floats(
        "maxigauge_seperator_inlet_pressure", "maxigauge_upper_roots_pressure",
        *(f"maxigauge_channel_{i}" for i in range(3, 7)),
    )),
    "NMR": ("Timestamp", "timestamp", [
        ("run_number", "int"), ("measurement_type", "text"), ("peak_amp", _F), ("peak_center", _F),
        ("beam_on", "int"), ("rf_level", _F), ("if_atten", _F), ("he_temperature", _F),
        ("he_pressure", _F), ("nmr_channel", "int"), ("temperature", _F),
        ("calibration_constant", _F), ("polarization", _F), ("polarization_std", _F), ("snr", _F),
        ("step_width", _F), ("center_freq", _F), ("freq_span", _F), ("area", _F),
        ("phase_voltage", _F), ("tune_voltage", _F),
    ]),
    "QT": ("Timestamp", "timestamp", _floats(
        "fc501_ai", "fc501_out", "fc502_ai", "fc502_out", "lit501_ai", "pt501_ai", "pt502_ai",
        "pt503_ai", "pt504_ai", "ait501_ai", "ti501_ai", "ti502_ai", "ti503_ai", "ti504_ai",
        "ti505_ai", "ti523_ai",
    )),
    "sensor_data": ("timestamp", "datetime", _floats("value")),
    "stress_test_data": ("Timestamp", "datetime", _floats(*(f"col{i}" for i in range(1, 6)))),
}

# id + time column + value columns, summed over every table.
EXPECTED_TOTAL_COLUMNS = sum(2 + len(cols) for _, _, cols in EXPECTED_SCHEMA.values())
