"""Phase 3: inspection plots."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sim.io import write_run
from sim.plotting import TYPE_COLORS, load_run, plot_run, with_gaps
from sim.scenario import ANOMALY_SPECS
from sim.stream import generate

from conftest import COUPLED_CHANNELS, make_config

ROOT = Path(__file__).resolve().parent.parent
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _is_png(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 1000 and path.read_bytes()[:8] == PNG_MAGIC


@pytest.fixture
def coupled_run(tmp_path):
    cfg = make_config(
        channels=COUPLED_CHANNELS,
        anomalies=[
            {"channel": "pol", "type": "linear_drift", "start": 600, "duration": 1800,
             "magnitude": -2.0, "magnitude_unit": "absolute", "hold_after": True},
            {"channel": "nmr", "type": "decoupling", "start": 1800, "duration": 600},
            {"channel": "pol", "type": "dropout", "start": 300, "duration": 30},
            {"channel": "pol", "type": "spike", "start": 100, "magnitude": 8},
        ],
    )
    return write_run(generate(cfg), tmp_path / "run")


def test_every_anomaly_type_has_a_color():
    assert set(ANOMALY_SPECS) <= set(TYPE_COLORS)


def test_plot_run_writes_overview_and_decoupling_scatter(coupled_run):
    paths = plot_run(coupled_run)
    names = sorted(p.name for p in paths)
    assert names[0].startswith("decoupling_nmr_")
    assert names[1] == "overview.png"
    for p in paths:
        assert p.parent == coupled_run / "plots"
        assert _is_png(p)


def test_no_scatter_option_skips_the_scatter(coupled_run, tmp_path):
    paths = plot_run(coupled_run, out_dir=tmp_path / "elsewhere", scatter=False)
    assert [p.name for p in paths] == ["overview.png"]
    assert paths[0].parent == tmp_path / "elsewhere"


def test_run_without_decoupling_makes_only_the_overview(tmp_path):
    run = write_run(generate(make_config(duration=600)), tmp_path / "plain")
    assert [p.name for p in plot_run(run)] == ["overview.png"]


def test_load_run_keeps_dropouts_as_nan(coupled_run):
    run = load_run(coupled_run)
    assert np.isnan(run.values["pol"]).sum() == 30
    assert len(run.times["nmr"]) == len(run.values["nmr"]) == 720  # 0.2 Hz for 3600 s


def test_with_gaps_breaks_the_line_where_rows_are_absent():
    t = np.array(["2026-01-01T00:00:00", "2026-01-01T00:00:01", "2026-01-01T00:00:02",
                  "2026-01-01T00:00:30", "2026-01-01T00:00:31"], dtype="datetime64[us]")
    v = np.arange(5, dtype=float)
    t2, v2 = with_gaps(t, v)
    assert len(t2) == 6
    assert np.isnan(v2[3]) and t2[2] < t2[3] < t2[4]
    # Regular data is returned unchanged.
    t3, v3 = with_gaps(t[:3], v[:3])
    assert len(t3) == 3


def test_plot_script_accepts_a_yaml(tmp_path):
    cfg = ROOT / "configs" / "mixed_demo.yaml"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "plot_run.py"), str(cfg), "--out", str(tmp_path)],
        capture_output=True, text=True, check=True,
    )
    assert "overview.png" in result.stdout
    assert _is_png(tmp_path / "mixed_demo" / "plots" / "overview.png")
