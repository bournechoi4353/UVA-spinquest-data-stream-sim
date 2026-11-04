"""Phase R4: fitting the simulator to a profile, using only simulated fixtures."""

import json
from pathlib import Path

import pytest

from sim.db_writer import write_fixture
from sim.from_profile import fit_config, fit_report, invert_noise, load_profile
from sim.layout_scenario import generate_layout
from sim.profiler import Profiler, ProfileSettings, SQLiteAdapter, write_report
from sim.registry import load_registry

ROOT = Path(__file__).resolve().parent.parent
REG = load_registry(ROOT / "configs" / "ptdb_layout.yaml")

TRUE = {
    "Labjack.buffer_pressure": {"baseline": 812.5, "sigma": 2.5, "ar1_phi": 0.6},
    "Labjack.thermocouple": {"baseline": 22.0, "sigma": 0.3, "ar1_phi": 0.0},
    "Flow_Rates.main_flow": {"baseline": 3.2, "sigma": 0.05, "ar1_phi": 0.3},
    "QT.ti501_ai": {"baseline": 0.35, "sigma": 0.002,
                                                               "ar1_phi": 0.8},
}
PERIODS = {t.name: t.nominal_period_s for t in REG.included_tables()}


def _profile(run, tmp_path, name):
    path = tmp_path / f"{name}.sqlite"
    write_fixture(run, path)
    a = SQLiteAdapter(path)
    try:
        rep = Profiler(a, REG, ProfileSettings(chunks=40, chunk_rows=1000, timeline_probes=500,
                                               window_hours=6)).run()
    finally:
        a.close()
    jpath, _ = write_report(rep, tmp_path / name)
    return jpath


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fit")
    cfg = {"seed": 8, "start_time": "2026-03-01T00:00:00", "duration_seconds": 24 * 3600, "channels": TRUE,
           "feeds": {"Labjack": {"gap_fraction": 0.002}}}
    run = generate_layout(cfg, REG)
    profile = load_profile(_profile(run, tmp, "truth"))
    return profile, fit_config(profile, REG), tmp


def test_invert_noise_is_identity_for_white_noise():
    sigma, phi = invert_noise(1.0, 0.0)
    assert sigma == pytest.approx(1.0, abs=0.05)
    assert phi < 0.05


@pytest.mark.parametrize("key", sorted(TRUE))
def test_fit_recovers_channel_parameters_within_10_percent(fitted, key):
    _, fit, _ = fitted
    got, true = fit["channels"][key], TRUE[key]
    assert got["baseline"] == pytest.approx(true["baseline"], rel=0.01)
    assert got["sigma"] == pytest.approx(true["sigma"], rel=0.10)
    assert got["ar1_phi"] == pytest.approx(true["ar1_phi"], abs=0.10)


def test_fit_recovers_feed_timing(fitted):
    _, fit, _ = fitted
    for feed, period in PERIODS.items():
        assert fit["feeds"][feed]["nominal_period_s"] == pytest.approx(period, rel=0.02)
    # Rough by design: whole-second timestamps hide part of the single missing rows.
    assert 0.0002 <= fit["feeds"]["Labjack"]["gap_fraction"] <= 0.004


def test_default_channels_stay_white_noise_at_the_default_level(fitted):
    _, fit, _ = fitted
    m = fit["channels"]["Labjack.magnet_pressure"]
    assert m["baseline"] == pytest.approx(100, abs=0.2)
    assert m["sigma"] == pytest.approx(1.0, rel=0.1)
    assert "slow_wander" not in m or m["slow_wander"]["amplitude"] < 0.3


def test_fitted_config_runs_and_reprofiles_close_to_the_original(fitted, tmp_path):
    profile, fit, _ = fitted
    run = generate_layout({"seed": 99, "start_time": "2026-04-01T00:00:00", "duration_seconds": 24 * 3600,
                           **fit}, REG)
    again = load_profile(_profile(run, tmp_path, "again"))
    for key in TRUE:
        table, col = key.split(".")
        a = profile["tables"][table]["channels"][col]
        b = again["tables"][table]["channels"][col]
        assert b["noise_mad_sigma"] == pytest.approx(a["noise_mad_sigma"], rel=0.10)
        assert b["noise_lag1"] == pytest.approx(a["noise_lag1"], abs=0.08)


def test_couplings_only_when_listed(fitted):
    profile, fit, _ = fitted
    assert "couplings" not in fit
    listed = [{"channel": "Labjack.magnet_pressure", "source": "Flow_Rates.magnet_flow", "gain": 0.1}]
    assert fit_config(profile, REG, listed)["couplings"] == listed


def test_fit_report_lists_outages_and_unfitted(fitted):
    profile, _, _ = fitted
    rep = fit_report(profile, REG)
    assert rep["fitted_channels"] == len(REG.included_channels()) and rep["not_fitted"] == []
    assert set(rep["outages"]) == set(PERIODS)


def test_partial_profile_is_accepted(fitted, tmp_path):
    profile, _, _ = fitted
    partial = {"banner": profile["banner"], "tables": {"Labjack": profile["tables"]["Labjack"]}}
    path = tmp_path / "partial.json"
    path.write_text(json.dumps(partial))
    fit = fit_config(load_profile(path), REG)
    assert set(fit["feeds"]) == {"Labjack"}
    run = generate_layout({"seed": 1, "start_time": "2026-01-01T00:00:00", "duration_seconds": 600, **fit}, REG)
    assert run.feeds["Flow_Rates"].settings.period_s == pytest.approx(PERIODS["Flow_Rates"])
