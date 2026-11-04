"""Phase R2: wide feeds, feed outages, replay, and the SQLite fixture."""

import sqlite3
from datetime import datetime, timezone

import numpy as np
import pytest

from sim.db_writer import read_fixture, write_fixture
from sim.layout_io import read_feeds, read_layout_labels, write_layout_run
from sim.layout_recipes import mixed_anomalies
from sim.layout_scenario import datetime_to_us, generate_layout, replay_layout
from sim.plotting import plot_layout_run
from sim.registry import load_registry
from sim.scenario import ScenarioError

REG = load_registry("configs/ptdb_layout.yaml")
START = "2026-01-01T00:00:00"


def cfg(anomalies=None, hours=1.0, seed=3, **extra):
    return {"seed": seed, "start_time": START, "duration_seconds": hours * 3600,
            "anomalies": anomalies or [], **extra}


@pytest.fixture(scope="module")
def mixed_run():
    c = cfg(hours=2, seed=11)
    c["anomalies"] = mixed_anomalies(c, REG)
    return generate_layout(c, REG)


def test_feeds_match_the_registry(mixed_run):
    assert list(mixed_run.feeds) == [t.name for t in REG.included_tables()]
    for name, feed in mixed_run.feeds.items():
        assert feed.columns == [c.db_column for c in REG.channels(name)]
        assert feed.time_column == "Timestamp"
        assert list(feed.ids) == list(range(1, feed.n_rows + 1))
    assert mixed_run.meta["synthetic"] is True


def test_defaults_are_plainly_synthetic():
    run = generate_layout(cfg(hours=0.5), REG)
    v = run.feeds["Labjack"].values["buffer_pressure"]
    assert abs(np.nanmean(v) - 100) < 0.2
    assert abs(np.nanstd(v) - 1.0) < 0.1


def test_per_channel_override():
    run = generate_layout(cfg(hours=0.5, channels={"Labjack.thermocouple": {"baseline": 4.2, "sigma": 0.01}}), REG)
    v = run.feeds["Labjack"].values["thermocouple"]
    assert abs(np.nanmean(v) - 4.2) < 0.01


def test_timestamps_are_whole_seconds_monotonic_and_can_repeat():
    run = generate_layout(cfg(hours=1), REG)
    for feed in run.feeds.values():
        assert np.all(feed.t_us % 1_000_000 == 0)
        assert np.all(np.diff(feed.t_us) >= 0)
    lab = run.feeds["Labjack"]   # a period between 1 and 2 s floored to 1 s: steps of 1 s and 2 s
    steps = set(np.diff(lab.t_us) // 1_000_000)
    assert {1, 2} <= steps


def test_jitter_stays_within_bounds():
    c = cfg(hours=1, feeds={f: {"gap_fraction": 0.0, "timestamp_resolution_s": None}
                            for f in [t.name for t in REG.included_tables()]})
    run = generate_layout(c, REG)
    start = datetime_to_us(datetime.fromisoformat(START))
    for feed in run.feeds.values():
        period = int(round(feed.settings.period_s * 1e6))
        k = np.arange(feed.n_rows)
        nominal = start + np.round(k * 1e6 / (1 / feed.settings.period_s)).astype(np.int64)
        jitter = feed.t_us - nominal
        assert jitter.min() >= 0
        assert jitter.max() <= feed.settings.jitter_s * 1e6 + 1
        assert feed.settings.jitter_s == pytest.approx(0.05 * feed.settings.period_s)
        assert period > 0


def test_rejects_jitter_of_half_a_period_or_more():
    with pytest.raises(ScenarioError, match="half the period"):
        generate_layout(cfg(feeds={"Labjack": {"jitter_s": 1.0}}), REG)


def test_feed_outage_removes_rows_and_labels_detectability():
    run = generate_layout(cfg(anomalies=[{"feed": "Labjack", "type": "feed_outage",
                                          "start": 1200, "duration": 300}]), REG)
    lab = run.feeds["Labjack"]
    start = datetime_to_us(datetime.fromisoformat(START))
    lo, hi = start + 1200 * 10**6, start + 1500 * 10**6
    assert not np.any((lab.t_us >= lo + 10**6) & (lab.t_us < hi))
    # The other feeds are untouched.
    other = run.feeds["Flow_Rates"].t_us
    assert np.any((other >= lo) & (other < hi))
    (label,) = run.labels
    assert label["type"] == "feed_outage" and label["channel"] == "Labjack"
    assert label["severity"] == "high"
    last = lab.t_us[lab.t_us < lo][-1]
    fd = datetime_to_us(datetime.fromisoformat(label["first_detectable"]))
    assert fd == last + int(3 * REG.table("Labjack").nominal_period_s * 1e6)


def test_existing_anomaly_types_work_on_wide_feeds():
    anoms = [
        {"channel": "Labjack.buffer_pressure", "type": "step_shift", "start": 1000, "magnitude": 6},
        {"channel": "Flow_Rates.main_flow", "type": "linear_drift", "start": 600, "duration": 1200,
         "magnitude": 8},
        {"channel": "QT.ti501_ai", "type": "dropout",
         "start": 300, "duration": 120},
    ]
    run = generate_layout(cfg(anomalies=anoms), REG)
    assert sorted(lb["type"] for lb in run.labels) == ["dropout", "linear_drift", "step_shift"]
    v = run.feeds["Labjack"].values["buffer_pressure"]
    t = run.feeds["Labjack"].t_us
    step_t = datetime_to_us(datetime.fromisoformat(START)) + 1001 * 10**6
    assert np.nanmean(v[t > step_t]) - np.nanmean(v[t < step_t]) > 5
    fr = run.feeds["QT"].values["ti501_ai"]
    assert np.isnan(fr).sum() > 0


def test_labels_use_realized_row_times(mixed_run):
    times = {name: set(f.t_us.tolist()) for name, f in mixed_run.feeds.items()}
    for lb in mixed_run.labels:
        if lb["type"] == "feed_outage":
            continue
        table = lb["channel"].split(".")[0]
        assert datetime_to_us(datetime.fromisoformat(lb["start"])) in times[table]
        if lb["first_detectable"]:
            assert datetime_to_us(datetime.fromisoformat(lb["first_detectable"])) in times[table]


def test_same_seed_same_run():
    a, b = generate_layout(cfg(hours=0.5), REG), generate_layout(cfg(hours=0.5), REG)
    for name in a.feeds:
        assert np.array_equal(a.feeds[name].t_us, b.feeds[name].t_us)
        for col in a.feeds[name].columns:
            assert np.array_equal(a.feeds[name].values[col], b.feeds[name].values[col], equal_nan=True)


def test_cross_feed_coupling():
    c = cfg(hours=1, couplings=[{"channel": "Labjack.magnet_pressure", "source": "Flow_Rates.magnet_flow",
                                 "gain": 0.5}],
            anomalies=[{"channel": "Flow_Rates.magnet_flow", "type": "step_shift", "start": 1800,
                        "magnitude": 20, "magnitude_unit": "absolute"}],
            channels={"Labjack.magnet_pressure": {"sigma": 1.0}, "Flow_Rates.magnet_flow": {"sigma": 1.0}})
    run = generate_layout(c, REG)
    t = run.feeds["Labjack"].t_us
    v = run.feeds["Labjack"].values["magnet_pressure"]
    mid = datetime_to_us(datetime.fromisoformat(START)) + 1800 * 10**6
    assert np.mean(v[t > mid + 10**7]) - np.mean(v[t < mid]) == pytest.approx(10, abs=0.5)


def test_replay_yields_one_event_per_row_in_order(mixed_run):
    events = list(replay_layout(mixed_run))
    assert len(events) == sum(f.n_rows for f in mixed_run.feeds.values())
    times = [t for t, _ in events]
    assert times == sorted(times)
    assert all(t.tzinfo == timezone.utc for t in times[:10])
    first_keys = set(events[0][1])
    assert all(k.count(".") == 1 for k in first_keys)


def test_replay_leaves_nulls_out():
    run = generate_layout(cfg(anomalies=[{"channel": "Labjack.thermocouple", "type": "dropout",
                                          "start": 100, "duration": 60}]), REG)
    lab_events = [e for _, e in replay_layout(run) if "Labjack.buffer_pressure" in e]
    missing = [e for e in lab_events if "Labjack.thermocouple" not in e]
    expected = 60 / REG.table("Labjack").nominal_period_s
    assert 0.8 * expected <= len(missing) <= 1.2 * expected
    assert all(v == v for _, e in replay_layout(run) for v in e.values())


def test_run_folder_round_trip_and_replay_from_folder(mixed_run, tmp_path):
    out = write_layout_run(mixed_run, tmp_path / "run")
    feeds = read_feeds(out)
    for name, f in mixed_run.feeds.items():
        assert np.array_equal(feeds[name].t_us, f.t_us)
        for col in f.columns:
            assert np.array_equal(feeds[name].values[col], f.values[col], equal_nan=True)
    assert read_layout_labels(out) == mixed_run.labels
    assert len(list(replay_layout(out))) == sum(f.n_rows for f in mixed_run.feeds.values())


def test_sqlite_round_trip_is_exact(mixed_run, tmp_path):
    path = tmp_path / "fixture.sqlite"
    write_fixture(mixed_run, path)
    con = sqlite3.connect(path)
    names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    idx = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    con.close()
    assert names == set(mixed_run.feeds)
    assert "Labjack_Timestamp" in idx
    for name, f in mixed_run.feeds.items():
        df = read_fixture(path, name, chunk_rows=777)
        assert list(df.columns) == ["id", "t_us", *f.columns]
        assert np.array_equal(df["id"].to_numpy(), f.ids)
        assert np.array_equal(df["t_us"].to_numpy(), f.t_us)
        for col in f.columns:
            assert np.array_equal(df[col].to_numpy(), f.values[col], equal_nan=True)


def test_stress_options_are_deterministic_and_do_what_they_say(tmp_path):
    run = generate_layout(cfg(hours=1), REG)
    a = write_fixture(run, tmp_path / "a.sqlite", 0.01, 0.01, 0.01)
    b = write_fixture(run, tmp_path / "b.sqlite", 0.01, 0.01, 0.01)
    assert a == b
    assert (tmp_path / "a.sqlite").read_bytes() == (tmp_path / "b.sqlite").read_bytes()
    lab = read_fixture(tmp_path / "a.sqlite", "Labjack")
    n = run.feeds["Labjack"].n_rows
    assert len(lab) == n + a["Labjack"].duplicates
    assert a["Labjack"].duplicates > 0 and a["Labjack"].late > 0 and a["Labjack"].nulls > 0
    assert (np.diff(lab["t_us"].to_numpy()) < 0).any()          # late rows: time goes backward by id
    assert lab.duplicated(subset=["t_us", *run.feeds["Labjack"].columns]).any()
    assert lab["buffer_pressure"].isna().any()
    c = write_fixture(run, tmp_path / "c.sqlite", 0.01, 0.01, 0.01, seed=99)
    assert c != a


def test_layout_plots_say_simulated(mixed_run, tmp_path):
    out = write_layout_run(mixed_run, tmp_path / "run")
    paths = plot_layout_run(out)
    assert sorted(p.name for p in paths) == sorted(f"feed_{t.name}.png" for t in REG.included_tables())
    import inspect
    from sim import plotting
    assert "SIMULATED" in inspect.getsource(plotting.plot_layout_run)


def test_rejects_unknown_channel_and_bad_outage_feed():
    with pytest.raises(ScenarioError):
        generate_layout(cfg(channels={"Labjack.nope": {"sigma": 1}}), REG)
    with pytest.raises(ScenarioError):
        generate_layout(cfg(anomalies=[{"feed": "IVC", "type": "feed_outage", "start": 1, "duration": 5}]), REG)
    with pytest.raises(ScenarioError):
        generate_layout(cfg(synthetic=False), REG)
