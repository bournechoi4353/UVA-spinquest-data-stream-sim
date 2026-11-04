"""Detectors, state store, summary, live reader, and scoring."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from monitor.detectors import (Calibration, ChannelMonitor, DetectorSettings, FeedWatch, adapt_thresholds,
                               calibrate, siegmund_h)
from monitor.engine import Coupling, Engine
from monitor.ingest import PtdbReader
from monitor.scoring import score_run
from monitor.summary import build_summary
from sim import replay
from sim.db_writer import write_fixture
from sim.io import write_run
from sim.layout_scenario import generate_layout, merge_feed_events
from sim.profiler.adapters import SQLiteAdapter
from sim.registry import load_registry
from sim.stream import generate

from conftest import COUPLED_CHANNELS, make_config

T0 = datetime(2026, 1, 1, 8, 0, 0)
REG = load_registry("configs/ptdb_layout.yaml")


def times(n, period=1.0):
    return [T0 + timedelta(seconds=k * period) for k in range(n)]


def healthy_cal():
    """Calibration the way real use works: from separate healthy data, thresholds adapted."""
    h = noise(20000, seed=99)
    return adapt_thresholds(h, calibrate(h), DetectorSettings())


def run_monitor(values, cal=None, settings=None):
    cal = cal or healthy_cal()
    m = ChannelMonitor("a", cal, settings)
    raised = []
    for t, x in zip(times(len(values)), values):
        raised += [c for c in m.update(t, x) if c.active or c.kind == "spike"]
    return raised


def noise(n, seed=0, sigma=0.5, base=10.0):
    return base + sigma * np.random.default_rng(seed).standard_normal(n)


# ---- single detectors ---------------------------------------------------------------
def test_calibrate_uses_median_and_mad():
    v = noise(5000)
    cal = calibrate(v)
    assert cal.target == pytest.approx(10.0, abs=0.03)
    assert cal.sigma == pytest.approx(0.5, rel=0.05)


def test_isolated_spike_is_reported_once_and_a_step_is_not_a_spike():
    v = noise(600)
    v[300] += 4.0                                   # 8 sigma
    kinds = [c.kind for c in run_monitor(v)]
    assert kinds.count("spike") == 1
    step = noise(600)
    step[300:] += 4.0
    raised = run_monitor(step)
    assert not [c for c in raised if c.kind == "spike"]
    assert [c.kind for c in raised] == ["drift_up"]


def test_cusum_catches_a_drift_and_reports_since():
    v = noise(3000, seed=1)
    v[1000:] += np.linspace(0, 3.0, 2000)
    raised = run_monitor(v)
    drift = [c for c in raised if c.kind == "drift_up"]
    assert drift
    assert drift[0].since <= drift[0].detected_at
    assert (drift[0].since - T0).total_seconds() >= 900


def test_one_spike_cannot_trigger_the_cusum():
    v = noise(2000, seed=2)
    v[1000] += 50.0
    assert [c.kind for c in run_monitor(v)] == ["spike"]


def test_stuck_missing_and_nan_does_not_reset_state():
    v = noise(800, seed=3)
    v[200:300] = v[199]
    v[500:510] = np.nan
    kinds = [c.kind for c in run_monitor(v)]
    assert "stuck" in kinds and "missing" in kinds


def test_noise_growth_is_reported_as_noisy():
    # Known limit: strong noise growth can ALSO trip the drift CUSUM (large random swings add up),
    # so only the noisy condition is required here.
    v = noise(3000, seed=4)
    v[1500:] = 10.0 + 2.0 * np.random.default_rng(5).standard_normal(1500)
    noisy = [c for c in run_monitor(v) if c.kind == "noisy"]
    assert noisy and (noisy[0].since - T0).total_seconds() >= 1500


def test_adaptive_thresholds_quiet_on_autocorrelated_healthy_noise():
    from sim.channels import ar1_noise
    rng = np.random.default_rng(6)
    healthy = 10 + ar1_noise(rng.standard_normal(20000), 0.5, 0.8)
    test = 10 + ar1_noise(rng.standard_normal(20000), 0.5, 0.8)
    plain = run_monitor(test, calibrate(healthy))        # plain Phase 5 thresholds
    cal = adapt_thresholds(healthy, calibrate(healthy), DetectorSettings())
    adapted = run_monitor(test, cal)
    assert len([c for c in plain if c.kind.startswith("drift")]) > 5
    assert not [c for c in adapted if c.kind.startswith("drift")]
    assert cal.cusum_scale > 2.5


def test_siegmund_matches_the_known_phase5_value():
    assert siegmund_h(930) == pytest.approx(5.0, abs=0.05)


def test_feed_watch_outage_and_recovery():
    w = FeedWatch("Labjack", 1.79)
    assert w.row(T0) == []
    assert w.tick(T0 + timedelta(seconds=5)) == []
    (c,) = w.tick(T0 + timedelta(seconds=10))
    assert c.kind == "feed_outage" and c.active
    (end,) = w.row(T0 + timedelta(seconds=300))
    assert not end.active and end.detail["gap_s"] == pytest.approx(300)


# ---- engine and summary ---------------------------------------------------------------
@pytest.fixture(scope="module")
def demo_engine():
    eng = Engine.from_healthy(replay("configs/healthy.yaml"),
                              couplings=[Coupling("nmr_signal", "polarization", 0.15)])
    for t, ev in replay("configs/mixed_demo.yaml"):
        eng.step(t, ev)
    return eng


def test_engine_finds_every_mixed_demo_anomaly(demo_engine):
    got = {(c.subject, c.kind) for c in demo_engine.raised}
    assert ("polarization", "drift_down") in got
    assert ("rf_power", "drift_up") in got
    assert ("nmr_signal", "decoupled") in got
    assert sum(1 for c in demo_engine.raised if c.subject == "beam_current" and c.kind == "spike") == 3
    assert not any(c.subject == "target_temp" for c in demo_engine.raised)


def test_summary_states_facts_from_the_store(demo_engine):
    text = build_summary(demo_engine.store)
    assert text.startswith("Status at 2026-01-01 13:59 UTC. 5 channels monitored, 3 abnormal.")
    assert "polarization: running low since 10:03" in text
    assert "no longer tracking polarization" in text
    assert "Normal: beam_current, target_temp." in text


def test_state_store_round_trips_to_json(demo_engine, tmp_path):
    import json
    demo_engine.store.save(tmp_path / "state.json")
    data = json.loads((tmp_path / "state.json").read_text())
    assert set(data["channels"]) == {"polarization", "beam_current", "rf_power", "target_temp", "nmr_signal"}
    assert data["channels"]["rf_power"]["active"][0]["kind"] == "drift_up"


def test_empty_store_summary():
    from monitor.state import StateStore
    assert build_summary(StateStore()).startswith("Status: no data yet.")


# ---- live reader ------------------------------------------------------------------------
@pytest.fixture(scope="module")
def stressed(tmp_path_factory):
    run = generate_layout({"seed": 4, "start_time": "2026-01-01T00:00:00", "duration_seconds": 3 * 3600}, REG)
    path = tmp_path_factory.mktemp("rd") / "f.sqlite"
    counts = write_fixture(run, path, null_fraction=0.01, dup_fraction=0.01, late_fraction=0.01)
    return run, path, counts


def test_reader_drops_duplicates_reorders_late_rows_and_skips_nulls(stressed):
    run, path, counts = stressed
    r = PtdbReader(SQLiteAdapter(path), REG, batch_rows=700)
    events = list(r.read_all())
    ts = [t for t, _ in events]
    assert ts == sorted(ts)
    assert all(t.tzinfo == timezone.utc for t in ts[:5])
    assert r.stats.duplicates == sum(c.duplicates for c in counts.values())
    assert r.stats.too_late == 0
    assert r.stats.emitted == sum(f.n_rows for f in run.feeds.values())
    assert r.stats.nulls == sum(c.nulls for c in counts.values())
    assert all(v == v for _, e in events for v in e.values())
    clean = list(merge_feed_events(run.feeds))
    assert len(events) == len(clean)


def test_reader_polls_incrementally(stressed):
    _, path, _ = stressed
    r = PtdbReader(SQLiteAdapter(path), REG)
    r.start_at_time(datetime(2026, 1, 1, 2, 0, tzinfo=timezone.utc))
    first = r.poll(flush=True)
    assert first and first[0][0] >= datetime(2026, 1, 1, 1, 59, tzinfo=timezone.utc)
    assert r.poll(flush=True) == []                    # nothing new
    r2 = PtdbReader(SQLiteAdapter(path), REG)
    r2.start_at_end()
    assert r2.poll(flush=True) == []


def _brute_first_id(path, table, when):
    import sqlite3
    con = sqlite3.connect(path)
    try:
        rows = con.execute(f'SELECT id, "Timestamp" FROM "{table}" ORDER BY id').fetchall()
    finally:
        con.close()
    for rid, ts in rows:
        if datetime.fromisoformat(ts).replace(tzinfo=timezone.utc) >= when:
            return rid
    return rows[-1][0] + 1 if rows else 1


def _qt_db(path, rows):
    """A minimal QT table: (id, 'YYYY-MM-DD HH:MM:SS') rows, ids may have gaps."""
    import sqlite3
    con = sqlite3.connect(path)
    con.execute('CREATE TABLE "QT" (id INTEGER PRIMARY KEY, "Timestamp" TEXT NOT NULL)')
    con.executemany('INSERT INTO "QT" (id, "Timestamp") VALUES (?, ?)', rows)
    con.commit()
    con.close()


@pytest.fixture(scope="module")
def clean_with_outage(tmp_path_factory):
    from sim.layout_recipes import mixed_anomalies
    raw = {"seed": 8, "start_time": "2026-01-01T00:00:00", "duration_seconds": 2 * 3600}
    raw["anomalies"] = [a for a in mixed_anomalies(raw, REG) if a.get("type") == "feed_outage"]
    run = generate_layout(raw, REG)
    path = tmp_path_factory.mktemp("bs") / "f.sqlite"
    write_fixture(run, path)
    return path


def test_start_at_time_matches_brute_force_without_a_time_query(clean_with_outage):
    path = clean_with_outage
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    whens = [base - timedelta(hours=1), base, base + timedelta(seconds=1), base + timedelta(minutes=37, seconds=13),
             base + timedelta(minutes=61), base + timedelta(hours=1, minutes=59, seconds=59),
             base + timedelta(hours=5)]
    whens += [base + timedelta(seconds=int(s)) for s in np.random.default_rng(3).integers(0, 7200, 25)]
    for when in whens:
        a = SQLiteAdapter(path)
        r = PtdbReader(a, REG)
        r.start_at_time(when)
        for t in r.tables:
            assert r.stats.last_id[t.name] + 1 == _brute_first_id(path, t.name, when), (t.name, when)
        # every statement is a primary-key probe or MIN/MAX(id): none filters on the time column
        assert all("Timestamp\" >=" not in s and "Timestamp >=" not in s for s in a.statements)
        assert len(a.statements) <= len(r.tables) * 30


def test_start_at_time_handles_id_gaps_and_the_edges(tmp_path):
    path = tmp_path / "gaps.sqlite"
    base = datetime(2026, 8, 14, 19, 0, tzinfo=timezone.utc)
    ids = [5, 6, 7, 40, 41, 900, 901, 902, 10_000, 10_001]
    _qt_db(path, [(i, (base + timedelta(seconds=10 * k)).strftime("%Y-%m-%d %H:%M:%S")) for k, i in enumerate(ids)])
    for k in range(-1, len(ids) + 1):
        when = base + timedelta(seconds=10 * k - 3)
        r = PtdbReader(SQLiteAdapter(path), REG, tables=["QT"])
        r.start_at_time(when)
        expect = ids[max(k, 0)] if k < len(ids) else ids[-1] + 1
        assert r.stats.last_id["QT"] + 1 == expect, (k, when)
    empty = tmp_path / "empty.sqlite"
    _qt_db(empty, [])
    r = PtdbReader(SQLiteAdapter(empty), REG, tables=["QT"])
    r.start_at_time(base)
    assert r.stats.last_id["QT"] == 0


def test_start_at_time_with_late_rows_starts_within_the_reorder_window(stressed):
    _, path, _ = stressed
    when = datetime(2026, 1, 1, 1, 30, tzinfo=timezone.utc)
    r = PtdbReader(SQLiteAdapter(path), REG)
    r.start_at_time(when)
    events = list(r.read_all())
    assert events
    assert events[0][0] >= when - timedelta(seconds=60)
    assert events[0][0] <= when + timedelta(seconds=60)


def test_reader_only_issues_guarded_selects(stressed):
    _, path, _ = stressed
    a = SQLiteAdapter(path)
    list(PtdbReader(a, REG).read_all())
    assert a.statements and all(s.lstrip().upper().startswith("SELECT") for s in a.statements)


def test_engine_on_wide_feeds_detects_dropout_and_outage(tmp_path):
    cfg = {"seed": 9, "start_time": "2026-01-01T00:00:00", "duration_seconds": 4 * 3600,
           "anomalies": [{"feed": "Labjack", "type": "feed_outage", "start": 9000, "duration": 600},
                         {"channel": "Flow_Rates.main_flow", "type": "dropout", "start": 10000, "duration": 120}]}
    run = generate_layout(cfg, REG)
    events = list(merge_feed_events(run.feeds))
    healthy = [e for e in events if (e[0] - events[0][0]).total_seconds() < 7200]
    feeds = {t.name: [c.key for c in t.channels] for t in REG.included_tables()}
    eng = Engine.from_healthy(healthy, feeds=feeds)
    for t, ev in events:
        eng.step(t, ev)
    kinds = {(c.subject, c.kind) for c in eng.raised}
    assert ("Labjack", "feed_outage") in kinds
    assert ("Flow_Rates.main_flow", "missing") in kinds


# ---- scoring -------------------------------------------------------------------------------
def test_score_run_matches_labels_without_reading_them_during_detection(tmp_path):
    cfg = make_config(channels=COUPLED_CHANNELS, duration=7200, anomalies=[
        {"channel": "pol", "type": "step_shift", "start": 3600, "magnitude": 8},
        {"channel": "pol", "type": "spike", "start": 1000, "magnitude": 10},
    ])
    out = write_run(generate(cfg), tmp_path / "run")
    sc = score_run(out)
    assert len(sc.matched) == 2
    assert all(m["delay_s"] is not None and m["delay_s"] < 60 for m in sc.matched.values())


# ---- real-data patterns found by the profiler -------------------------------------------------
def test_constant_channel_is_calibrated_and_not_reported_stuck():
    cal = calibrate(np.zeros(5000))
    assert cal.flat and cal.stuck_normal and cal.sigma > 0
    m = ChannelMonitor("Flow_Rates.heat_exchanger_flow", cal)
    raised = [c for t, x in zip(times(3000), np.zeros(3000)) for c in m.update(t, x)]
    assert not raised


def test_held_values_turn_stuck_detection_off():
    held = np.repeat(noise(500, seed=11), 4)          # every value held for 4 samples, as real Labjack data
    cal = calibrate(held)
    assert cal.stuck_normal and not cal.flat


def test_partial_rows_do_not_look_missing_but_a_real_dropout_does():
    chans = {key: {"null_fraction": 0.53} for key in [c.key for c in REG.channels("Labjack")]}
    base = {"seed": 21, "start_time": "2026-01-01T00:00:00", "channels": chans}
    healthy = generate_layout({**base, "duration_seconds": 3 * 3600}, REG)
    run = generate_layout({**base, "seed": 22, "duration_seconds": 3 * 3600,
                           "anomalies": [{"channel": "Labjack.buffer_pressure", "type": "dropout",
                                          "start": 5000, "duration": 300}]}, REG)
    feeds = {t.name: [c.key for c in t.channels] for t in REG.included_tables()}
    eng = Engine.from_healthy(merge_feed_events(healthy.feeds), feeds=feeds)
    for t, ev in merge_feed_events(run.feeds):
        eng.step(t, ev)
    missing = [c for c in eng.raised if c.kind == "missing"]
    assert [c.subject for c in missing] == ["Labjack.buffer_pressure"]
    # "since" is the last value before the gap: just before the dropout starts at 5000 s.
    since_s = (missing[0].since - datetime(2026, 1, 1, tzinfo=timezone.utc)).total_seconds()
    assert 4950 <= since_s <= 5000
