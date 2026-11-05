"""Detectors, state store, summary, live reader, and scoring."""

import math
import re
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


def _feed_events(period, n, pauses=(), start=datetime(2026, 1, 1, tzinfo=timezone.utc), seed=0):
    """One-channel QT-like feed: a row every ``period`` s, plus (index, extra seconds) pauses."""
    rng = np.random.default_rng(seed)
    extra = dict(pauses)
    t, out = start, []
    for i in range(n):
        t = t + timedelta(seconds=period + extra.get(i, 0.0))
        out.append((t, {"QT.pt501_ai": 5.0 + 0.01 * rng.standard_normal()}))
    return out


def test_feed_outage_limit_is_learned_from_healthy_pauses():
    feeds = {"QT": ["QT.pt501_ai"]}
    healthy = _feed_events(2.0, 3000, pauses=[(500, 28.0), (1700, 18.0)])      # largest healthy gap 30 s
    eng = Engine.from_healthy(healthy, feeds=feeds)
    assert eng.feed_watch["QT"].limit() == pytest.approx(45.0)                # 1.5 x 30 s
    # a 30 s healthy pause later does not alarm; a 5 minute stop does
    test = _feed_events(2.0, 600, pauses=[(100, 28.0), (400, 300.0)], start=healthy[-1][0], seed=1)
    for t, ev in test:
        eng.step(t, ev)
    outages = [c for c in eng.raised if c.kind == "feed_outage"]
    assert len(outages) == 1 and not outages[0].active


def test_feed_outage_limit_is_clamped():
    feeds = {"QT": ["QT.pt501_ai"]}
    steady = Engine.from_healthy(_feed_events(2.0, 3000), feeds=feeds)
    assert steady.feed_watch["QT"].limit() == pytest.approx(10.0, rel=0.05)   # never below 5 periods
    with_outage = Engine.from_healthy(_feed_events(2.0, 3000, pauses=[(1000, 7200.0)]), feeds=feeds)
    assert with_outage.feed_watch["QT"].limit() == pytest.approx(300.0)       # a real outage cannot raise it to hours


def test_coverage_names_channels_without_calibration_data():
    feeds = {"QT": ["QT.pt501_ai", "QT.pt502_ai"]}
    healthy = _feed_events(2.0, 500)
    healthy[3][1]["QT.pt502_ai"] = 1.0                                          # only 1 value
    eng = Engine.from_healthy(healthy, feeds=feeds)
    assert "QT.pt502_ai" not in eng.monitors
    cov = dict(eng.coverage())
    assert list(cov) == ["QT.pt502_ai"] and "1 values" in cov["QT.pt502_ai"]


def test_run_monitor_writes_alarm_log_coverage_and_digest(stressed, tmp_path):
    import json
    import subprocess
    from pathlib import Path
    import sys
    _, path, _ = stressed
    root = Path(__file__).resolve().parent.parent
    cmd = [sys.executable, str(root / "scripts/run_monitor.py"), "--sqlite", str(path), "--calibrate-minutes", "60",
           "--out", str(tmp_path), "--name", "t"]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=root)
    assert res.returncode == 0, res.stderr
    folder = tmp_path / "t"
    rows = [json.loads(x) for x in (folder / "alarms.jsonl").read_text().splitlines()]
    assert all({"subject", "kind", "since", "detected_at"} <= set(r) for r in rows)
    cov = (folder / "coverage.txt").read_text()
    assert "Monitored channels: 29" in cov and "Not monitored: 0" in cov and "Feed QT:" in cov
    res = subprocess.run([sys.executable, str(root / "scripts/summarize_alarms.py"), str(folder),
                          "--test-start", "2026-01-01T01:00:00", "--out", str(folder / "digest.txt")],
                         capture_output=True, text=True, cwd=root)
    assert res.returncode == 0, res.stderr
    text = (folder / "digest.txt").read_text()
    assert "== Coverage" in text and "== Window: test from 2026-01-01T01:00:00+00:00" in text


# ---- real-data patterns seen on ptdb, 2025-11-04 to 11-10 -------------------------------------
def _held(n, rng, levels=(0.0,), p_update=0.02, step=0.0):
    """A held-value channel: mostly repeats, occasionally moves to another level or by +-step."""
    v, x = [], levels[0]
    for _ in range(n):
        if rng.random() < p_update:
            x = rng.choice(levels) if len(levels) > 1 else x + step * rng.choice([-1, 1])
        v.append(x)
    return np.array(v)


def test_two_code_toggling_level_sensor_is_not_a_spike_storm():
    # LIT501: toggles between two readout codes 0.088 apart, mostly on the lower one
    rng = np.random.default_rng(0)
    lo, hi = -0.0477, -0.0477 + 0.0877
    vals = np.where(rng.random(40_000) < 0.03, hi, lo)
    cal = calibrate(vals[:20_000], np.arange(20_000) * 2.0)
    assert cal.sigma >= 0.0877 * 0.9 / 1.0 / 4.653          # at least the central-98% spread
    adapt_thresholds(vals[:20_000], cal, DetectorSettings())
    m = ChannelMonitor("lit", cal)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    raised = [c for i, x in enumerate(vals[20_000:]) for c in m.update(t0 + timedelta(seconds=2 * i), float(x))]
    assert sum(c.kind == "spike" for c in raised) == 0


def test_recurring_healthy_spike_pattern_is_learned_but_bigger_spikes_still_report():
    # LIT501 (2025-11): noisy baseline, and about 0.2% of readings jump by exactly +0.0898 for one reading
    rng = np.random.default_rng(5)

    def stream(n, extra=()):
        v = -0.05 + rng.normal(0, 0.003, n)
        hits = rng.random(n) < 0.002
        v[hits] += 0.0898
        for i, size in extra:
            v[i] += size
        return v

    healthy = stream(100_000)
    cal = calibrate(healthy, np.arange(100_000) * 2.0)
    adapt_thresholds(healthy, cal, DetectorSettings())
    assert cal.healthy_spikes >= 50 and cal.spike_z > 29
    m = ChannelMonitor("lit", cal)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    test = stream(50_000, extra=[(20_000, 0.30)])            # one jump 3x the familiar size
    spikes = [c for i, x in enumerate(test) for c in m.update(t0 + timedelta(seconds=2 * i), float(x))
              if c.kind == "spike"]
    assert len(spikes) == 1 and spikes[0].magnitude == pytest.approx(0.30, abs=0.02)


def test_a_few_healthy_spikes_do_not_raise_the_threshold():
    rng = np.random.default_rng(6)
    healthy = rng.normal(0, 1, 50_000)
    healthy[[1000, 30_000]] += 9.0                            # two isolated spikes: maybe real, keep 5 sigma
    cal = calibrate(healthy, np.arange(50_000.0))
    adapt_thresholds(healthy, cal, DetectorSettings())
    assert cal.healthy_spikes == 2 and cal.spike_z is None


def test_channel_constant_in_calibration_does_not_give_thousand_sigma_alarms():
    # FC502 flow: constant 0.112124 for the whole calibration day, later moves by 0.0036
    cal = calibrate(np.full(5000, 0.112124), np.arange(5000) * 2.0)
    assert cal.flat and cal.sigma == pytest.approx(1e-3)
    adapt_thresholds(np.full(5000, 0.112124), cal, DetectorSettings())   # as Engine.from_healthy does
    m = ChannelMonitor("fc502", cal)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    raised = []
    for i in range(3000):
        x = 0.112124 - (0.00362 if 1000 <= i < 1003 else 0.0)
        raised += m.update(t0 + timedelta(seconds=2 * i), x)
    assert [c.kind for c in raised] == []                   # a 3.6 sigma blip is not a spike


def test_drift_needs_one_sigma_and_clears_after_settling():
    s = DetectorSettings()
    rng = np.random.default_rng(1)
    cal = calibrate(rng.normal(10, 1, 5000), np.arange(5000.0))
    cal.cusum_h = 8.0
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    # a persistent 0.6 sigma offset is above the CUSUM slack but must not be called abnormal
    m = ChannelMonitor("ti505", cal, s)
    raised = [c for i in range(20_000) for c in m.update(t0 + timedelta(seconds=i), 10.6 + rng.normal(0, 1))]
    assert not any(c.kind.startswith("drift") for c in raised)
    # a 3 sigma step is a drift; after the channel returns to 0.6 sigma it ends within about 30 min
    m = ChannelMonitor("pt", cal, s)
    seq = [13.0] * 3000 + [10.6] * 6000
    raised = [c for i, x in enumerate(seq) for c in m.update(t0 + timedelta(seconds=i), x + rng.normal(0, 1))]
    ups = [c for c in raised if c.kind == "drift_up"]
    assert ups and not ups[0].active
    assert (ups[0].ended_at - (t0 + timedelta(seconds=3000))).total_seconds() < 1800 + 300
    assert len({id(c) for c in ups}) == 1                   # it does not come back while at 0.6 sigma


def test_noise_alarm_does_not_flap_at_its_threshold():
    rng = np.random.default_rng(2)
    healthy = rng.normal(0, 1, 5000)
    cal = calibrate(healthy, np.arange(5000.0))
    adapt_thresholds(healthy, cal, DetectorSettings())
    m = ChannelMonitor("fc501", cal)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ratio = math.sqrt(m.noisy_level / cal.diff_var)
    x = rng.normal(0, ratio * 1.02, 40_000)                 # noise hovering just above the threshold
    raised = [c for i, v in enumerate(x) for c in m.update(t0 + timedelta(seconds=i), float(v))]
    assert len({id(c) for c in raised if c.kind == "noisy"}) <= 3


def test_missing_limit_covers_the_largest_healthy_gap():
    # Labjack at local midnight: a 110 s gap once a day
    t = np.cumsum(np.full(3000, 2.0))
    t[1500:] += 110.0
    cal = calibrate(np.random.default_rng(3).normal(0, 1, 3000), t)
    assert cal.missing_after_s >= 1.5 * 112.0 - 1e-6
    t[1500:] += 7200.0                                       # a 2 h writer absence cannot raise it past 7.5 min
    cal = calibrate(np.random.default_rng(3).normal(0, 1, 3000), t)
    assert cal.missing_after_s <= 450.0 + 1e-6


def test_digest_counts_episodes_and_channels_that_alarm_together(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path
    folder = tmp_path / "run"
    folder.mkdir()
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for k in range(3):                                      # three re-raises 10 min apart: one episode
        s, e = t0 + timedelta(minutes=20 * k), t0 + timedelta(minutes=20 * k + 10)
        rows.append({"subject": "QT.fc501_ai", "kind": "noisy", "since": s.isoformat(), "detected_at": s.isoformat(),
                     "active": False, "ended_at": e.isoformat(), "magnitude": None, "magnitude_sigma": 6.4,
                     "peak_sigma": None, "detail": {}})
    for k in range(2):                                      # one writer: three flows missing together, twice
        s = t0 + timedelta(hours=2 + k)
        for ch in ("Flow_Rates.magnet_flow", "Flow_Rates.main_flow", "Flow_Rates.seperator_flow"):
            rows.append({"subject": ch, "kind": "missing", "since": s.isoformat(), "detected_at": s.isoformat(),
                         "active": False, "ended_at": (s + timedelta(minutes=5)).isoformat(), "magnitude": None,
                         "magnitude_sigma": None, "peak_sigma": None, "detail": {}})
    (folder / "alarms.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (folder / "state.json").write_text(json.dumps({"now": (t0 + timedelta(hours=6)).isoformat(), "channels": {}}))
    root = Path(__file__).resolve().parent.parent
    res = subprocess.run([sys.executable, str(root / "scripts/summarize_alarms.py"), str(folder)],
                         capture_output=True, text=True, cwd=root)
    assert res.returncode == 0, res.stderr
    assert re.search(r"noisy\s+3 raised \(\s*[\d.]+/h\)\s+1 episodes", res.stdout), res.stdout
    assert "missing: Flow_Rates.magnet_flow + Flow_Rates.main_flow + Flow_Rates.seperator_flow: together 2 times" \
        in res.stdout


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
