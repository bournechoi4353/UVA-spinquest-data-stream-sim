"""The simulated live ptdb: incremental writing, a reader following it, pacing, and scoring."""

import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from monitor.engine import Engine
from monitor.ingest import PtdbReader
from sim.layout_scenario import datetime_to_us, generate_layout
from sim.live import LiveWriter, run_live, score_alarms
from sim.profiler.adapters import SQLiteAdapter
from sim.realistic import planted_anomalies, realistic_config
from sim.registry import load_registry

REG = load_registry("configs/ptdb_layout.yaml")
ROOT = Path(__file__).resolve().parent.parent
LIVE_START = datetime(2026, 5, 2, tzinfo=timezone.utc)


def _run(history_h, live_h, plan=True, seed=3):
    start = (LIVE_START - timedelta(hours=history_h)).replace(tzinfo=None).isoformat()
    cfg = realistic_config(REG, None, history_h + live_h, seed, start=start)
    if plan:
        cfg["anomalies"] = planted_anomalies(REG, live_h, offset_s=history_h * 3600)
    return generate_layout(cfg, REG)


def _count(path):
    con = sqlite3.connect(path)
    try:
        return sum(con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in ("Flow_Rates", "Labjack", "QT"))
    finally:
        con.close()


def test_history_first_then_rows_as_the_clock_passes(tmp_path):
    run = _run(0.5, 0.5, plan=False)
    w = LiveWriter(run, tmp_path / "live.sqlite", seed=1)
    live_us = datetime_to_us(LIVE_START)
    n_hist = w.write_until(live_us)
    assert 0 < n_hist < len(w.pending) and _count(w.path) == n_hist
    assert w.write_until(live_us) == 0                                  # nothing new yet
    n_more = w.write_until(live_us + 600 * 10**6)                     # ten simulated minutes later
    assert n_more > 0 and _count(w.path) == n_hist + n_more
    w.write_until(live_us + 10 * 3600 * 10**6)
    assert w.done and _count(w.path) == len(w.pending)
    with pytest.raises(FileExistsError):
        LiveWriter(run, tmp_path / "live.sqlite")                       # never overwrites


def test_reader_follows_the_growing_file_and_late_rows_are_not_lost(tmp_path):
    run = _run(0.5, 0.5, plan=False)
    w = LiveWriter(run, tmp_path / "live.sqlite", seed=2, dup_fraction=0.01, late_fraction=0.01)
    live_us = datetime_to_us(LIVE_START)
    w.write_until(live_us)
    reader = PtdbReader(SQLiteAdapter(w.path), REG)
    first = list(reader.read_all())
    got = len(first)
    for k in range(1, 7):                                               # poll every 5 simulated minutes
        w.write_until(live_us + k * 300 * 10**6)
        got += len(reader.poll())
    w.write_until(live_us + 10 * 3600 * 10**6)
    got += len(reader.poll(flush=True))
    s = reader.stats
    assert s.too_late == 0
    assert got == s.emitted == s.rows - s.duplicates
    assert s.duplicates > 0


def test_pacing_with_a_fake_clock_announces_each_planted_anomaly(tmp_path):
    run = _run(0.5, 1.0)
    w = LiveWriter(run, tmp_path / "live.sqlite", seed=1)
    wall = [0.0]
    msgs = []
    run_live(w, datetime_to_us(LIVE_START), speed=600, tick_s=0.5, announce=msgs.append, labels=run.labels,
             sleep=lambda s: wall.__setitem__(0, wall[0] + s), clock=lambda: wall[0])
    assert w.done
    planted = [m for m in msgs if "planted now" in m]
    assert len(planted) == len([lb for lb in run.labels if lb["start"] >= LIVE_START.isoformat()]) == 7
    assert 5.0 <= wall[0] <= 7.0                                       # 1 h at 600x is 6 s of wall time


def test_a_live_run_is_scored_against_its_ground_truth(tmp_path):
    run = _run(4.0, 4.0)
    w = LiveWriter(run, tmp_path / "live.sqlite", seed=4)
    live_us = datetime_to_us(LIVE_START)
    w.write_until(live_us)
    reader = PtdbReader(SQLiteAdapter(w.path), REG)
    healthy = list(reader.read_all())
    feeds = {t.name: [c.key for c in t.channels] for t in REG.included_tables()}
    eng = Engine.from_healthy(healthy, feeds=feeds)
    for t, ev in healthy:
        eng.step(t, ev)
    for k in range(1, 50):                                              # follow, 5 simulated minutes per poll
        w.write_until(live_us + k * 300 * 10**6)
        for t, ev in reader.poll(flush=w.done):
            eng.step(t, ev)
    alarms = [c.to_dict() for c in eng.raised]
    r = score_alarms(alarms, run.labels, LIVE_START, eng.store.now)
    assert r["planted"] == 7
    assert not r["missed"], r["missed"]
    assert len(r["false_alarms"]) <= 2, r["false_alarms"]


def test_cli_score_mode_reads_the_saved_ground_truth(tmp_path):
    sim_dir, mon = tmp_path / "sim", tmp_path / "mon"
    sim_dir.mkdir()
    mon.mkdir()
    t0 = LIVE_START
    labels = [{"id": "ev_0001", "channel": "QT.fc501_ai", "type": "spike", "start": t0.isoformat(),
               "end": t0.isoformat(), "first_detectable": t0.isoformat()}]
    (sim_dir / "labels.json").write_text(json.dumps(labels))
    (sim_dir / "live_info.json").write_text(json.dumps({"live_start": "2026-05-02 00:00:00"}))
    alarm = {"subject": "QT.fc501_ai", "kind": "spike", "since": t0.isoformat(),
             "detected_at": (t0 + timedelta(seconds=2)).isoformat()}
    (mon / "alarms.jsonl").write_text(json.dumps(alarm) + "\n")
    (mon / "state.json").write_text(json.dumps({"now": (t0 + timedelta(hours=1)).isoformat()}))
    res = subprocess.run([sys.executable, str(ROOT / "scripts/live_sim.py"), "--score", str(sim_dir),
                          "--monitor-dir", str(mon)], capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "Found: 1" in res.stdout and "Missed: 0" in res.stdout and "False alarms: 0" in res.stdout
