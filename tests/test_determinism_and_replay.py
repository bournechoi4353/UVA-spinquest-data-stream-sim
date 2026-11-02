"""Reproducibility, file output, and the replay interface."""

import math
from datetime import datetime

import numpy as np
import pytest

from conftest import COUPLED_CHANNELS, make_config
from sim import replay
from sim.io import read_labels, write_run
from sim.scenario import load_scenario
from sim.stream import generate


def test_same_seed_gives_byte_identical_files(tmp_path):
    cfg = load_scenario("configs/mixed_demo.yaml")
    a = write_run(generate(cfg), tmp_path / "a")
    b = write_run(generate(load_scenario("configs/mixed_demo.yaml")), tmp_path / "b")
    for name in ("data.csv", "labels.json", "config_used.yaml"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_different_seed_gives_different_data():
    a = generate(make_config(seed=1)).channels["a"].values
    b = generate(make_config(seed=2)).channels["a"].values
    assert not np.array_equal(a, b)


def test_config_used_reloads_to_the_same_run(tmp_path):
    cfg = load_scenario("configs/mixed_demo.yaml")
    run = write_run(generate(cfg), tmp_path / "run")
    again = write_run(generate(load_scenario(run / "config_used.yaml")), tmp_path / "again")
    assert (run / "data.csv").read_bytes() == (again / "data.csv").read_bytes()


def test_csv_layout_empty_cell_for_no_sample_and_nan_for_dropout(tmp_path):
    cfg = make_config(COUPLED_CHANNELS, [{"channel": "pol", "type": "dropout", "start": 2, "duration": 2}], duration=20)
    run = write_run(generate(cfg), tmp_path)
    lines = (run / "data.csv").read_text().splitlines()
    assert lines[0] == "timestamp,pol,nmr"
    cells = [line.split(",") for line in lines[1:]]
    assert cells[0][2] != "" and cells[4][2] == "" and cells[5][2] != ""  # nmr every 5th second
    assert cells[2][1] == "nan" and cells[3][1] == "nan" and cells[4][1] != "nan"  # dropout, not "no sample"
    assert len(cells) == 20


def test_labels_json_matches_generate(tmp_path):
    result = generate(load_scenario("configs/mixed_demo.yaml"))
    run = write_run(result, tmp_path)
    assert read_labels(run / "labels.json") == result.labels


def test_replay_is_time_ordered_and_merges_rates():
    events = list(replay(make_config(COUPLED_CHANNELS, duration=30)))
    times = [t for t, _ in events]
    assert all(isinstance(t, datetime) for t in times)
    assert times == sorted(times) and len(set(times)) == len(times)
    assert len(events) == 30
    for t, event in events:
        sec = int((t - times[0]).total_seconds())
        assert set(event) == ({"pol", "nmr"} if sec % 5 == 0 else {"pol"})


def test_replay_exposes_no_labels():
    cfg = make_config(anomalies=[{"channel": "a", "type": "spike", "start": 3, "magnitude": 9}], duration=10)
    for t, event in replay(cfg):
        assert set(event) == {"a"}
        assert all(isinstance(v, float) for v in event.values())


def test_replay_matches_generate_and_the_csv(tmp_path):
    cfg = load_scenario("configs/mixed_demo.yaml")
    result = generate(cfg)
    run = write_run(result, tmp_path)
    from_config = list(replay(cfg))
    from_csv = list(replay(run))
    assert [t for t, _ in from_config] == [t for t, _ in from_csv]
    pol = result.channels["polarization"].values
    assert [e["polarization"] for _, e in from_config] == list(pol)
    for (_, a), (_, b) in zip(from_config[:2000], from_csv[:2000]):
        assert a.keys() == b.keys()
        for k in a:
            assert a[k] == pytest.approx(b[k], rel=1e-9)


def test_replay_marks_dropouts_as_nan():
    cfg = make_config(anomalies=[{"channel": "a", "type": "dropout", "start": 3, "duration": 2}], duration=10)
    values = [e["a"] for _, e in replay(cfg)]
    assert [math.isnan(v) for v in values] == [False] * 3 + [True] * 2 + [False] * 5


def test_replay_speed_sleeps_between_events(monkeypatch):
    slept = []
    monkeypatch.setattr("sim.stream.time.sleep", slept.append)
    list(replay(make_config(duration=5), speed=2.0))
    assert slept == [0.5] * 4
    slept.clear()
    list(replay(make_config(duration=5)))  # speed=None: no sleeping
    assert slept == []


def test_replay_rejects_bad_speed():
    with pytest.raises(ValueError):
        list(replay(make_config(duration=5), speed=0))
