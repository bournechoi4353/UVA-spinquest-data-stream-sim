"""Labels match what was injected, exactly."""

from datetime import datetime, timedelta

import numpy as np
import pytest

from conftest import COUPLED_CHANNELS, START, effect_of, make_config
from sim.stream import generate

T0 = datetime.fromisoformat(START)


def at(seconds):
    return (T0 + timedelta(seconds=float(seconds))).isoformat()


def seconds(iso):
    return (datetime.fromisoformat(iso) - T0).total_seconds()


def one_label(anomalies, channels=None, duration=3600):
    labels = generate(make_config(channels, anomalies, duration)).labels
    assert len(labels) == len(anomalies)
    return labels


def test_label_schema_and_ids_in_time_order():
    anoms = [
        {"channel": "a", "type": "step_shift", "start": 2000, "magnitude": 3},
        {"channel": "a", "type": "spike", "start": 100, "magnitude": 8},
    ]
    labels = one_label(anoms)
    assert [l["id"] for l in labels] == ["ev_0001", "ev_0002"]
    assert [l["type"] for l in labels] == ["spike", "step_shift"]
    expected_keys = {"id", "channel", "type", "start", "end", "first_detectable", "magnitude",
                     "magnitude_sigma", "params", "severity"}
    assert all(set(l) == expected_keys for l in labels)


def test_spike_label():
    (label,) = one_label([{"channel": "a", "type": "spike", "start": 100, "magnitude": -8}])
    assert label["start"] == at(100)
    assert label["end"] == at(101)
    assert label["magnitude"] == pytest.approx(-4.0)
    assert label["magnitude_sigma"] == pytest.approx(-8.0)
    assert label["severity"] == "low"
    assert label["first_detectable"] == at(100)
    assert label["params"]["magnitude"] == -8


def test_small_spike_is_never_detectable():
    (label,) = one_label([{"channel": "a", "type": "spike", "start": 100, "magnitude": 0.5}])
    assert label["first_detectable"] is None


def test_drift_label_first_detectable_comes_after_start_and_matches_the_model():
    anom = {"channel": "a", "type": "linear_drift", "start": 1000, "duration": 2000, "magnitude": -20}
    (label,) = one_label([anom])
    eff, _, t = effect_of([anom])
    first = t[np.argmax(np.abs(eff) > 0.5)]  # 1 sigma = 0.5
    assert label["start"] == at(1000)
    assert label["end"] == at(3000)
    assert label["first_detectable"] == at(first)
    assert first > 1000
    assert label["magnitude_sigma"] == pytest.approx(-20)
    assert label["magnitude"] == pytest.approx(-10)
    assert label["severity"] == "high"


def test_end_is_null_when_effect_never_ends():
    step = {"channel": "a", "type": "step_shift", "start": 100, "magnitude": 3}
    held = {"channel": "a", "type": "linear_drift", "start": 200, "duration": 100, "magnitude": 3,
            "hold_after": True}
    labels = one_label([step, held])
    assert all(l["end"] is None for l in labels)


def test_severity_rules():
    sev = lambda a: one_label([a])[0]["severity"]
    assert sev({"channel": "a", "type": "dropout", "start": 100, "duration": 4}) == "low"
    assert sev({"channel": "a", "type": "dropout", "start": 100, "duration": 5}) == "high"
    assert sev({"channel": "a", "type": "spike_burst", "start": 100, "magnitude": 5, "count": 3, "spacing": 5}) == "high"
    assert sev({"channel": "a", "type": "stuck_sensor", "start": 100, "duration": 50}) == "high"
    assert sev({"channel": "a", "type": "transient_decay", "start": 100, "magnitude": 5, "time_constant": 30}) == "high"
    assert sev({"channel": "a", "type": "exponential_drift", "start": 100, "duration": 300, "magnitude": 5}) == "high"


def test_noise_growth_and_dropout_and_stuck_labels():
    st, ng, do = one_label([
        {"channel": "a", "type": "noise_growth", "start": 1000, "duration": 1000, "factor": 4},
        {"channel": "a", "type": "dropout", "start": 3000, "duration": 10},
        {"channel": "a", "type": "stuck_sensor", "start": 200, "duration": 50},
    ])
    # noise_growth: magnitude is the final factor; extra noise = 0.5 * sqrt(f^2 - 1) exceeds 0.5 at f = sqrt(2)
    assert ng["magnitude"] == 4 and ng["magnitude_sigma"] is None
    f_needed = np.sqrt(2.0)
    expected = 1000 + 1000 * (f_needed - 1) / 3
    assert expected - 1 < seconds(ng["first_detectable"]) <= expected + 1
    # dropout: no magnitude, detectable at once
    assert do["magnitude"] is None and do["first_detectable"] == do["start"] == at(3000)
    assert do["end"] == at(3010)
    # stuck sensor: magnitude is the frozen value minus baseline
    healthy = generate(make_config()).channels["a"].values
    assert st["magnitude"] == pytest.approx(healthy[199] - 10.0, abs=1e-6)
    assert st["magnitude_sigma"] == pytest.approx((healthy[199] - 10.0) / 0.5, abs=1e-5)


def test_stuck_first_detectable_matches_definition():
    anom = {"channel": "a", "type": "stuck_sensor", "start": 200, "duration": 50}
    (label,) = one_label([anom])
    result = generate(make_config(anomalies=[anom])).channels["a"].values
    healthy = generate(make_config()).channels["a"].values
    window = np.arange(200, 250)
    idx = window[np.argmax(np.abs(result[window] - healthy[window]) > 0.5)]
    assert label["first_detectable"] == at(idx)


def test_decoupling_label():
    drift = {"channel": "pol", "type": "linear_drift", "start": 200, "duration": 3000, "magnitude": -3.0,
             "magnitude_unit": "absolute", "hold_after": True}
    dec = {"channel": "nmr", "type": "decoupling", "start": 1500, "duration": 1000, "hold_after": True}
    labels = generate(make_config(COUPLED_CHANNELS, [drift, dec], duration=4000)).labels
    label = next(l for l in labels if l["type"] == "decoupling")
    assert label["magnitude"] is None
    assert label["end"] is None
    assert label["start"] == at(1500)
    # Recompute "missing coupled contribution > 1 sigma" from the data.
    result = generate(make_config(COUPLED_CHANNELS, [drift, dec], duration=4000))
    pol = result.channels["pol"].values
    t = result.channels["nmr"].t_us // 1_000_000
    progress = np.clip((t - 1500) / 1000, 0, 1)
    missing = 0.15 * progress * (pol[t.astype(int)] - 41.0)
    first = t[np.argmax(np.abs(missing) > 0.03)]
    assert label["first_detectable"] == at(first)
    assert first > 1500


def test_labels_match_config_for_the_demo():
    from sim.scenario import load_scenario

    cfg = load_scenario("configs/mixed_demo.yaml")
    labels = generate(cfg).labels
    assert len(labels) == len(cfg.anomalies)
    assert sorted((l["channel"], l["type"]) for l in labels) == sorted((a.channel, a.type) for a in cfg.anomalies)
