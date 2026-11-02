"""Config validation rejects bad input with readable messages."""

import copy

import pytest

from conftest import COUPLED_CHANNELS, make_config
from sim.scenario import ScenarioError, load_scenario, parse_scenario


def spike(**kw):
    return {"channel": "a", "type": "spike", "start": 100, "magnitude": 5, **kw}


def with_channel(**kw):
    cfg = make_config()
    cfg["channels"][0].update(kw)
    return cfg


def with_top(**kw):
    cfg = make_config()
    cfg.update(kw)
    return cfg


BAD = [
    ("unknown channel", make_config(anomalies=[spike(channel="zzz")]), "unknown channel 'zzz'"),
    ("unknown type", make_config(anomalies=[spike(type="bogus")]), "unknown anomaly type 'bogus'"),
    ("start outside duration", make_config(anomalies=[spike(start=5000)]), "outside the scenario duration"),
    ("negative start", make_config(anomalies=[spike(start=-1)]), "'start' must be >= 0"),
    ("window past the end",
     make_config(anomalies=[{"channel": "a", "type": "dropout", "start": 3500, "duration": 200}]),
     "after the scenario duration"),
    ("negative duration",
     make_config(anomalies=[{"channel": "a", "type": "dropout", "start": 100, "duration": -5}]),
     "'duration' must be positive"),
    ("missing parameter", make_config(anomalies=[{"channel": "a", "type": "spike", "start": 5}]),
     "missing required parameter 'magnitude'"),
    ("unknown parameter", make_config(anomalies=[spike(bogus=1)]), "unknown key"),
    ("bad unit", make_config(anomalies=[spike(magnitude_unit="volts")]), "'magnitude_unit' must be one of"),
    ("zero magnitude", make_config(anomalies=[spike(magnitude=0)]), "must be non-zero"),
    ("noise factor below 1",
     make_config(anomalies=[{"channel": "a", "type": "noise_growth", "start": 10, "duration": 50, "factor": 0.5}]),
     "greater than 1"),
    ("burst count",
     make_config(anomalies=[{"channel": "a", "type": "spike_burst", "start": 10, "magnitude": 5, "count": 1,
                             "spacing": 5}]),
     "'count' must be an integer >= 2"),
    ("burst runs off the end",
     make_config(anomalies=[{"channel": "a", "type": "spike_burst", "start": 3500, "magnitude": 5, "count": 50,
                             "spacing": 10}]),
     "last spike"),
    ("burst spacing below sample period",
     make_config(anomalies=[{"channel": "a", "type": "spike_burst", "start": 10, "magnitude": 5, "count": 3,
                             "spacing": 0.5}]),
     "shorter than the sample period"),
    ("decoupling without coupling",
     make_config(anomalies=[{"channel": "a", "type": "decoupling", "start": 10, "duration": 50}]),
     "nothing to decouple"),
    ("negative sigma", with_channel(sigma=-1.0), "'sigma' must be positive"),
    ("bad phi", with_channel(ar1_phi=1.0), "'ar1_phi' must be in"),
    ("zero rate", with_channel(sample_rate_hz=0), "'sample_rate_hz' must be positive"),
    ("unknown channel key", with_channel(color="red"), "unknown key"),
    ("coupling to unknown channel", with_channel(couples_to={"channel": "nope"}), "unknown channel 'nope'"),
    ("self coupling", with_channel(couples_to={"channel": "a"}), "cannot couple to itself"),
    ("zero duration", with_top(duration_seconds=0), "'duration_seconds' must be positive"),
    ("bad start_time", with_top(start_time="yesterday"), "ISO string"),
    ("negative seed", with_top(seed=-3), "non-negative integer"),
    ("float seed", with_top(seed=1.5), "non-negative integer"),
    ("missing seed", {k: v for k, v in make_config().items() if k != "seed"}, "missing required key 'seed'"),
    ("no channels", with_top(channels=[]), "non-empty list"),
    ("duplicate channels", with_top(channels=make_config()["channels"] * 2), "duplicate channel"),
    ("too short for the rate", with_top(duration_seconds=1), "fewer than 2 samples"),
]


@pytest.mark.parametrize("label, cfg, message", BAD, ids=[b[0] for b in BAD])
def test_bad_config_is_rejected_with_a_readable_message(label, cfg, message):
    with pytest.raises(ScenarioError, match=message):
        parse_scenario(copy.deepcopy(cfg))


def test_coupling_loop_is_rejected():
    chans = [
        {"name": "a", "baseline": 0, "sigma": 1, "couples_to": {"channel": "b", "gain": 0.1}},
        {"name": "b", "baseline": 0, "sigma": 1, "couples_to": {"channel": "a", "gain": 0.1}},
    ]
    with pytest.raises(ScenarioError, match="coupling loop"):
        parse_scenario(make_config(chans))


def test_window_shorter_than_sample_period_is_rejected_at_generation():
    from sim.stream import generate

    cfg = make_config(COUPLED_CHANNELS, [{"channel": "nmr", "type": "dropout", "start": 11, "duration": 2}])
    with pytest.raises(ScenarioError, match="contains no samples"):
        generate(cfg)


def test_defaults_are_filled_in():
    cfg = parse_scenario(make_config(COUPLED_CHANNELS, [
        {"channel": "pol", "type": "spike", "start": 5, "magnitude": 4},
        {"channel": "pol", "type": "exponential_drift", "start": 10, "duration": 300, "magnitude": 1},
    ]))
    assert cfg.channel("nmr").gain == 0.15
    assert cfg.anomalies[0].params["magnitude_unit"] == "sigma"
    assert cfg.anomalies[1].params["time_constant"] == pytest.approx(100.0)


def test_error_names_the_file(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("seed: 1\n")
    with pytest.raises(ScenarioError, match="bad.yaml"):
        load_scenario(path)


def test_example_configs_load():
    assert load_scenario("configs/healthy.yaml").anomalies == ()
    assert len(load_scenario("configs/mixed_demo.yaml").anomalies) == 6
