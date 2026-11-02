"""Each anomaly type produces the expected effect at the expected time.

`effect_of` subtracts the healthy run from the anomalous run, which leaves the
injected effect exactly (same seed, same noise).
"""

import numpy as np
import pytest

from conftest import COUPLED_CHANNELS, effect_of, make_config
from sim.stream import generate

SIGMA = 0.5  # sigma of the default test channel "a"


def anomaly(type_, start, **params):
    return {"channel": "a", "type": type_, "start": start, **params}


def test_spike_hits_exactly_one_sample_with_the_right_size():
    eff, _, _ = effect_of([anomaly("spike", 1000, magnitude=-6)])
    assert np.flatnonzero(eff).tolist() == [1000]
    assert eff[1000] == pytest.approx(-6 * SIGMA)


def test_spike_burst_count_spacing_and_alternating_sign():
    eff, _, _ = effect_of([anomaly("spike_burst", 500, magnitude=5, count=4, spacing=10, alternate_sign=True)])
    assert np.flatnonzero(eff).tolist() == [500, 510, 520, 530]
    assert eff[[500, 510, 520, 530]] == pytest.approx(np.array([1, -1, 1, -1]) * 5 * SIGMA)


def test_step_shift_offset_and_window():
    eff, _, _ = effect_of([anomaly("step_shift", 1000, magnitude=3, duration=600)])
    assert np.all(eff[:1000] == 0)
    assert eff[1000:1600] == pytest.approx(3 * SIGMA)
    assert np.all(eff[1600:] == 0)


def test_step_shift_with_no_end_lasts_forever_and_supports_absolute_units():
    eff, _, _ = effect_of([anomaly("step_shift", 1000, magnitude=2.0, magnitude_unit="absolute")])
    assert eff[1000:] == pytest.approx(2.0)
    assert np.all(eff[:1000] == 0)


def test_linear_drift_reaches_magnitude_at_end_and_then_returns_or_holds():
    eff, _, _ = effect_of([anomaly("linear_drift", 1000, duration=1000, magnitude=-4, magnitude_unit="absolute")])
    assert eff[1000] == pytest.approx(0)
    assert eff[1500] == pytest.approx(-2.0)
    assert eff[2000] == pytest.approx(-4.0)  # the sample AT the end has the full effect
    assert np.all(eff[2001:] == 0)

    held, _, _ = effect_of(
        [anomaly("linear_drift", 1000, duration=1000, magnitude=-4, magnitude_unit="absolute", hold_after=True)]
    )
    assert held[2001:] == pytest.approx(-4.0)


def test_exponential_drift_is_slow_then_fast_and_reaches_magnitude():
    eff, _, _ = effect_of([anomaly("exponential_drift", 1000, duration=1000, time_constant=250, magnitude=4,
                                   magnitude_unit="absolute")])
    assert eff[1000] == pytest.approx(0, abs=1e-12)
    assert eff[2000] == pytest.approx(4.0)
    assert eff[1500] < 0.5 * 4.0  # slower than linear in the first half
    assert np.all(np.diff(eff[1000:2001]) > 0)  # monotone, accelerating
    assert np.all(np.diff(eff[1000:2001], 2) > 0)
    assert np.all(eff[2001:] == 0)


def test_exponential_drift_extreme_time_constant_does_not_overflow():
    eff, _, _ = effect_of([anomaly("exponential_drift", 100, duration=1000, time_constant=0.5, magnitude=1,
                                   magnitude_unit="absolute")])
    assert np.all(np.isfinite(eff))
    assert eff[1100] == pytest.approx(1.0)


def test_noise_growth_changes_spread_but_not_mean():
    duration = 40_000
    anomalies = [anomaly("noise_growth", 10_000, duration=20_000, factor=4)]
    eff, healthy, _ = effect_of(anomalies, duration=duration)
    anomalous = healthy + eff
    before = anomalous[:10_000]
    after_end = anomalous[30_001:]  # back to normal (hold_after false)
    last_part = anomalous[28_000:30_001]
    assert before.std() == pytest.approx(SIGMA, rel=0.05)
    assert after_end.std() == pytest.approx(SIGMA, rel=0.05)
    # Near the end of the window the noise is ~4x (factor 3.7 to 4.0 over this slice, rms about 3.85).
    assert last_part.std() == pytest.approx(SIGMA * 3.85, rel=0.08)
    assert anomalous[10_000:30_001].mean() == pytest.approx(healthy.mean(), abs=0.1)


def test_noise_growth_exact_scaling_of_noise_part():
    eff, healthy, _ = effect_of([anomaly("noise_growth", 1000, duration=1000, factor=3)])
    # effect = (f(t) - 1) * noise, so effect / (healthy - baseline) = f(t) - 1 = 2 * progress
    noise = healthy - 10.0
    assert eff[1500] / noise[1500] == pytest.approx(1.0)  # f = 2 at the midpoint
    assert eff[2000] / noise[2000] == pytest.approx(2.0)  # f = 3 at the end


def test_transient_decay_jump_and_time_constant():
    eff, _, _ = effect_of([anomaly("transient_decay", 1000, magnitude=10, time_constant=100)])
    assert np.all(eff[:1000] == 0)
    assert eff[1000] == pytest.approx(10 * SIGMA)
    assert eff[1100] == pytest.approx(10 * SIGMA / np.e)
    assert eff[1500] == pytest.approx(10 * SIGMA * np.exp(-5))


def test_dropout_gives_nans_only_inside_the_window():
    result = generate(make_config(anomalies=[anomaly("dropout", 1000, duration=30)]))
    v = result.channels["a"].values
    assert np.flatnonzero(np.isnan(v)).tolist() == list(range(1000, 1030))


def test_stuck_sensor_freezes_at_last_value_with_zero_noise():
    result = generate(make_config(anomalies=[anomaly("stuck_sensor", 1000, duration=500)]))
    healthy = generate(make_config()).channels["a"].values
    v = result.channels["a"].values
    assert np.all(v[1000:1500] == healthy[999])
    assert np.array_equal(v[:1000], healthy[:1000])
    assert np.array_equal(v[1500:], healthy[1500:])


def test_decoupling_breaks_the_relationship_but_not_the_channels():
    drift = {"channel": "pol", "type": "linear_drift", "start": 200, "duration": 3000, "magnitude": -3.0,
             "magnitude_unit": "absolute", "hold_after": True}
    dec = {"channel": "nmr", "type": "decoupling", "start": 1500, "duration": 1000, "hold_after": True}
    with_drift = generate(make_config(COUPLED_CHANNELS, [drift], duration=4000))
    both = generate(make_config(COUPLED_CHANNELS, [drift, dec], duration=4000))
    pol = both.channels["pol"].values
    nmr_c, nmr_d = with_drift.channels["nmr"], both.channels["nmr"]
    # Polarization is untouched by decoupling a different channel.
    assert np.array_equal(pol, with_drift.channels["pol"].values)
    # Before the window: identical. After it: the whole coupled term (0.15 * deviation) is gone.
    t = nmr_d.t_us // 1_000_000
    before, after = t < 1500, t >= 2500
    assert np.array_equal(nmr_c.values[before], nmr_d.values[before])
    pol_dev = pol[t[after].astype(int)] - 41.0
    assert nmr_d.values[after] - nmr_c.values[after] == pytest.approx(-0.15 * pol_dev)
    # With the drift at -3, the coupled channel without decoupling moved by about -0.45;
    # the decoupled one stays near its baseline.
    assert nmr_c.values[t == 3500].item() < 2.5 - 0.3
    assert abs(nmr_d.values[t == 3500].item() - 2.5) < 0.15


def test_coupled_channel_follows_source_anomalies_but_not_source_measurement_faults():
    drift = {"channel": "pol", "type": "step_shift", "start": 100, "magnitude": -2.0, "magnitude_unit": "absolute"}
    dropout = {"channel": "pol", "type": "dropout", "start": 300, "duration": 100}
    stuck = {"channel": "pol", "type": "stuck_sensor", "start": 500, "duration": 100}
    base = generate(make_config(COUPLED_CHANNELS, [], duration=1000)).channels["nmr"].values
    run = generate(make_config(COUPLED_CHANNELS, [drift], duration=1000)).channels["nmr"].values
    faults = generate(make_config(COUPLED_CHANNELS, [drift, dropout, stuck], duration=1000)).channels["nmr"].values
    t = np.arange(len(base)) * 5
    assert (run - base)[t >= 100] == pytest.approx(0.15 * -2.0)  # physical shift propagates
    assert np.array_equal(run, faults)  # faults on the source do not
    assert not np.isnan(faults).any()


def test_anomalies_compose_on_one_channel_and_across_channels():
    chans = [{"name": "a", "baseline": 10.0, "sigma": 0.5}, {"name": "b", "baseline": 0.0, "sigma": 1.0}]
    anoms = [
        {"channel": "a", "type": "step_shift", "start": 100, "magnitude": 2, "magnitude_unit": "absolute"},
        {"channel": "a", "type": "linear_drift", "start": 200, "duration": 100, "magnitude": 1,
         "magnitude_unit": "absolute", "hold_after": True},
        {"channel": "a", "type": "spike", "start": 250, "magnitude": 10},
        {"channel": "b", "type": "step_shift", "start": 150, "magnitude": 3},
    ]
    eff_a, _, _ = effect_of(anoms, "a", chans, duration=500)
    eff_b, _, _ = effect_of(anoms, "b", chans, duration=500)
    assert eff_a[400] == pytest.approx(2.0 + 1.0)  # step + held drift
    assert eff_a[250] == pytest.approx(2.0 + 0.5 + 10 * 0.5)  # step + half drift + spike
    assert eff_b[400] == pytest.approx(3.0)
