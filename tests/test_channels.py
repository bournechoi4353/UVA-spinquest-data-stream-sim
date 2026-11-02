"""Healthy signal model: statistics, AR(1) noise, wander, coupling."""

import numpy as np
import pytest

from conftest import COUPLED_CHANNELS, make_config
from sim.channels import ar1_noise
from sim.scenario import ScenarioError
from sim.stream import generate

LONG = 200_000  # seconds, so statistical error is small


def test_healthy_mean_and_std_match_config():
    cfg = make_config([{"name": "a", "baseline": 41.0, "sigma": 0.15}], duration=LONG)
    x = generate(cfg).channels["a"].values
    n = len(x)
    assert abs(x.mean() - 41.0) < 5 * 0.15 / np.sqrt(n)
    assert x.std() == pytest.approx(0.15, rel=0.02)


@pytest.mark.parametrize("phi", [0.3, 0.7, 0.9])
def test_ar1_keeps_target_stationary_std_and_autocorrelation(phi):
    cfg = make_config([{"name": "a", "baseline": 0.0, "sigma": 2.0, "ar1_phi": phi}], duration=LONG)
    x = generate(cfg).channels["a"].values
    assert x.std() == pytest.approx(2.0, rel=0.04)
    lag1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    assert lag1 == pytest.approx(phi, abs=0.02)


def test_ar1_fast_version_matches_plain_loop():
    z = np.random.default_rng(3).standard_normal(500)
    sigma, phi = 1.5, 0.8
    expected = np.empty_like(z)
    expected[0] = sigma * z[0]
    for t in range(1, len(z)):
        expected[t] = phi * expected[t - 1] + sigma * np.sqrt(1 - phi**2) * z[t]
    assert ar1_noise(z.copy(), sigma, phi) == pytest.approx(expected, abs=1e-9)


def test_wander_amplitude_and_period():
    amp, period = 1.0, 1000.0
    cfg = make_config(
        [{"name": "a", "baseline": 0.0, "sigma": 0.01,
          "slow_wander": {"amplitude": amp, "period_seconds": period}}],
        duration=10_000,
    )
    ch = generate(cfg).channels["a"]
    t = ch.t_us / 1e6
    # Least squares fit of  c + s*sin(wt) + k*cos(wt)  at the configured period.
    w = 2 * np.pi / period
    design = np.column_stack([np.ones_like(t), np.sin(w * t), np.cos(w * t)])
    coef, *_ = np.linalg.lstsq(design, ch.values, rcond=None)
    assert np.hypot(coef[1], coef[2]) == pytest.approx(amp, rel=0.01)
    # The dominant spectral peak is at the configured period.
    spectrum = np.abs(np.fft.rfft(ch.values - ch.values.mean()))
    freqs = np.fft.rfftfreq(len(t), d=1.0)
    assert 1.0 / freqs[np.argmax(spectrum)] == pytest.approx(period, rel=0.01)


def test_coupled_channel_total_sigma_and_correlation():
    cfg = make_config(COUPLED_CHANNELS, duration=LONG)
    result = generate(cfg)
    pol, nmr = result.channels["pol"], result.channels["nmr"]
    assert nmr.values.std() == pytest.approx(0.03, rel=0.03)  # TOTAL sigma is preserved
    pol_at_nmr = pol.values[(nmr.t_us // 1_000_000).astype(int)]
    corr = np.corrcoef(pol_at_nmr, nmr.values)[0, 1]
    assert corr == pytest.approx(0.15 * 0.15 / 0.03, abs=0.02)  # gain * sigma_src / sigma = 0.75


def test_nmr_is_sampled_every_five_seconds():
    t = generate(make_config(COUPLED_CHANNELS, duration=100)).channels["nmr"].t_us
    assert list(t[:4]) == [0, 5_000_000, 10_000_000, 15_000_000]


def test_gain_that_leaves_no_own_noise_is_rejected():
    bad = [dict(COUPLED_CHANNELS[0]), dict(COUPLED_CHANNELS[1])]
    bad[1]["couples_to"] = {"channel": "pol", "gain": 0.2}  # 0.2 * 0.15 = 0.03 = total sigma
    with pytest.raises(ScenarioError, match="no noise of its own"):
        generate(make_config(bad))


def test_adding_anomalies_does_not_change_healthy_noise():
    anom = [{"channel": "a", "type": "spike", "start": 100, "magnitude": 5}]
    a = generate(make_config(anomalies=[])).channels["a"].values
    b = generate(make_config(anomalies=anom)).channels["a"].values
    assert np.count_nonzero(a != b) == 1
