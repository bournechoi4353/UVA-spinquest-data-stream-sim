"""Reference detectors (Phase 5, extended).

Every channel gets one ChannelMonitor, calibrated from healthy data:
``target`` is the median and ``sigma`` the MAD estimate
(sigma = 1.4826 * median(|x - median|)) of the first-difference-free healthy
values, and ``period_s`` is the typical sample spacing.

Detectors inside a ChannelMonitor (all causal, one sample at a time):

1. spike      z = (x - mean) / sigma with an adaptive mean (alpha 0.01) that
              does NOT update on flagged points. |z| > spike_z (default 5).
2. drift      two-sided CUSUM against the FROZEN healthy target, slack 0.5
              sigma, threshold 5 sigma. Inputs are clipped to +/- 3 sigma so
              one spike cannot trigger it. ``since`` is the time the score
              first rose above zero. Reported as drift_up or drift_down.
3. stuck      the same value repeated for stuck_samples samples in a row.
4. noisy      an EWMA of squared sample-to-sample differences above
              noisy_ratio^2 (default 2^2) times its healthy level, for noisy_hold samples.
5. missing    missing_samples NaN readings in a row (a dropout).
Per feed (FeedWatch): no row for longer than its limit (a feed outage). The limit is
outage_periods periods, or, after Engine calibration, 1.5 times the largest healthy
gap between rows, kept between outage_periods periods and outage_max_s.
Per coupled pair (CouplingWatch): CUSUM on the residual of the coupled
channel after removing gain * source deviation (decoupled).

NaN readings are skipped without resetting any state.

Each detector produces Condition objects that the state store keeps. A
condition is active until its detector clears it (for example the CUSUM score
returns to zero, or the value changes again).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

MAD_SCALE = 1.4826


@dataclass
class DetectorSettings:
    spike_z: float = 5.0
    spike_return_z: float = 3.0
    alpha: float = 0.01
    cusum_slack: float = 0.5
    cusum_threshold: float = 5.0
    cusum_clip: float = 3.0
    stuck_samples: int = 20
    stuck_min_seconds: float = 60.0
    noisy_ratio: float = 2.0
    noisy_alpha: float = 0.02
    noisy_hold: int = 30
    missing_samples: int = 3
    drift_min_sigma: float = 1.0     # a drift alarm needs the recent mean deviation to be at least this
    drift_clear_s: float = 1800.0    # an active drift ends after this long with the recent deviation below it
    dev_alpha: float = 0.01          # EWMA weight of the recent mean deviation (about 100 samples)
    drift_rearm_sigma: float = 2.0   # after a drift ends by settling, the same side needs this much to raise again
    outage_periods: float = 5.0
    outage_max_s: float = 300.0     # calibrated outage limits never exceed this (5 min)
    decouple_threshold: float = 8.0


@dataclass
class Condition:
    """One conclusion about a channel or feed. Times are datetimes (UTC or naive, as the stream gives)."""

    subject: str                 # channel key, or a feed name for feed_outage
    kind: str                    # spike, drift_up, drift_down, stuck, noisy, missing, feed_outage, decoupled
    since: datetime              # when the effect started, as far as the detector can tell
    detected_at: datetime        # when the detector raised it
    active: bool = True
    ended_at: datetime | None = None
    magnitude: float | None = None        # in channel units, signed (deviation from target)
    magnitude_sigma: float | None = None  # in sigma
    peak_sigma: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        def iso(t):
            return None if t is None else t.isoformat()
        return {"subject": self.subject, "kind": self.kind, "since": iso(self.since),
                "detected_at": iso(self.detected_at), "active": self.active, "ended_at": iso(self.ended_at),
                "magnitude": _round(self.magnitude), "magnitude_sigma": _round(self.magnitude_sigma),
                "peak_sigma": _round(self.peak_sigma), "detail": self.detail}


def _round(x: float | None, digits: int = 4) -> float | None:
    if x is None or not math.isfinite(x):
        return None
    return float(f"{x:.{digits}g}")


@dataclass
class Calibration:
    """Healthy reference for one channel.

    ``cusum_h`` and ``noisy_level`` are per-channel thresholds. With the plain
    Phase 5 settings they are the fixed defaults (5 sigma and noisy_ratio^2).
    ``adapt_thresholds`` raises them to ADAPT_MARGIN times the largest score
    seen on the channel's own healthy data, which accounts for autocorrelated
    noise and slow wander (both make the fixed 5 sigma CUSUM fire constantly).
    """

    target: float
    sigma: float
    period_s: float
    cusum_h: float | None = None
    noisy_level: float | None = None
    diff_var: float = 1.0      # healthy mean of ((x[k] - x[k-1]) / (sigma * sqrt 2))^2; 1 for white noise
    cusum_scale: float = 1.0   # CUSUM inputs are divided by sigma * cusum_scale (autocorrelation correction)
    stuck_normal: bool = False  # the channel holds values a lot when healthy: no stuck detection
    flat: bool = False          # every healthy value was identical (for example a channel reading 0)
    missing_after_s: float | None = None   # no value for this long counts as missing (wide feeds)


STUCK_NORMAL_SHARE = 0.2        # healthy share of repeated consecutive values above which "stuck" is off
MISSING_GAP_QUANTILE = 99.9
MISSING_GAP_FACTOR = 2.0
MISSING_MIN_PERIODS = 10.0
MISSING_LARGEST_FACTOR = 1.5    # a channel may go 1.5 x its largest healthy gap without a value ...
MISSING_LARGEST_MAX_S = 300.0   # ... but that rule never asks for more than 5 minutes
FLAT_SIGMA_REL = 1e-3           # sigma of a channel that was constant in calibration: 0.1% of its level (min 1e-3)
SPREAD_Z = 4.653                # P99 - P1 of a Gaussian, in sigma
HELD_DIFF_VAR = 0.5             # healthy noise reference floor for holding and constant channels


def calibrate(values: np.ndarray, times_s: np.ndarray | None = None, default_period_s: float = 1.0) -> Calibration:
    """Robust target and sigma from healthy values (NaN ignored).

    Real channels can be quantized, hold values for long stretches, or be constant (a channel that
    is not connected). The MAD then badly underestimates how much the channel moves: a level sensor
    that toggles between two readout codes, or a temperature that updates in steps, has a MAD close
    to 0, and every normal update would look like a many-sigma spike (seen on real ptdb data, QT
    LIT501 and FC502, 2025-11). So:

    * channels that hold values (repeat share above STUCK_NORMAL_SHARE) get the largest of the MAD
      sigma, the central-98% spread in sigma units ((P99 - P1) / 4.653), and the median size of a
      non-zero update (so one ordinary update is at most 1 sigma);
    * a channel that was constant during calibration gets FLAT_SIGMA_REL of its level (at least
      1e-3), so a small later change is not thousands of sigma;
    * holding channels also get ``stuck_normal`` (stuck detection off).

    Channels that do not hold values keep the plain MAD sigma.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 10:
        raise ValueError("need at least 10 healthy values to calibrate")
    target = float(np.median(v))
    sigma = MAD_SCALE * float(np.median(np.abs(v - target)))
    steps = np.abs(np.diff(v))
    repeat_share = float((steps == 0).mean()) if len(steps) else 0.0
    flat = bool(np.all(v == v[0]))
    nonzero = steps[steps > 0]
    if flat:
        sigma = FLAT_SIGMA_REL * max(abs(target), 1.0)
    elif repeat_share > STUCK_NORMAL_SHARE:
        lo, hi = np.percentile(v, [1.0, 99.0])
        sigma = max(sigma, float(hi - lo) / SPREAD_Z, float(np.median(nonzero)) if len(nonzero) else 0.0,
                    1e-9 * max(abs(target), 1.0))
    elif sigma <= 0:
        sigma = max(float(np.std(v)), float(nonzero.min()) if len(nonzero) else 0.0,
                    1e-9 * max(abs(target), 1.0))
    period = default_period_s
    missing_after = None
    if times_s is not None and len(times_s) > 2:
        t = np.asarray(times_s, dtype=float)
        d = np.diff(t)
        d = d[d > 0]
        if len(d):
            period = float(np.mean(d[d < 5 * np.median(d)])) if np.median(d) > 0 else float(np.mean(d))
            missing_after = max(MISSING_GAP_FACTOR * float(np.percentile(d, MISSING_GAP_QUANTILE)),
                                MISSING_MIN_PERIODS * period,
                                MISSING_LARGEST_FACTOR * min(float(d.max()), MISSING_LARGEST_MAX_S))
    d = np.diff(v) / (sigma * math.sqrt(2.0))
    diff_var = float(np.mean(np.minimum(d * d, 25.0))) if len(d) > 5 else 1.0
    if flat or repeat_share > STUCK_NORMAL_SHARE:
        # Mostly zero differences make the healthy noise level close to 0, and then any burst of
        # ordinary updates is "noisy". Floor it at HELD_DIFF_VAR (updates of one sigma on every
        # other sample count as healthy).
        diff_var = max(diff_var, HELD_DIFF_VAR)
    return Calibration(target, sigma, period, diff_var=max(diff_var, 1e-6),
                       stuck_normal=flat or repeat_share > STUCK_NORMAL_SHARE, flat=flat,
                       missing_after_s=missing_after)


ADAPT_MARGIN = 1.5
FALSE_ALARM_DAYS = 30.0        # design target: one false drift alarm per channel per 30 days


def siegmund_h(arl_samples: float, k: float = 0.5) -> float:
    """CUSUM threshold h with in-control average run length ``arl_samples`` (one side).

    Siegmund's approximation for unit-variance Gaussian increments and slack k:
    ARL0 = (exp(2 k b) - 2 k b - 1) / (2 k^2), with b = h + 1.166. Solved by bisection.
    """
    lo, hi = 0.0, 200.0
    for _ in range(100):
        h = 0.5 * (lo + hi)
        b = h + 1.166
        arl = (math.exp(min(2 * k * b, 700)) - 2 * k * b - 1) / (2 * k * k)
        lo, hi = (h, hi) if arl < arl_samples else (lo, h)
    return 0.5 * (lo + hi)


def lag1(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 20:
        return 0.0
    v = v - v.mean()
    den = float((v * v).sum())
    return float((v[:-1] * v[1:]).sum() / den) if den > 0 else 0.0


def healthy_scores(values: np.ndarray, cal: Calibration, s: DetectorSettings) -> tuple[float, float]:
    """Largest CUSUM score and largest noise EWMA on healthy values, using the detector recursions."""
    up = dn = best = 0.0
    var, best_var, last = cal.diff_var, 0.0, None
    k = math.sqrt(2.0) * cal.sigma
    for x in values:
        if x != x:
            continue
        zt = max(-s.cusum_clip, min(s.cusum_clip, (x - cal.target) / (cal.sigma * cal.cusum_scale)))
        up = max(0.0, up + zt - s.cusum_slack)
        dn = max(0.0, dn - zt - s.cusum_slack)
        best = max(best, up, dn)
        if last is not None:
            var += s.noisy_alpha * (min(((x - last) / k) ** 2, 25.0) - var)
            best_var = max(best_var, var)
        last = x
    return best, best_var


def adapt_thresholds(values: np.ndarray, cal: Calibration, s: DetectorSettings,
                     margin: float = ADAPT_MARGIN) -> Calibration:
    """Per-channel thresholds from healthy data, never below the Phase 5 defaults.

    1. Autocorrelation: CUSUM inputs are scaled by sqrt((1 + phi) / (1 - phi)), the
       long-run standard deviation factor of AR(1) noise, with phi the healthy lag-1
       autocorrelation. Correlated noise otherwise sums up like a drift.
    2. Threshold: the largest of the Phase 5 value, the Siegmund threshold for one false
       alarm per FALSE_ALARM_DAYS (two-sided, at the channel's sample rate), and
       ``margin`` times the largest score on the healthy data (covers slow wander).
    """
    v = np.asarray(values, dtype=float)
    phi = min(max(lag1(v - cal.target), 0.0), 0.98)
    cal.cusum_scale = math.sqrt((1.0 + phi) / (1.0 - phi))
    best, best_var = healthy_scores(v, cal, s)
    arl = 2.0 * FALSE_ALARM_DAYS * 86400.0 / max(cal.period_s, 1e-3)     # two one-sided CUSUMs
    cal.cusum_h = max(s.cusum_threshold, siegmund_h(arl, s.cusum_slack), margin * best)
    cal.noisy_level = max(s.noisy_ratio ** 2 * cal.diff_var, margin * best_var)
    return cal


class ChannelMonitor:
    """All per-channel detectors for one channel."""

    def __init__(self, key: str, cal: Calibration, settings: DetectorSettings | None = None):
        self.key = key
        self.cal = cal
        self.s = settings or DetectorSettings()
        self.h = cal.cusum_h if cal.cusum_h is not None else self.s.cusum_threshold
        self.noisy_level = (cal.noisy_level if cal.noisy_level is not None
                            else self.s.noisy_ratio ** 2 * cal.diff_var)
        self.pending_spike: Condition | None = None
        self.flag_run = 0
        self.mean = cal.target
        self.s_up = 0.0
        self.s_dn = 0.0
        self.up_since: datetime | None = None
        self.dn_since: datetime | None = None
        self.drift: Condition | None = None
        self.dev_ewma: float | None = None       # recent mean deviation from target, channel units
        self.small_since: datetime | None = None  # when the active drift's deviation fell below drift_min_sigma
        self.calm_count = 0                      # samples in a row with the noise EWMA below its clear level
        self.armed = {"drift_up": True, "drift_down": True}   # False after a settled clear, until the CUSUM is 0
        self.last_value: float | None = None
        self.last_time: datetime | None = None
        self.repeat = 0
        self.repeat_since: datetime | None = None
        self.stuck: Condition | None = None
        self.var_ewma = cal.diff_var
        self.noisy_count = 0
        self.noisy_since: datetime | None = None
        self.noisy: Condition | None = None
        self.nan_run = 0
        self.nan_since: datetime | None = None
        self.missing: Condition | None = None
        self.n = 0

    # Each update returns the list of conditions that were raised or ended by this sample.
    def update(self, t: datetime, x: float) -> list[Condition]:
        out: list[Condition] = []
        s, cal = self.s, self.cal
        if x is None or x != x:
            self.nan_run += 1
            if self.nan_run == 1:
                self.nan_since = t
            if self.nan_run >= s.missing_samples and self.missing is None:
                self.missing = Condition(self.key, "missing", self.nan_since, t,
                                         detail={"samples": self.nan_run})
                out.append(self.missing)
            return out
        if self.missing is not None:
            self.missing.active, self.missing.ended_at = False, t
            self.missing.detail["samples"] = self.nan_run
            out.append(self.missing)
            self.missing = None
        self.nan_run = 0
        self.n += 1

        z = (x - self.mean) / cal.sigma
        # 1. spike against the adaptive mean (no update on flagged points). A flagged sample is
        #    confirmed as a spike only when the next sample is back within spike_return_z; if the
        #    next one is flagged too, it is a level change (left to the CUSUM), not a spike.
        flagged = abs(z) > s.spike_z and self.drift is None
        if self.pending_spike is not None:
            # Back near normal, and far enough from the healthy target: it really was a spike.
            # (A step ending can look like a jump against the lagging adaptive mean.)
            if abs(z) < s.spike_return_z and abs(self.pending_spike.magnitude_sigma) >= s.spike_return_z:
                out.append(self.pending_spike)
            self.pending_spike = None
        if flagged:
            self.flag_run += 1
            if self.flag_run == 1:
                self.pending_spike = Condition(self.key, "spike", t, t, active=False, ended_at=t,
                                               magnitude=x - cal.target,
                                               magnitude_sigma=(x - cal.target) / cal.sigma, peak_sigma=z)
        else:
            self.flag_run = 0
            self.mean += s.alpha * (x - self.mean)

        # 2. CUSUM against the frozen target
        zt = max(-s.cusum_clip, min(s.cusum_clip, (x - cal.target) / (cal.sigma * cal.cusum_scale)))
        prev_up, prev_dn = self.s_up, self.s_dn
        cap = 2.0 * self.h   # a capped score clears soon after the signal is back to normal
        self.s_up = min(cap, max(0.0, self.s_up + zt - s.cusum_slack))
        self.s_dn = min(cap, max(0.0, self.s_dn - zt - s.cusum_slack))
        if prev_up == 0 and self.s_up > 0:
            self.up_since = t
        if prev_dn == 0 and self.s_dn > 0:
            self.dn_since = t
        dev = x - cal.target
        self.dev_ewma = dev if self.dev_ewma is None else self.dev_ewma + s.dev_alpha * (dev - self.dev_ewma)
        # The magnitude gate and settling are part of the adaptive mode (cusum_h set by
        # adapt_thresholds); the plain Phase 5 mode keeps the bare CUSUM as its baseline.
        gated = cal.cusum_h is not None
        big = s.drift_min_sigma * cal.sigma if gated else -math.inf
        if self.s_up == 0.0:
            self.armed["drift_up"] = True
        if self.s_dn == 0.0:
            self.armed["drift_down"] = True
        if self.drift is None:
            # The CUSUM decides that the level moved; the recent mean deviation must also be at
            # least drift_min_sigma, so a channel sitting just above the slack is not "abnormal".
            # After a drift ended by settling, the same side needs drift_rearm_sigma until the
            # channel has really come back (its CUSUM reached 0), so it does not flap.
            need_up = big if self.armed["drift_up"] else s.drift_rearm_sigma * cal.sigma
            need_dn = big if self.armed["drift_down"] else s.drift_rearm_sigma * cal.sigma
            if self.s_up > self.h and self.dev_ewma >= need_up:
                self.drift = Condition(self.key, "drift_up", self.up_since or t, t)
                out.append(self.drift)
            elif self.s_dn > self.h and -self.dev_ewma >= need_dn:
                self.drift = Condition(self.key, "drift_down", self.dn_since or t, t)
                out.append(self.drift)
            self.small_since = None
        if self.drift is not None:
            up = self.drift.kind == "drift_up"
            score = self.s_up if up else self.s_dn
            # Magnitude: a slow EWMA of the deviation while the alarm is active.
            m = self.drift.magnitude
            self.drift.magnitude = dev if m is None else m + 0.05 * (dev - m)
            self.drift.magnitude_sigma = self.drift.magnitude / cal.sigma
            self.drift.peak_sigma = max(self.drift.peak_sigma or 0.0, abs(self.drift.magnitude_sigma))
            small = (self.dev_ewma < big) if up else (self.dev_ewma > -big)
            if small:
                self.small_since = self.small_since or t
            else:
                self.small_since = None
            settled = gated and small and (t - self.small_since).total_seconds() >= s.drift_clear_s
            if score == 0.0 or settled:
                self.drift.active, self.drift.ended_at = False, t
                out.append(self.drift)
                self.drift = None
                self.small_since = None
                if settled:          # this side needs a clearly larger change before it raises again
                    self.armed["drift_up" if up else "drift_down"] = False

        # 3. stuck: identical consecutive values
        if self.last_value is not None and x == self.last_value:
            self.repeat += 1
        else:
            self.repeat = 0
            self.repeat_since = t
            if self.stuck is not None:
                self.stuck.active, self.stuck.ended_at = False, t
                out.append(self.stuck)
                self.stuck = None
        need = max(s.stuck_samples, int(math.ceil(s.stuck_min_seconds / max(cal.period_s, 1e-9))))
        if self.repeat + 1 >= need and self.stuck is None and not cal.stuck_normal:
            self.stuck = Condition(self.key, "stuck", self.repeat_since or t, t, magnitude=x - cal.target,
                                   magnitude_sigma=(x - cal.target) / cal.sigma, detail={"value": _round(x, 8)})
            out.append(self.stuck)

        # 4. noisy: EWMA of squared sample-to-sample differences (a level change moves one
        #    difference only, so steps and drifts do not look like noise)
        if self.last_value is not None:
            dd = min(((x - self.last_value) / (math.sqrt(2.0) * cal.sigma)) ** 2, 25.0)
            self.var_ewma += s.noisy_alpha * (dd - self.var_ewma)
        if self.var_ewma > self.noisy_level:
            if self.noisy_count == 0:
                self.noisy_since = t
            self.noisy_count += 1
            if self.noisy_count >= s.noisy_hold and self.noisy is None:
                self.noisy = Condition(self.key, "noisy", self.noisy_since or t, t)
                out.append(self.noisy)
            if self.noisy is not None:
                ratio = math.sqrt(self.var_ewma / cal.diff_var)
                self.noisy.magnitude_sigma = ratio
                self.noisy.detail["noise_ratio"] = _round(ratio)
        else:
            self.noisy_count = 0
        # The noise alarm ends only after noisy_hold samples in a row below the clear level (halfway
        # between healthy and the threshold), so noise hovering at the threshold does not flap.
        if self.noisy is not None:
            if self.var_ewma < 0.5 * (cal.diff_var + self.noisy_level):
                self.calm_count += 1
            else:
                self.calm_count = 0
            if self.calm_count >= s.noisy_hold:
                self.noisy.active, self.noisy.ended_at = False, t
                out.append(self.noisy)
                self.noisy = None
                self.calm_count = 0

        self.last_value, self.last_time = x, t
        return out


class FeedWatch:
    """Raises feed_outage when a feed has had no row for outage_periods periods."""

    def __init__(self, feed: str, period_s: float, settings: DetectorSettings | None = None):
        self.feed = feed
        self.period_s = period_s
        self.s = settings or DetectorSettings()
        self.last_row: datetime | None = None
        self.outage: Condition | None = None

    def row(self, t: datetime) -> list[Condition]:
        out = []
        if self.outage is not None:
            self.outage.active, self.outage.ended_at = False, t
            self.outage.detail["gap_s"] = _round((t - self.outage.since).total_seconds())
            out.append(self.outage)
            self.outage = None
        self.last_row = t
        return out

    limit_s: float | None = None   # set by Engine calibration from healthy row gaps; else outage_periods

    def limit(self) -> float:
        return self.limit_s if self.limit_s is not None else self.s.outage_periods * self.period_s

    def tick(self, now: datetime) -> list[Condition]:
        if self.last_row is None or self.outage is not None:
            return []
        gap = (now - self.last_row).total_seconds()
        if gap > self.limit():
            self.outage = Condition(self.feed, "feed_outage", self.last_row, now,
                                    detail={"gap_s": _round(gap)})
            return [self.outage]
        return []


class CouplingWatch:
    """Raises decoupled when a coupled channel stops following its source.

    residual r = (y - target_y) - gain * (x_latest - target_x), in units of the
    coupled channel's sigma, fed to a two-sided CUSUM (slack 0.5, clip 3).
    """

    def __init__(self, channel: str, source: str, gain: float, cal_y: Calibration, cal_x: Calibration,
                 settings: DetectorSettings | None = None):
        self.channel, self.source, self.gain = channel, source, gain
        self.cy, self.cx = cal_y, cal_x
        self.s = settings or DetectorSettings()
        self.x: float | None = None
        self.s_up = self.s_dn = 0.0
        self.since: datetime | None = None
        self.cond: Condition | None = None
        self.resid_sigma = cal_y.sigma
        self.offset = 0.0
        self.threshold = self.s.decouple_threshold
        self.s_cap = None

    def source_value(self, x: float) -> None:
        if x == x:
            self.x = x

    def update(self, t: datetime, y: float) -> list[Condition]:
        if self.x is None or y != y:
            return []
        r = ((y - self.cy.target) - self.gain * (self.x - self.cx.target) - self.offset) / self.resid_sigma
        r = max(-3.0, min(3.0, r))
        prev = max(self.s_up, self.s_dn)
        cap = 2.0 * self.threshold
        self.s_up = min(cap, max(0.0, self.s_up + r - 0.5))
        self.s_dn = min(cap, max(0.0, self.s_dn - r - 0.5))
        score = max(self.s_up, self.s_dn)
        if prev == 0 and score > 0:
            self.since = t
        out = []
        if self.cond is None and score > self.threshold:
            self.cond = Condition(self.channel, "decoupled", self.since or t, t,
                                  detail={"source": self.source, "gain": self.gain})
            out.append(self.cond)
        elif self.cond is not None and score == 0:
            self.cond.active, self.cond.ended_at = False, t
            out.append(self.cond)
            self.cond = None
        return out
