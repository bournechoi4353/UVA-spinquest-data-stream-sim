"""Aggregate statistics for the profiler. Pure numpy and pandas, no database access.

Raw values only ever pass through these functions in memory, one chunk at a
time. Each accumulator keeps counts, sums, and pooled sample arrays (used only
to compute percentiles in memory) and hands back aggregates.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MAD_SCALE = 1.4826
ROLLING_WINDOW = 61
PERCENTILES = (1, 5, 50, 95, 99)
MAX_GAPS = 20


def _r(x: float | None, digits: int = 6) -> float | None:
    """Round for output; None for missing or non-finite values."""
    if x is None or not math.isfinite(x):
        return None
    return float(f"{x:.{digits}g}")


def mad_sigma(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    return MAD_SCALE * float(np.median(np.abs(x - np.median(x))))


def residual(values: np.ndarray, window: int = ROLLING_WINDOW) -> np.ndarray:
    """Values minus a centered rolling median (NaN where the value or the median is missing)."""
    s = pd.Series(values)
    med = s.rolling(window, center=True, min_periods=window // 2 + 1).median()
    return (s - med).to_numpy()


@dataclass
class ChannelAccumulator:
    """Per-channel statistics, fed one chunk of consecutive rows at a time."""

    n: int = 0
    nulls: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    vmin: float = math.inf
    vmax: float = -math.inf
    samples: list[np.ndarray] = field(default_factory=list)
    residuals: list[np.ndarray] = field(default_factory=list)
    lag_sxy: float = 0.0
    lag_sxx: float = 0.0
    lag_syy: float = 0.0
    lag_sx: float = 0.0
    lag_sy: float = 0.0
    lag_n: int = 0
    within_var_sum: float = 0.0
    resid_var_sum: float = 0.0
    pairs: int = 0
    repeats: int = 0
    longest_repeat: int = 0

    def add_chunk(self, v: np.ndarray) -> None:
        v = np.asarray(v, dtype=np.float64)
        ok = np.isfinite(v)
        self.n += len(v)
        self.nulls += int((~ok).sum())
        good = v[ok]
        if len(good):
            self.total += float(good.sum())
            self.total_sq += float((good ** 2).sum())
            self.vmin = min(self.vmin, float(good.min()))
            self.vmax = max(self.vmax, float(good.max()))
            self.samples.append(good.astype(np.float32))

        res = residual(v)
        rok = np.isfinite(res)
        if rok.sum() > 2:
            self.residuals.append(res[rok].astype(np.float32))
            both = rok[:-1] & rok[1:]
            x, y = res[:-1][both], res[1:][both]
            self.lag_n += len(x)
            self.lag_sx += float(x.sum())
            self.lag_sy += float(y.sum())
            self.lag_sxx += float((x * x).sum())
            self.lag_syy += float((y * y).sum())
            self.lag_sxy += float((x * y).sum())
            if len(good) > 2:
                self.within_var_sum += float(np.var(good)) * len(good)
                self.resid_var_sum += float(np.var(res[rok])) * len(good)

        # Stuck-value indicators on consecutive non-missing values.
        if len(good) > 1:
            same = good[1:] == good[:-1]
            self.pairs += len(same)
            self.repeats += int(same.sum())
            edges = np.flatnonzero(np.diff(np.concatenate(([0], same.astype(np.int8), [0]))))
            runs = edges[1::2] - edges[0::2]          # lengths of runs of "same as previous"
            longest = int(runs.max()) + 1 if len(runs) else 1
            self.longest_repeat = max(self.longest_repeat, longest)

    def result(self) -> dict:
        good_n = self.n - self.nulls
        pooled = np.concatenate(self.samples) if self.samples else np.array([], dtype=np.float32)
        pooled64 = pooled.astype(np.float64)
        resid = np.concatenate(self.residuals).astype(np.float64) if self.residuals else np.array([])
        mean = self.total / good_n if good_n else float("nan")
        var = self.total_sq / good_n - mean ** 2 if good_n else float("nan")
        lag1 = float("nan")
        if self.lag_n > 2:
            n = self.lag_n
            cov = self.lag_sxy / n - (self.lag_sx / n) * (self.lag_sy / n)
            vx = self.lag_sxx / n - (self.lag_sx / n) ** 2
            vy = self.lag_syy / n - (self.lag_sy / n) ** 2
            if vx > 0 and vy > 0:
                lag1 = cov / math.sqrt(vx * vy)
        slow = float("nan")
        if self.within_var_sum > 0:
            slow = 1.0 - self.resid_var_sum / self.within_var_sum
        return {
            "n": self.n,
            "null_fraction": _r(self.nulls / self.n if self.n else float("nan")),
            "min": _r(self.vmin if good_n else float("nan"), 8),
            "max": _r(self.vmax if good_n else float("nan"), 8),
            "mean": _r(mean, 8),
            "std": _r(math.sqrt(max(var, 0.0)) if good_n else float("nan")),
            "median": _r(float(np.median(pooled64)) if len(pooled64) else float("nan"), 8),
            "mad_sigma": _r(mad_sigma(pooled64)),
            "noise_mad_sigma": _r(mad_sigma(resid)),
            "noise_lag1": _r(lag1, 4),
            "slow_share": _r(slow, 4),
            "repeat_share": _r(self.repeats / self.pairs if self.pairs else float("nan"), 4),
            "longest_repeat": self.longest_repeat,
        }


@dataclass
class TimingAccumulator:
    """Per-table id and spacing statistics from chunks of consecutive ids."""

    id_pairs: int = 0
    id_step1: int = 0
    time_pairs: int = 0
    backward: int = 0
    duplicate_ts: int = 0
    spacings: list[np.ndarray] = field(default_factory=list)
    gaps: list[tuple[float, int, int, bool]] = field(default_factory=list)  # heap of (length, start, end, exact)
    sampled_seconds: float = 0.0

    def add_chunk(self, ids: np.ndarray, t_us: np.ndarray) -> None:
        if len(ids) < 2:
            return
        d_id = np.diff(ids)
        self.id_pairs += len(d_id)
        self.id_step1 += int((d_id == 1).sum())
        dt = np.diff(t_us)
        self.time_pairs += len(dt)
        self.backward += int((dt < 0).sum())
        self.duplicate_ts += int((dt == 0).sum())
        fwd = dt[dt >= 0] / 1e6
        self.spacings.append(fwd.astype(np.float64))
        self.sampled_seconds += float(max(t_us.max() - t_us.min(), 0)) / 1e6
        for i in np.argsort(dt)[-MAX_GAPS:]:
            if dt[i] > 0:
                self.offer_gap(dt[i] / 1e6, int(t_us[i]), int(t_us[i + 1]), True)

    def offer_gap(self, length_s: float, start_us: int, end_us: int, exact: bool) -> None:
        item = (length_s, start_us, end_us, exact)
        if len(self.gaps) < MAX_GAPS:
            heapq.heappush(self.gaps, item)
        elif item > self.gaps[0]:
            heapq.heapreplace(self.gaps, item)

    def spacing_result(self) -> dict:
        sp = np.concatenate(self.spacings) if self.spacings else np.array([])
        out: dict = {}
        if len(sp):
            q = np.percentile(sp, PERCENTILES)
            out["spacing_s"] = {f"p{p}": _r(float(v)) for p, v in zip(PERCENTILES, q)}
            out["spacing_s"]["max"] = _r(float(sp.max()))
            whole = np.rint(sp)
            out["spacing_share"] = {
                "0": _r(float((whole == 0).mean()), 4), "1": _r(float((whole == 1).mean()), 4),
                "2": _r(float((whole == 2).mean()), 4), "gt2": _r(float((whole > 2).mean()), 4),
            }
            med = float(np.median(sp))
            out["mean_spacing_s"] = _r(float(sp.mean()))
            # Gap counts use the mean spacing when the median is 0 (many shared timestamps).
            ref = med if med > 0 else float(sp.mean())
            out["gaps_gt_5x_median"] = int((sp > 5 * ref).sum()) if ref > 0 else 0
            mean = float(sp.mean())
            # A step counts as long (a missing row) when it is longer than 1.5 mean spacings AND longer
            # than the largest normal step. With whole-second timestamps a 1.1 s period gives normal
            # steps of 1 s and 2 s, so 1.5 periods alone would count healthy 2 s steps.
            thr = max(1.5 * mean, math.ceil(mean) + 0.5)
            out["long_step_share"] = _r(float((sp > thr).mean()), 4) if mean > 0 else None
        else:
            out["spacing_s"] = {f"p{p}": None for p in PERCENTILES} | {"max": None}
            out["spacing_share"] = {"0": None, "1": None, "2": None, "gt2": None}
            out["mean_spacing_s"] = None
            out["gaps_gt_5x_median"] = 0
            out["long_step_share"] = None
        out["sampled_hours"] = _r(self.sampled_seconds / 3600.0, 4)
        out["id_step_1_share"] = _r(self.id_step1 / self.id_pairs if self.id_pairs else float("nan"), 6)
        out["backward_ts_share"] = _r(self.backward / self.time_pairs if self.time_pairs else float("nan"), 6)
        out["duplicate_ts_share"] = _r(self.duplicate_ts / self.time_pairs if self.time_pairs else float("nan"), 6)
        return out


def correlation_pairs(frame: pd.DataFrame, top: int = 30) -> tuple[list[dict], list[dict]]:
    """Top channel pairs by |r| of first differences and of levels.

    ``frame`` has one column per channel key on a common grid, and a 'window'
    column so first differences are taken within each window only.
    """
    cols = [c for c in frame.columns if c != "window"]
    levels = frame[cols]
    diffs = frame.groupby("window")[cols].diff()

    def pairs(df: pd.DataFrame) -> list[dict]:
        corr = df.corr(min_periods=30)
        out = []
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                r = corr.loc[a, b]
                if np.isfinite(r):
                    out.append({"a": a, "b": b, "r": _r(float(r), 4)})
        out.sort(key=lambda d: (-abs(d["r"]), d["a"], d["b"]))
        return out[:top]

    return pairs(diffs), pairs(levels)
