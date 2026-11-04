"""Inspection plots for one run.

Two kinds of figure, both saved as PNG (no interactive windows):

* ``overview.png``: one subplot per channel over time. The healthy baseline is
  a dashed line with a shaded +/- 3 sigma band. Each labeled anomaly window is
  shaded in a color for its type and annotated with the type name. A small
  triangle at the top edge marks ``first_detectable``. Missing data shows as
  gaps in the line.
* ``decoupling_<channel>.png``: for every decoupling label, a scatter plot of
  the coupled pair (source channel against the coupled channel), with healthy
  samples and samples inside the decoupling window in different colors. While
  coupled, the points lie along a line of slope ``gain``; once decoupled, they
  stop following it.

Plotting reads labels on purpose: plots are for people inspecting ground truth.
Detectors must still use ``replay``, which never exposes labels.

Usage::

    from sim.plotting import plot_run
    paths = plot_run("out/mixed_demo")        # a run folder written by make_run.py
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # file output only, never open a window

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from .io import read_data_csv, read_labels  # noqa: E402
from .scenario import ScenarioConfig, parse_scenario  # noqa: E402

# One fixed color per anomaly type so every plot uses the same legend.
TYPE_COLORS = {
    "spike": "#d62728",
    "spike_burst": "#ff7f0e",
    "step_shift": "#9467bd",
    "linear_drift": "#1f77b4",
    "exponential_drift": "#17becf",
    "noise_growth": "#bcbd22",
    "transient_decay": "#e377c2",
    "dropout": "#7f7f7f",
    "stuck_sensor": "#8c564b",
    "decoupling": "#2ca02c",
    "feed_outage": "#000000",
}
DEFAULT_TYPE_COLOR = "#444444"

# A window shorter than this share of the run is drawn as a thin line instead
# of a shaded span (a 1 s spike would be invisible as a span over 6 hours).
POINT_EVENT_FRACTION = 0.002

# A time step longer than this many median steps is drawn as a gap even if the
# rows on both sides have values (rows that are absent, not NaN).
GAP_FACTOR = 3.0


@dataclass
class RunData:
    """Everything the plots need, read from one run folder."""

    name: str
    config: ScenarioConfig
    times: dict[str, np.ndarray]    # channel -> datetime64[us] sample times
    values: dict[str, np.ndarray]   # channel -> float values (NaN = missing reading)
    labels: list[dict[str, Any]]
    start: np.datetime64
    end: np.datetime64


def load_run(run_dir: str | Path) -> RunData:
    """Read data.csv, labels.json and config_used.yaml from a run folder."""
    run_dir = Path(run_dir)
    raw = yaml.safe_load((run_dir / "config_used.yaml").read_text())
    cfg = parse_scenario(raw)
    names = [ch.name for ch in cfg.channels]
    t_lists: dict[str, list[datetime]] = {n: [] for n in names}
    v_lists: dict[str, list[float]] = {n: [] for n in names}
    for ts, event in read_data_csv(run_dir / "data.csv"):
        for name, value in event.items():
            t_lists[name].append(ts)
            v_lists[name].append(value)
    times = {n: np.array(t_lists[n], dtype="datetime64[us]") for n in names}
    values = {n: np.array(v_lists[n], dtype=float) for n in names}
    start = np.datetime64(cfg.start_time, "us")
    end = start + np.timedelta64(int(round(cfg.duration_seconds * 1e6)), "us")
    return RunData(run_dir.name, cfg, times, values, read_labels(run_dir / "labels.json"),
                   start, end)


def with_gaps(t: np.ndarray, v: np.ndarray, factor: float = GAP_FACTOR) -> tuple[np.ndarray, np.ndarray]:
    """Insert a NaN wherever the time step is longer than ``factor`` median steps.

    matplotlib breaks a line at NaN, so absent rows show up as gaps. NaN
    readings already in ``v`` (dropouts) are left alone and show as gaps too.
    """
    if len(t) < 3:
        return t, v
    step = np.diff(t.astype("int64"))
    median = np.median(step)
    breaks = np.nonzero(step > factor * median)[0]
    if len(breaks) == 0:
        return t, v
    t_out = np.insert(t, breaks + 1, t[breaks] + np.timedelta64(int(median), "us"))
    v_out = np.insert(v, breaks + 1, np.nan)
    return t_out, v_out


def _label_window(label: dict[str, Any], run_end: np.datetime64) -> tuple[np.datetime64, np.datetime64]:
    start = np.datetime64(label["start"], "us")
    end = np.datetime64(label["end"], "us") if label["end"] else run_end
    return start, end


def _draw_labels(ax, labels: list[dict[str, Any]], run: RunData) -> None:
    span_min = (run.end - run.start) * POINT_EVENT_FRACTION
    top = ax.get_xaxis_transform()  # x in data units, y in axes units (0 bottom, 1 top)
    for label in labels:
        color = TYPE_COLORS.get(label["type"], DEFAULT_TYPE_COLOR)
        start, end = _label_window(label, run.end)
        if end - start < span_min:
            ax.axvline(start, color=color, lw=1.2, alpha=0.8)
        else:
            ax.axvspan(start, end, color=color, alpha=0.15, lw=0)
            ax.axvline(start, color=color, lw=0.8, alpha=0.6)
        ax.text(start, 0.97, " " + label["type"], transform=top, color=color,
                fontsize=7, va="top", ha="left", clip_on=True)
        if label["first_detectable"]:
            fd = np.datetime64(label["first_detectable"], "us")
            ax.plot([fd], [1.0], marker="v", color=color, markersize=6,
                    transform=top, clip_on=False)


def plot_overview(run: RunData, path: Path, title_prefix: str = "") -> Path:
    """One subplot per channel with baseline, 3 sigma band, and labeled windows."""
    channels = list(run.config.channels)
    fig, axes = plt.subplots(len(channels), 1, figsize=(13, 2.2 * len(channels) + 0.8),
                             sharex=True, squeeze=False)
    for ax, ch in zip(axes[:, 0], channels):
        t, v = with_gaps(run.times[ch.name], run.values[ch.name])
        ax.axhspan(ch.baseline - 3 * ch.sigma, ch.baseline + 3 * ch.sigma,
                   color="#1f77b4", alpha=0.08, lw=0, label="baseline +/- 3 sigma")
        ax.axhline(ch.baseline, color="#555555", lw=1, ls="--", label="baseline")
        ax.plot(t, v, color="#222222", lw=0.5)
        ax.set_ylabel(ch.name, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_xlim(run.start, run.end)
        _draw_labels(ax, [lb for lb in run.labels if lb["channel"] == ch.name], run)
        n_missing = int(np.isnan(run.values[ch.name]).sum())
        if n_missing:
            ax.text(0.995, 0.03, f"{n_missing} missing readings", transform=ax.transAxes,
                    fontsize=7, ha="right", va="bottom", color="#7f7f7f")

    bottom = axes[-1, 0]
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    bottom.set_xlabel(f"time (UTC), starting {run.config.start_time.isoformat()}", fontsize=9)
    axes[0, 0].legend(loc="upper right", fontsize=7, framealpha=0.8)
    fig.suptitle(f"{title_prefix}Run {run.name} (seed {run.config.seed}): "
                 f"{len(run.labels)} labeled anomalies", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _value_at(t_src: np.ndarray, v_src: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Source value at each time in ``t`` (latest source sample at or before it)."""
    idx = np.searchsorted(t_src, t, side="right") - 1
    out = np.full(len(t), np.nan)
    ok = idx >= 0
    out[ok] = v_src[idx[ok]]
    return out


def plot_decoupling(run: RunData, label: dict[str, Any], path: Path, title_prefix: str = "") -> Path:
    """Scatter of the coupled pair, healthy period against the decoupling window."""
    ch = run.config.channel(label["channel"])
    src = run.config.channel(ch.couples_to)
    t = run.times[ch.name]
    y = run.values[ch.name]
    x = _value_at(run.times[src.name], run.values[src.name], t)
    start, end = _label_window(label, run.end)
    inside = (t >= start) & (t <= end)
    before = t < start

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(x[before], y[before], s=6, alpha=0.5, color="#1f77b4", label="healthy (before window)")
    ax.scatter(x[inside], y[inside], s=6, alpha=0.6, color=TYPE_COLORS["decoupling"],
               label="decoupling window")
    xs = np.array([np.nanmin(x), np.nanmax(x)])
    ax.plot(xs, ch.baseline + ch.gain * (xs - src.baseline), color="#555555", ls="--", lw=1,
            label=f"healthy coupling (gain {ch.gain:g})")
    ax.set_xlabel(src.name)
    ax.set_ylabel(ch.name)
    ax.legend(fontsize=8)
    ax.set_title(f"{title_prefix}{ch.name} against {src.name}, decoupling from "
                 f"{label['start'][11:19]}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_layout_run(run_dir: str | Path, out_dir: str | Path | None = None) -> list[Path]:
    """One SIMULATED overview figure per feed of a wide-feed run (Phase R2).

    Each feed figure has one panel per channel, using the real column names.
    A feed_outage label is shaded on every panel of its feed. Titles say
    SIMULATED because the names are real but the data is not.
    """
    from .layout_io import read_feeds, read_layout_labels, read_meta

    run_dir = Path(run_dir)
    meta = read_meta(run_dir)
    if meta.get("synthetic") is not True:
        raise ValueError(f"{run_dir}: meta.json must say synthetic: true")
    feeds = read_feeds(run_dir)
    labels = read_layout_labels(run_dir)
    models = meta.get("channel_models", {})
    out = Path(out_dir) if out_dir is not None else run_dir / "plots"
    out.mkdir(parents=True, exist_ok=True)

    def naive(iso: str | None) -> str | None:
        return None if iso is None else iso.replace("+00:00", "")

    start = np.datetime64(naive(meta["start"]), "us")
    end = np.datetime64(naive(meta["end"]), "us")
    stub = RunData(run_dir.name, None, {}, {}, [], start, end)  # only start and end are used below

    paths = []
    for name, feed in feeds.items():
        fig, axes = plt.subplots(len(feed.columns), 1, figsize=(13, 1.7 * len(feed.columns) + 0.9),
                                 sharex=True, squeeze=False)
        t_all = (feed.t_us).astype("datetime64[us]")
        for ax, col in zip(axes[:, 0], feed.columns):
            key = f"{name}.{col}"
            model = models.get(key, {})
            t, v = with_gaps(t_all, feed.values[col])
            if "baseline" in model and "sigma" in model:
                b, s = model["baseline"], model["sigma"]
                ax.axhspan(b - 3 * s, b + 3 * s, color="#1f77b4", alpha=0.08, lw=0)
                ax.axhline(b, color="#555555", lw=1, ls="--")
            ax.plot(t, v, color="#222222", lw=0.5)
            ax.set_ylabel(col, fontsize=7, rotation=0, ha="right", va="center")
            ax.tick_params(labelsize=7)
            ax.set_xlim(start, end)
            mine = [{**lb, "start": naive(lb["start"]), "end": naive(lb["end"]),
                     "first_detectable": naive(lb["first_detectable"])}
                    for lb in labels if lb["channel"] in (key, name)]
            _draw_labels(ax, mine, stub)
        axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axes[-1, 0].set_xlabel(f"time (UTC), starting {meta['start']}", fontsize=9)
        fig.suptitle(f"SIMULATED data with real ptdb names: {name} "
                     f"(seed {meta['seed']}, {feed.n_rows} rows)", fontsize=11)
        fig.tight_layout()
        path = out / f"feed_{name}.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_run(run_dir: str | Path, out_dir: str | Path | None = None,
             scatter: bool = True, title_prefix: str = "") -> list[Path]:
    """Make every plot for one run folder. Returns the PNG paths written.

    PNGs go to ``<run_dir>/plots/`` unless ``out_dir`` is given.
    """
    run = load_run(run_dir)
    out = Path(out_dir) if out_dir is not None else Path(run_dir) / "plots"
    out.mkdir(parents=True, exist_ok=True)
    paths = [plot_overview(run, out / "overview.png", title_prefix)]
    if scatter:
        for label in run.labels:
            if label["type"] == "decoupling":
                name = f"decoupling_{label['channel']}_{label['id']}.png"
                paths.append(plot_decoupling(run, label, out / name, title_prefix))
    return paths
