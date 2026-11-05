"""Summary builder: the state store as a short, factual text for the LLM.

The summary is the ONLY thing the LLM sees. It states facts the detectors
concluded and nothing else: no causes, no advice. Format::

    Status at 2026-01-01 12:00 UTC. 5 channels monitored, 2 abnormal.
    - polarization: drifting down since 10:08 (now 3.1 below normal, 20.7 sigma). Normal is 41.0.
    - rf_power: shifted up since 12:00 (now 2.4 above normal, 6.0 sigma). Normal is 100.0.
    Recent spikes (last 60 min): beam_current at 11:31 (+9.0 sigma).
    Feeds: all reporting.
    Normal: beam_current, target_temp, nmr_signal.

Numbers are rounded to a few significant digits. Times are shown as HH:MM in
UTC when the run is within one day of "now", otherwise with the date.
"""

from __future__ import annotations

from datetime import datetime

from .detectors import Condition
from .state import ChannelState, StateStore

RECENT_SPIKE_S = 3600.0

_PHRASE = {
    "drift_up": "running high",
    "drift_down": "running low",
    "stuck": "stuck at a constant value",
    "noisy": "noisier than normal",
    "missing": "not reporting values (missing readings)",
    "decoupled": "no longer tracking",
}
_ORDER = ["missing", "stuck", "decoupled", "drift_up", "drift_down", "noisy"]


def fmt_num(x: float | None) -> str:
    if x is None:
        return "unknown"
    ax = abs(x)
    if ax != 0 and (ax >= 1e5 or ax < 1e-3):
        return f"{x:.3g}"
    if ax >= 100:
        return f"{x:.1f}"
    return f"{x:.3g}"


def fmt_time(t: datetime | None, now: datetime | None) -> str:
    if t is None:
        return "unknown time"
    if now is not None and abs((now - t).total_seconds()) < 86400 and now.date() == t.date():
        return t.strftime("%H:%M")
    return t.strftime("%Y-%m-%d %H:%M")


def _unit(ch: ChannelState) -> str:
    return f" {ch.unit}" if ch.unit else ""


def describe(ch: ChannelState, cond: Condition, now: datetime | None) -> str:
    since = fmt_time(cond.since, now)
    if cond.kind in ("drift_up", "drift_down"):
        dev = cond.magnitude if cond.magnitude is not None else 0.0
        word = "above" if dev >= 0 else "below"
        sig = abs(cond.magnitude_sigma or 0.0)
        return (f"{_PHRASE[cond.kind]} since {since} (about {fmt_num(abs(dev))}{_unit(ch)} {word} normal, "
                f"{sig:.1f} sigma)")
    if cond.kind == "stuck":
        return f"{_PHRASE['stuck']} of {fmt_num(cond.detail.get('value'))}{_unit(ch)} since {since}"
    if cond.kind == "noisy":
        ratio = cond.detail.get("noise_ratio")
        return f"{_PHRASE['noisy']} since {since} (noise about {fmt_num(ratio)} times normal)"
    if cond.kind == "missing":
        return f"{_PHRASE['missing']} since {since}"
    if cond.kind == "decoupled":
        return f"{_PHRASE['decoupled']} {cond.detail.get('source', 'its source')} since {since}"
    return f"{cond.kind} since {since}"


def spike_groups(store: StateStore, within_s: float = RECENT_SPIKE_S) -> list[dict]:
    """Recent spikes grouped by channel, in order of the latest spike on each channel."""
    by: dict[str, list[Condition]] = {}
    for s in store.recent_spikes(within_s):
        by.setdefault(s.subject, []).append(s)
    out = []
    for key, sp in sorted(by.items(), key=lambda kv: kv[1][-1].detected_at):
        name = store.channels[key].display_name
        peak = max(sp, key=lambda c: abs(c.magnitude_sigma or 0))
        t = fmt_time(sp[-1].detected_at, store.now)
        sig = f"{(peak.magnitude_sigma or 0):+.1f} sigma"
        if len(sp) == 1:
            text = f"{name} at {t} ({sig})"
        else:
            text = f"{name}: {len(sp)} spikes, the last at {t} (up to {sig})"
        out.append({"name": name, "time": t, "sigma": sig, "count": len(sp), "text": text})
    return out


def build_summary(store: StateStore, recent_spike_s: float = RECENT_SPIKE_S) -> str:
    now = store.now
    if now is None:
        # Nothing has arrived: do not list channels as "Normal" or feeds as "reporting".
        return f"Status: no data yet. {len(store.channels)} channels monitored; no feed has sent any rows yet."
    head = f"Status at {now.strftime('%Y-%m-%d %H:%M')} UTC."
    abnormal = store.abnormal_channels()
    # Feeds that are out, or have never sent a row: their channels have no current data, so they
    # are not listed as "Normal" (a stale "normal" is misleading; found with the hard question set).
    stale_feeds = {f.name: f for f in store.feeds.values() if f.outage is not None or f.last_row is None}
    stale = {name: [c.display_name for c in store.channels.values() if c.feed == name and not c.active]
             for name in stale_feeds}
    n_stale = sum(len(v) for v in stale.values())
    extra = f", {n_stale} without current data" if n_stale else ""
    lines = [f"{head} {len(store.channels)} channels monitored, {len(abnormal)} abnormal{extra}."]
    for ch in sorted(abnormal, key=lambda c: min(x.since for x in c.active.values())):
        conds = sorted(ch.active.values(), key=lambda c: _ORDER.index(c.kind) if c.kind in _ORDER else 99)
        parts = "; ".join(describe(ch, c, now) for c in conds)
        lines.append(f"- {ch.display_name}: {parts}. Normal is {fmt_num(ch.cal.target)}{_unit(ch)}.")
    groups = spike_groups(store, recent_spike_s)
    if groups:
        items = ", ".join(g["text"] for g in groups[:6])
        more = f", and spikes on {len(groups) - 6} more channels" if len(groups) > 6 else ""
        lines.append(f"Recent spikes (last {int(recent_spike_s // 60)} min): {items}{more}.")
    if store.feeds:
        # A feed with an outage, or one that has never sent a row with a monitored value (seen on
        # real ptdb data: Labjack sent nothing on 2026-08-17), is not "reporting".
        parts = [f"{f.name} has sent no rows since {fmt_time(f.outage.since, now)}"
                 for f in store.feeds.values() if f.outage is not None]
        parts += [f"{f.name} has sent no rows yet" for f in store.feeds.values()
                  if f.outage is None and f.last_row is None]
        lines.append("Feeds: " + ("; ".join(parts) if parts else "all reporting") + ".")
    for name, chans in stale.items():
        if chans:
            f = stale_feeds[name]
            why = (f"{name} has sent no rows since {fmt_time(f.outage.since, now)}" if f.outage is not None
                   else f"{name} has sent no rows yet")
            lines.append(f"No current data ({why}): " + ", ".join(chans) + ".")
    stale_names = {n for v in stale.values() for n in v}
    normal = [c.display_name for c in store.channels.values() if not c.active and c.display_name not in stale_names]
    if normal:
        # Every normal channel is named, so a reader (or the LLM) never has to guess
        # whether an unlisted name is a real, healthy channel.
        lines.append("Normal: " + ", ".join(normal) + ".")
    return "\n".join(lines)
