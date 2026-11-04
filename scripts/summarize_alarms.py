"""Digest of a monitor run's alarms: counts, durations and rates, no raw readings.

    python scripts/summarize_alarms.py out/monitor/real_nov2025 --test-start 2025-11-04T00:00:00

Reads <dir>/alarms.jsonl (every condition raised, from run_monitor.py), <dir>/state.json (for the
end time) and <dir>/coverage.txt. Alarms raised before --test-start (the calibration window, which
the monitor also steps through) are counted separately, so they do not inflate the test rates.

Prints:
  * coverage (monitored and not monitored channels, feed outage limits)
  * per condition kind: count in the test window, and alarms per hour
  * per channel and kind: count, first and last time, active time, median and longest duration
  * spikes per channel: count, rate per hour, share upward, median and largest size in sigma
  * the longest feed outages
  * alarms per day per kind
The output contains times, counts and sizes of alarms only. Review it before sharing.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def _t(text: str | None) -> datetime | None:
    if not text:
        return None
    t = datetime.fromisoformat(text)
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _dur(seconds: float) -> str:
    if seconds < 120:
        return f"{seconds:.0f} s"
    if seconds < 7200:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


EPISODE_GAP_S = 1800.0   # a re-raise this soon after the previous one ended continues the same episode
TOGETHER_S = 2.0         # conditions of one kind starting this close together count as one group


def _episodes(rows: list[dict], end: datetime | None) -> int:
    """Number of episodes: spikes count one each; other kinds merge re-raises within EPISODE_GAP_S."""
    if rows and rows[0]["kind"] == "spike":
        return len(rows)
    n, last_end = 0, None
    for r in sorted(rows, key=lambda r: r["_since"]):
        if last_end is None or (r["_since"] - last_end).total_seconds() > EPISODE_GAP_S:
            n += 1
        e = r["_end"] or end or r["_since"]
        last_end = e if last_end is None else max(last_end, e)
    return n


def _together(alarms: list[dict], names: dict[str, str]) -> list[tuple[tuple[str, tuple[str, ...]], int]]:
    """Sets of two or more channels that raise the same kind at (nearly) the same time, and how often."""
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for a in alarms:
        if a["kind"] not in ("spike", "feed_outage") and a["_since"]:
            by_kind[a["kind"]].append(a)
    counts: Counter = Counter()
    for kind, rows in by_kind.items():
        rows.sort(key=lambda r: r["_since"])
        i = 0
        while i < len(rows):
            j = i + 1
            while j < len(rows) and (rows[j]["_since"] - rows[i]["_since"]).total_seconds() <= TOGETHER_S:
                j += 1
            members = tuple(sorted({names.get(r["subject"], r["subject"]) for r in rows[i:j]}))
            if len(members) > 1:
                counts[(kind, members)] += 1
            i = j
    return [(k, n) for k, n in counts.most_common() if n >= 2]


def digest(folder: Path, test_start: datetime | None) -> str:
    alarms = [json.loads(line) for line in (folder / "alarms.jsonl").read_text(encoding="utf-8").splitlines()
              if line.strip()]
    state = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    end = _t(state.get("now"))
    names = {k: c.get("display_name") or k for k, c in state.get("channels", {}).items()}
    out: list[str] = []
    add = out.append

    cov = folder / "coverage.txt"
    if cov.exists():
        add("== Coverage")
        add(cov.read_text(encoding="utf-8").rstrip())
        add("")

    for a in alarms:
        a["_since"], a["_det"], a["_end"] = _t(a["since"]), _t(a["detected_at"]), _t(a.get("ended_at"))
    first = min((a["_det"] for a in alarms), default=None)
    start = test_start or first
    test = [a for a in alarms if start is None or a["_det"] >= start]
    cal = len(alarms) - len(test)
    hours = (end - start).total_seconds() / 3600 if (end and start) else float("nan")
    add(f"== Window: test from {start.isoformat() if start else '?'} to {end.isoformat() if end else '?'}"
        f" ({hours:.1f} h); {cal} alarms in the calibration window not counted below")
    add("")

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for a in test:
        groups[(a["subject"], a["kind"])].append(a)
    episodes = {key: _episodes(rows, end) for key, rows in groups.items()}
    per_kind_eps = Counter()
    for (subj, kind), n in episodes.items():
        per_kind_eps[kind] += n

    add(f"== Per kind (test window). Episodes merge re-raises of the same channel and kind that start "
        f"within {EPISODE_GAP_S / 60:.0f} min of the previous one ending; spikes are not merged.")
    for kind, n in Counter(a["kind"] for a in test).most_common():
        if hours == hours and hours > 0:
            add(f"  {kind:12s} {n:6d} raised ({n / hours:7.3f}/h)   {per_kind_eps[kind]:6d} episodes "
                f"({per_kind_eps[kind] / hours:7.3f}/h)")
        else:
            add(f"  {kind} {n} raised, {per_kind_eps[kind]} episodes")
    add("")

    together = _together(test, names)
    if together:
        add(f"== Channels that alarm together (same kind, starting within {TOGETHER_S:.0f} s of each other)")
        for (kind, members), n in together:
            add(f"  {kind}: {' + '.join(members)}: together {n} times")
        add("")

    add("== Per channel and kind (test window; durations exclude spikes)")
    for (subj, kind), rows in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        line = f"  {names.get(subj, subj)} [{subj}] {kind}: {len(rows)}"
        if kind != "spike":
            line += f" in {episodes[(subj, kind)]} episodes"
        line += f", first {rows[0]['_det']:%Y-%m-%d %H:%M}, last {rows[-1]['_det']:%Y-%m-%d %H:%M}"
        if kind != "spike":
            durs = [((r["_end"] or end) - r["_since"]).total_seconds() for r in rows if r["_since"] and (r["_end"] or end)]
            still = sum(1 for r in rows if r["_end"] is None)
            if durs:
                line += f"; active {_dur(sum(durs))} in total, median {_dur(statistics.median(durs))}, " \
                        f"longest {_dur(max(durs))}"
            if still:
                line += f"; {still} still active at the end"
        add(line)
    add("")

    spikes = [a for a in test if a["kind"] == "spike"]
    if spikes:
        add("== Spikes per channel (test window)")
        per: dict[str, list[dict]] = defaultdict(list)
        for a in spikes:
            per[a["subject"]].append(a)
        for subj, rows in sorted(per.items(), key=lambda kv: -len(kv[1])):
            sig = [abs(r["magnitude_sigma"]) for r in rows if r.get("magnitude_sigma") is not None]
            up = sum(1 for r in rows if (r.get("magnitude") or 0) > 0)
            mags = sorted(abs(r["magnitude"]) for r in rows if r.get("magnitude") is not None)
            rate = f"{len(rows) / hours:.2f}/h" if hours == hours and hours > 0 else "?"
            line = f"  {names.get(subj, subj)}: {len(rows)} ({rate}), {100 * up / len(rows):.0f}% upward"
            if sig:
                line += f", size median {statistics.median(sig):.1f} sigma, largest {max(sig):.1f} sigma"
            if mags:
                line += f" (|change| {mags[0]:.3g} to {mags[-1]:.3g}, median {statistics.median(mags):.3g})"
            add(line)
        add("")

    outages = [a for a in test if a["kind"] == "feed_outage"]
    if outages:
        add("== Longest feed outages (test window)")
        rows = sorted(outages, key=lambda r: -((r["_end"] or end) - r["_since"]).total_seconds())
        for r in rows[:15]:
            length = ((r["_end"] or end) - r["_since"]).total_seconds()
            add(f"  {r['subject']}: from {r['_since']:%Y-%m-%d %H:%M:%S}, {_dur(length)}"
                + (" (still out at the end)" if r["_end"] is None else ""))
        add("")

    add("== Per day and kind (test window)")
    days: dict[str, Counter] = defaultdict(Counter)
    for a in test:
        days[f"{a['_det']:%Y-%m-%d}"][a["kind"]] += 1
    for day in sorted(days):
        add(f"  {day}: " + ", ".join(f"{k} {n}" for k, n in days[day].most_common()))
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", help="a run_monitor.py output folder (with alarms.jsonl and state.json)")
    p.add_argument("--test-start", help="UTC time (ISO) where the test window starts (end of calibration)")
    p.add_argument("--out", help="also write the digest to this file")
    args = p.parse_args(argv)
    folder = Path(args.folder)
    if not (folder / "alarms.jsonl").exists():
        print(f"No alarms.jsonl in {folder}. Run scripts/run_monitor.py first.", file=sys.stderr)
        return 2
    text = digest(folder, _t(args.test_start))
    print(text, end="")
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
