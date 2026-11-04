"""Run the monitor (detectors -> state store -> summary) over a stream.

Sources (pick one):
    --run DIR          a simulator run folder (data.csv); healthy calibration from --healthy
    --layout-run DIR   a wide-feed run folder (SIMULATED, real names)
    --sqlite PATH      a ptdb-shaped SQLite fixture, read with the live reader
    --mysql            the real ptdb, read-only, credentials from PTDB_* environment variables
                       (run this yourself, on Shannon; never from an automated session)

Calibration (pick one):
    --healthy SOURCE        a healthy scenario YAML or run folder with the same channels
    --calibrate-minutes N   use the first N minutes of the stream itself (default 60)

Output (default out/monitor/<name>/):
    state.json        the state store after the last event
    summary.txt       the summary after the last event
    snapshots.jsonl   {"time", "summary", "state"} every --every-minutes of stream time
    alarms.jsonl      every condition raised (spikes too), with when it ended; one per line
    coverage.txt      which channels are monitored, which are not and why, and each feed's outage limit

Summarize alarms.jsonl with scripts/summarize_alarms.py.

With --follow (sqlite or mysql) the monitor keeps polling every --poll-seconds and
rewrites state.json and summary.txt after each poll. The Discord bot reads those files.

Examples:
    python scripts/run_monitor.py --run out/mixed_demo --healthy configs/healthy.yaml
    python scripts/run_monitor.py --layout-run out/layout/seed1_mixed --calibrate-minutes 30
    python scripts/run_monitor.py --mysql --since 2025-11-03T00:00:00 --until-hours 168 --calibrate-minutes 1440
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monitor.engine import Coupling, Engine  # noqa: E402
from monitor.ingest import PtdbReader  # noqa: E402
from monitor.scoring import couplings_of  # noqa: E402
from monitor.summary import build_summary  # noqa: E402
from sim.layout_scenario import replay_layout  # noqa: E402
from sim.registry import load_registry  # noqa: E402
from sim.stream import replay  # noqa: E402


def _limit_hours(events, hours: float):
    """Stop after ``hours`` of data from the first event (reading stops there too)."""
    first = None
    for t, ev in events:
        if first is None:
            first = t
        if (t - first).total_seconds() > hours * 3600:
            return
        yield t, ev


def _split_calibration(events, minutes: float):
    """Take the first ``minutes`` of events for calibration; return (healthy list, rest iterator)."""
    events = iter(events)
    first = next(events, None)
    if first is None:
        return [], iter(())
    healthy = [first]
    for t, ev in events:
        if (t - first[0]).total_seconds() > minutes * 60:
            return healthy, itertools.chain([(t, ev)], events)
        healthy.append((t, ev))
    return healthy, iter(())


def coverage_text(engine: Engine) -> str:
    """Monitored and unmonitored channels, and feed outage limits (for people, not the model)."""
    missing = engine.coverage()
    lines = [f"Monitored channels: {len(engine.monitors)}", f"Not monitored: {len(missing)}"]
    lines += [f"  {k}: {why}" for k, why in missing]
    for f, w in engine.feed_watch.items():
        lines.append(f"Feed {f}: period {w.period_s:.3g} s, outage after {w.limit():.3g} s without a row")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run")
    src.add_argument("--layout-run")
    src.add_argument("--sqlite")
    src.add_argument("--mysql", action="store_true")
    p.add_argument("--healthy")
    p.add_argument("--calibrate-minutes", type=float, default=60.0)
    p.add_argument("--couplings", help="YAML with a 'couplings:' list of {channel, source, gain}")
    p.add_argument("--layout", default="configs/ptdb_layout.yaml")
    p.add_argument("--every-minutes", type=float, default=10.0)
    p.add_argument("--follow", action="store_true")
    p.add_argument("--poll-seconds", type=float, default=30.0)
    p.add_argument("--plain", action="store_true", help="plain Phase 5 thresholds (no adaptation)")
    p.add_argument("--out", default="out/monitor")
    p.add_argument("--name")
    p.add_argument("--since", help="sqlite or mysql: start reading at this UTC time (ISO), not at the first row")
    p.add_argument("--until-hours", type=float,
                   help="sqlite or mysql: stop after this many hours of data from --since (or the first row)")
    args = p.parse_args(argv)

    couplings: list[Coupling] = []
    display, feeds, reader = {}, {}, None
    if args.run:
        raw = yaml.safe_load((Path(args.run) / "config_used.yaml").read_text())
        couplings = couplings_of(raw)
        events = replay(args.run)
        name = args.name or Path(args.run).name
    else:
        reg = load_registry(args.layout)
        display = {c.key: (c.display_name, c.unit) for c in reg.included_channels()}
        feeds = {t.name: [c.key for c in t.channels] for t in reg.included_tables()}
        if args.layout_run:
            events = replay_layout(args.layout_run)
            name = args.name or Path(args.layout_run).name
        else:
            from sim.profiler.adapters import MySQLAdapter, SQLiteAdapter
            adapter = SQLiteAdapter(args.sqlite) if args.sqlite else MySQLAdapter()
            reader = PtdbReader(adapter, reg)
            if args.since:
                since = datetime.fromisoformat(args.since)
                reader.start_at_time(since if since.tzinfo else since.replace(tzinfo=timezone.utc))
            events = reader.read_all()
            if args.until_hours:
                events = _limit_hours(events, args.until_hours)
            name = args.name or (Path(args.sqlite).parent.name if args.sqlite else "ptdb")
    if args.couplings:
        listed = (yaml.safe_load(Path(args.couplings).read_text()) or {}).get("couplings") or []
        couplings += [Coupling(c["channel"], c["source"], float(c.get("gain", 0.15))) for c in listed]

    if args.healthy:
        hp = Path(args.healthy)
        healthy = list(replay(hp) if args.run else replay_layout(hp))
        rest = events
    else:
        healthy, rest = _split_calibration(events, args.calibrate_minutes)
    if not healthy:
        print("No data to calibrate from.", file=sys.stderr)
        return 2
    engine = Engine.from_healthy(healthy, adaptive=not args.plain, couplings=couplings, feeds=feeds,
                                 display=display)
    if not args.healthy:
        for t, ev in healthy:          # the calibration window is part of the stream too
            engine.step(t, ev)

    out = Path(args.out) / name
    out.mkdir(parents=True, exist_ok=True)
    snaps = open(out / "snapshots.jsonl", "w", encoding="utf-8")
    every = args.every_minutes * 60
    next_snap = None

    def write_now():
        engine.store.save(out / "state.json")
        (out / "summary.txt").write_text(build_summary(engine.store) + "\n", encoding="utf-8")
        with open(out / "alarms.jsonl", "w", encoding="utf-8") as fh:
            for c in engine.raised:
                fh.write(json.dumps(c.to_dict()) + "\n")
        (out / "coverage.txt").write_text(coverage_text(engine), encoding="utf-8")

    def consume(evs):
        nonlocal next_snap
        for t, ev in evs:
            engine.step(t, ev)
            if next_snap is None:
                next_snap = t.timestamp() + every
            if t.timestamp() >= next_snap:
                snaps.write(json.dumps({"time": t.isoformat(), "summary": build_summary(engine.store),
                                        "state": engine.store.to_dict()}) + "\n")
                next_snap += every

    consume(rest)
    write_now()
    print(build_summary(engine.store))
    print()
    print(coverage_text(engine), end="")
    if args.follow and reader is not None:
        print(f"\nFollowing: polling every {args.poll_seconds:g} s (Ctrl+C to stop)")
        try:
            while True:
                time.sleep(args.poll_seconds)
                consume(reader.poll())
                write_now()
        except KeyboardInterrupt:
            pass
    snaps.close()
    print(f"\nWrote {out / 'state.json'}, {out / 'summary.txt'}, {out / 'snapshots.jsonl'}")
    if reader is not None:
        s = reader.stats
        print(f"Reader: {s.rows} rows, {s.emitted} events, {s.duplicates} duplicates dropped, "
              f"{s.too_late} too late, {s.nulls} NULL values")
    return 0


if __name__ == "__main__":
    sys.exit(main())
