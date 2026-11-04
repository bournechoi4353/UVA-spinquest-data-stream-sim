"""Read-only, aggregates-only profiler for ptdb (Phase R3).

Run against the real database ONLY by hand, on Shannon, by the person
responsible for the data. Never from an automated session.

    # dry run: prints every planned statement, connects to nothing
    python scripts/profile_db.py --dry-run

    # real run: credentials from the environment only
    export PTDB_HOST=... PTDB_PORT=3306 PTDB_USER=... PTDB_PASSWORD=... PTDB_DATABASE=ptdb
    python scripts/profile_db.py --pause-seconds 0.5

    # against a local SQLite fixture (tests and the R4 comparison)
    python scripts/profile_db.py --sqlite out/layout/seed1_mixed/fixture.sqlite --out out/profile_fixture

Output: <out>/ptdb_profile.json and ptdb_profile.md (default out/profile/, gitignored).
Both files contain aggregates only. Review them before sharing with anyone, including in a chat.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.profiler import (BANNER, DryRunAdapter, MySQLAdapter, Profiler, ProfileSettings,  # noqa: E402
                          SQLiteAdapter, write_report)
from sim.registry import load_registry  # noqa: E402


def _time(text: str | None) -> datetime | None:
    if text is None:
        return None
    dt = datetime.fromisoformat(text)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="print the planned statements and run nothing")
    p.add_argument("--sqlite", help="profile a local SQLite fixture instead of the real database")
    p.add_argument("--layout", default="configs/ptdb_layout.yaml")
    p.add_argument("--tables", nargs="*", help="tables to profile (default: the included feeds)")
    p.add_argument("--since", help="UTC start, for example 2026-01-01T00:00:00")
    p.add_argument("--until", help="UTC end")
    p.add_argument("--chunks", type=int, default=200)
    p.add_argument("--chunk-rows", type=int, default=5000)
    p.add_argument("--timeline-probes", type=int, default=2000)
    p.add_argument("--window-hours", type=float, default=24.0)
    p.add_argument("--pause-seconds", type=float, default=0.5, help="pause after every query (real database)")
    p.add_argument("--out", default="out/profile")
    args = p.parse_args(argv)

    reg = load_registry(args.layout)
    settings = ProfileSettings(chunks=args.chunks, chunk_rows=args.chunk_rows,
                               timeline_probes=args.timeline_probes, window_hours=args.window_hours,
                               since=_time(args.since), until=_time(args.until), tables=args.tables)
    if args.dry_run:
        print("DRY RUN: nothing is executed and no connection is made. Planned statements:")
        Profiler(DryRunAdapter(), reg, settings).dry_run()
        return 0

    adapter = SQLiteAdapter(args.sqlite) if args.sqlite else MySQLAdapter(pause_seconds=args.pause_seconds)
    try:
        report = Profiler(adapter, reg, settings, log=print).run()
    finally:
        adapter.close()
    jpath, mpath = write_report(report, args.out)
    print(f"Wrote {jpath} and {mpath} ({report['statements_run']} statements)")
    print(BANNER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
