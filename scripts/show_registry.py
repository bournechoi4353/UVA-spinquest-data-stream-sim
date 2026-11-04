"""Print the channel registry: included channels in a table, then the other tables.

Usage::

    python scripts/show_registry.py
    python scripts/show_registry.py --layout configs/ptdb_layout.yaml --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.registry import load_registry  # noqa: E402


def _print_table(rows: list[list[str]], header: list[str]) -> None:
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    print(line)
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(str(x).ljust(w) for x, w in zip(r, widths)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layout", default="configs/ptdb_layout.yaml")
    parser.add_argument("--all", action="store_true",
                        help="list the channels of every table, not only the included ones")
    args = parser.parse_args(argv)

    reg = load_registry(args.layout)
    tables = reg.tables if args.all else reg.included_tables()
    rows = []
    for t in tables:
        for c in t.channels:
            rows.append([c.key, c.display_name, c.group, c.role, c.registry_status,
                         c.unit or "unknown", "yes" if t.include else "no"])
    _print_table(rows, ["key", "display name", "group", "role", "status", "unit", "included"])

    print()
    print("Feeds:")
    _print_table(
        [[t.name, t.time_column, t.time_type, t.nominal_period_s or "", len(t.channels),
          "yes" if t.include else "no", t.reason or ""] for t in reg.tables],
        ["table", "time column", "time type", "period s", "channels", "included", "reason"],
    )

    included = reg.included_channels()
    print()
    print(f"{len(included)} included channels in {len(reg.included_tables())} feeds. "
          f"{reg.total_columns} columns in {len(reg.tables)} tables in total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
