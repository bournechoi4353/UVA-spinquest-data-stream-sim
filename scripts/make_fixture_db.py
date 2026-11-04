"""Write a SIMULATED run into a local SQLite fixture shaped like ptdb (Phase R2).

Usage:
    python scripts/make_fixture_db.py out/layout/seed1_mixed
    python scripts/make_fixture_db.py configs/layout_demo.yaml --out out/layout/demo/fixture.sqlite
    python scripts/make_fixture_db.py out/layout/seed1_mixed --late-fraction 0.01 --dup-fraction 0.005 --null-fraction 0.01

SOURCE is a layout run folder (re-generated from its config_used.yaml, so the
fixture matches it exactly) or a layout scenario YAML. The fixture has the real
table and column names, an id primary key in insertion order, and a TEXT UTC
time column with an index.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.db_writer import write_fixture  # noqa: E402
from sim.layout_scenario import generate_layout  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="layout run folder or layout scenario YAML")
    p.add_argument("--out", help="SQLite path (default: <run folder>/fixture.sqlite)")
    p.add_argument("--late-fraction", type=float, default=0.0)
    p.add_argument("--dup-fraction", type=float, default=0.0)
    p.add_argument("--null-fraction", type=float, default=0.0)
    p.add_argument("--seed", type=int, help="seed for the stress options (default: the run's seed)")
    args = p.parse_args(argv)

    src = Path(args.source)
    cfg_path = src / "config_used.yaml" if src.is_dir() else src
    run = generate_layout(cfg_path)
    out = Path(args.out) if args.out else (src / "fixture.sqlite" if src.is_dir()
                                            else Path("out/layout") / src.stem / "fixture.sqlite")
    counts = write_fixture(run, out, args.null_fraction, args.dup_fraction, args.late_fraction, args.seed)
    print(f"Wrote {out} (SIMULATED)")
    for t, c in counts.items():
        print(f"  {t}: {run.feeds[t].n_rows} rows, +{c.duplicates} duplicates, {c.late} late, {c.nulls} NULLs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
