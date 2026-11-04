"""Build a SIMULATED wide-feed scenario over the included ptdb feeds (Phase R2).

Usage:
    python scripts/make_layout_scenario.py --hours 6 --seed 1 --anomalies mixed
    python scripts/make_layout_scenario.py --hours 6 --seed 1 --anomalies none --fixture
    python scripts/make_layout_scenario.py --config configs/layout_demo.yaml
    python scripts/make_layout_scenario.py --hours 6 --seed 1 --profile out/profile/ptdb_profile.json

Writes out/layout/<name>/ (feeds/*.csv, labels.json, meta.json, config_used.yaml),
plus fixture.sqlite with --fixture and plots/ with --plot. The data is synthetic even
though it uses the real table and column names.

--anomalies:
    none     healthy data only
    mixed    a fixed recipe drawn from the seed: one feed outage and several channel anomalies
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.db_writer import write_fixture  # noqa: E402
from sim.layout_scenario import generate_layout, load_layout_config  # noqa: E402
from sim.layout_io import write_layout_run  # noqa: E402
from sim.layout_recipes import mixed_anomalies  # noqa: E402
from sim.registry import load_registry  # noqa: E402
from sim.scenario import ScenarioError  # noqa: E402


def build_config(args, reg) -> dict:
    if args.config:
        return load_layout_config(args.config)
    cfg = {
        "seed": args.seed,
        "start_time": args.start,
        "duration_seconds": args.hours * 3600.0,
        "layout": args.layout,
        "synthetic": True,
        "anomalies": [],
    }
    if args.profile:
        from sim.from_profile import fit_config, load_profile
        cfg.update(fit_config(load_profile(args.profile), reg))
    if args.anomalies == "mixed":
        cfg["anomalies"] = mixed_anomalies(cfg, reg)
    return cfg


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="a layout scenario YAML (overrides --hours, --seed, --anomalies)")
    p.add_argument("--hours", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--start", default="2026-01-01T00:00:00", help="UTC start time")
    p.add_argument("--anomalies", choices=["none", "mixed"], default="mixed")
    p.add_argument("--layout", default="configs/ptdb_layout.yaml")
    p.add_argument("--profile", help="fit the healthy model to a profiler JSON (Phase R4)")
    p.add_argument("--out", default="out/layout")
    p.add_argument("--name", help="run folder name (default: seed<seed>_<anomalies>)")
    p.add_argument("--fixture", action="store_true", help="also write fixture.sqlite")
    p.add_argument("--late-fraction", type=float, default=0.0)
    p.add_argument("--dup-fraction", type=float, default=0.0)
    p.add_argument("--null-fraction", type=float, default=0.0)
    p.add_argument("--plot", action="store_true", help="also write SIMULATED plots per feed")
    args = p.parse_args(argv)

    reg = load_registry(args.layout)
    try:
        cfg = build_config(args, reg)
        run = generate_layout(cfg, reg)
    except ScenarioError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2
    name = args.name or (Path(args.config).stem if args.config else f"seed{args.seed}_{args.anomalies}")
    out = write_layout_run(run, Path(args.out) / name)
    print(f"Wrote {out} (SIMULATED, synthetic: true)")
    for feed, info in run.meta["feeds"].items():
        print(f"  {feed}: {info['rows']} rows, period {info['nominal_period_s']:g} s")
    print(f"  labels: {len(run.labels)}")
    if args.fixture:
        counts = write_fixture(run, out / "fixture.sqlite", args.null_fraction, args.dup_fraction,
                               args.late_fraction)
        print(f"  fixture: {out / 'fixture.sqlite'}")
        for t, c in counts.items():
            if c.nulls or c.duplicates or c.late:
                print(f"    {t}: {c.nulls} NULLs, {c.duplicates} duplicates, {c.late} late rows")
    if args.plot:
        from sim.plotting import plot_layout_run
        for path in plot_layout_run(out):
            print(f"  plot: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
