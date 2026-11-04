"""Make inspection plots for one run.

Usage:
    python scripts/plot_run.py out/mixed_demo
    python scripts/plot_run.py configs/mixed_demo.yaml      (generates the run first)
    python scripts/plot_run.py out/mixed_demo --no-scatter

Given a run folder, writes PNGs to <run folder>/plots/. Given a scenario YAML,
generates the run into out/<config name>/ first (same as make_run.py), then plots it.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.io import write_run  # noqa: E402
from sim.plotting import plot_run  # noqa: E402
from sim.scenario import ScenarioError, load_scenario  # noqa: E402
from sim.stream import generate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="a run folder, or a scenario YAML file")
    parser.add_argument("--out", default="out", help="output root when SOURCE is a YAML (default: out)")
    parser.add_argument("--no-scatter", action="store_true", help="skip the decoupling scatter plots")
    args = parser.parse_args()

    source = Path(args.source)
    if source.suffix in (".yaml", ".yml"):
        try:
            cfg = load_scenario(source)
        except ScenarioError as exc:
            print(f"Config error: {exc}", file=sys.stderr)
            return 2
        run_dir = write_run(generate(cfg), Path(args.out) / source.stem)
    elif (source / "data.csv").is_file():
        run_dir = source
    else:
        print(f"{source} is neither a scenario YAML nor a run folder with data.csv", file=sys.stderr)
        return 2

    for path in plot_run(run_dir, scatter=not args.no_scatter):
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
