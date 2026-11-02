"""Generate one run from a scenario config.

Usage:
    python scripts/make_run.py configs/mixed_demo.yaml
    python scripts/make_run.py configs/mixed_demo.yaml --out out --name my_run

Writes out/<run_name>/data.csv, labels.json and config_used.yaml.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.io import write_run  # noqa: E402
from sim.scenario import ScenarioError, load_scenario  # noqa: E402
from sim.stream import generate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="path to a scenario YAML file")
    parser.add_argument("--out", default="out", help="output root folder (default: out)")
    parser.add_argument("--name", default=None, help="run folder name (default: the config file name)")
    args = parser.parse_args()

    try:
        cfg = load_scenario(args.config)
    except ScenarioError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    result = generate(cfg)
    run_dir = write_run(result, Path(args.out) / (args.name or Path(args.config).stem))
    n_rows = sum(len(c.t_us) for c in result.channels.values())
    print(f"Wrote {run_dir}")
    print(f"  channels: {', '.join(result.channels)} ({n_rows} samples in total)")
    print(f"  labels:   {len(result.labels)} injected anomalies")
    return 0


if __name__ == "__main__":
    sys.exit(main())
