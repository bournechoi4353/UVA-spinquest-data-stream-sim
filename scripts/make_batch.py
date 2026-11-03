"""Generate a batch of randomized scenarios split into train, val, and test.

Usage:
    python scripts/make_batch.py --n 200 --master-seed 1
    python scripts/make_batch.py --n 30 --master-seed 5 --out out/small --max-hours 2

Writes <out>/<split>/run_XXXX/{data.csv,labels.json,config_used.yaml} and <out>/manifest.json.
Each scenario gets its own seed derived from the master seed, and lands in exactly one split.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.batch import format_summary, generate_batch  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=200, help="number of scenarios (default 200)")
    p.add_argument("--master-seed", type=int, default=1, help="seed that determines the whole batch (default 1)")
    p.add_argument("--out", default="out/batch", help="output folder (default out/batch)")
    p.add_argument("--clean-frac", type=float, default=0.25,
                   help="fraction of ALL runs with no anomalies (default 0.25)")
    p.add_argument("--hard-frac", type=float, default=0.2,
                   help="fraction of the NON-CLEAN runs that are hard cases: very slow drifts, noise-like "
                        "spikes, step right after a spike, stuck sensors (default 0.2)")
    p.add_argument("--multi-frac", type=float, default=0.25,
                   help="fraction of ALL runs with several overlapping anomalies (default 0.25)")
    p.add_argument("--split", default="80,10,10", help="train,val,test percentages (default 80,10,10)")
    p.add_argument("--min-hours", type=float, default=1.0, help="shortest scenario in hours (default 1)")
    p.add_argument("--max-hours", type=float, default=12.0, help="longest scenario in hours (default 12)")
    p.add_argument("--overwrite", action="store_true",
                   help="replace an existing batch in --out (only manifest.json and train/val/test are removed)")
    args = p.parse_args()

    try:
        split = tuple(float(x) for x in args.split.split(","))
        manifest = generate_batch(
            n=args.n, master_seed=args.master_seed, out_dir=args.out, clean_frac=args.clean_frac,
            hard_frac=args.hard_frac, multi_frac=args.multi_frac, split=split,
            duration_range_hours=(args.min_hours, args.max_hours), overwrite=args.overwrite,
        )
    except (ValueError, FileExistsError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote {args.out}/manifest.json and {manifest['n']} runs ({manifest['split_counts']})")
    print()
    print(format_summary(manifest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
