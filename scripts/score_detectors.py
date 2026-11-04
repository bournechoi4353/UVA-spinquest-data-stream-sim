"""Score the reference detectors on a batch split (Phase 5).

Usage:
    python scripts/make_batch.py --n 200 --master-seed 1
    python scripts/score_detectors.py out/batch --split test
    python scripts/score_detectors.py out/batch --split test --plain   # Phase 5 thresholds, no adaptation

Runs every scenario of the split through ``replay`` (labels are never read
during detection), matches raised conditions to labels, and prints per anomaly
type: detection rate and median detection delay (alarm time minus
first_detectable), plus false alarms per hour on clean runs. Any type with a
detection rate under 90 percent is flagged. "masked" counts labels on a channel
that was already in alarm for an earlier anomaly (no new alarm can fire there).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monitor.scoring import score_run, summarize  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("batch", help="batch folder with manifest.json")
    p.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--plain", action="store_true", help="use the fixed Phase 5 thresholds for every channel")
    p.add_argument("--out", help="write the summary JSON here")
    args = p.parse_args(argv)

    batch = Path(args.batch)
    manifest = json.loads((batch / "manifest.json").read_text())
    runs = [r for r in manifest["runs"] if args.split == "all" or r["split"] == args.split]
    clean = {r["run"] for r in runs if r["category"] == "clean"}
    scores = []
    for i, r in enumerate(runs, 1):
        scores.append(score_run(batch / r["path"], adaptive=not args.plain))
        print(f"\r  scored {i}/{len(runs)}", end="", flush=True)
    print()
    summary = summarize(scores, clean)
    mode = "plain Phase 5 thresholds" if args.plain else "per-channel thresholds from healthy data"
    print(f"Detectors on split '{args.split}' ({len(runs)} runs, {mode})")
    print(f"{'anomaly type':20} {'n':>4} {'detected':>9} {'masked':>7} {'rate':>6} {'incl masked':>12} "
          f"{'median delay s':>15}")
    flagged = []
    for t, d in summary["by_type"].items():
        rate = d["rate"]
        flag = rate is not None and rate < 0.9
        if flag:
            flagged.append(t)
        delay = "" if d["median_delay_s"] is None else f"{d['median_delay_s']:.0f}"
        print(f"{t:20} {d['n']:>4} {d['detected']:>9} {d['masked']:>7} {rate:>6.0%} "
              f"{d['rate_incl_masked']:>12.0%} {delay:>15} {'UNDER 90%' if flag else ''}")
    fa = summary["false_alarms_per_hour_clean"]
    print(f"\nFalse alarms per hour on clean runs: {fa:.3f} ({summary['clean_runs']} runs, "
          f"{summary['clean_hours']:.1f} h)" if fa is not None else "\nNo clean runs in this split.")
    print(f"False alarms per hour over all runs: {summary['false_alarms_per_hour_all']:.3f}")
    print(f"Types under 90 percent: {', '.join(flagged) or 'none'}")
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
