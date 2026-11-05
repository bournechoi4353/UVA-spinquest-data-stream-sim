"""Randomized detector stress test on realistic SIMULATED ptdb data (see monitor/stress.py).

    python scripts/stress_test.py --seeds 50 --workers 6            # about 10 to 20 minutes
    python scripts/stress_test.py --seeds 4 --hours 4 --calibrate-hours 8   # quick check

Every seed gets its own random plan of 8 anomalies (every 5th seed is clean), with sizes from 2 to
10 sigma, random channels, durations and timing. The report gives detection rates with 95% Wilson
intervals per type and per size, median delays, false alarms per hour with 95% intervals, and lists
what was missed. Output: <out>/report.md, results.json and labels.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monitor.stress import TYPES, aggregate, run_seed  # noqa: E402


def _one(args):
    seed, hours, cal, profile = args
    return run_seed(seed, hours, cal, profile)


def pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.0f}%"


def ci(c) -> str:
    lo, hi = c
    return "n/a" if lo != lo else f"{100 * lo:.0f} to {100 * hi:.0f}%"


def render(agg, results, args) -> str:
    L = []
    add = L.append
    a = agg
    add("# Detector stress test (SIMULATED data, real channel names)\n")
    add(f"{a['runs']} runs ({a['clean_runs']} clean), {a['test_hours']:.0f} test hours, {a['labels']} planted anomalies. "
        f"Each run: {args.hours:g} h test, calibrated on a separate {args.calibrate_hours:g} h healthy run; both written "
        "to a ptdb-shaped SQLite file with 1% duplicate and 1% late rows and read through the live reader. "
        "Sizes are log-uniform from 2 to 10 sigma of the channel's simulated noise.\n")
    s = a["in_scope"]
    add("## Summary\n")
    add("| | n | detected | rate | 95% interval |")
    add("|---|---|---|---|---|")
    add(f"| Anomalies the detectors are meant to catch | {s['n']} | {s['detected']} | {pct(s['rate'])} | {ci(s['ci'])} |")
    b = a["by_design"]
    add(f"| Below a design limit (see below) | {b['n']} | {b['detected']} | {pct(b['rate'])} | {ci(b['ci'])} |")
    if a.get("by_design_reasons"):
        add("\nDesign limits: " + "; ".join(f"{k} ({v})" for k, v in
                                          sorted(a["by_design_reasons"].items(), key=lambda kv: -kv[1])) + ".")
    f, cf = a["false_alarms"], a["clean_false_alarms"]
    add(f"\nFalse alarms (alarms matching no planted anomaly): {f['total']} in {a['test_hours']:.0f} h = "
        f"{f['per_hour']:.3f} per hour (95%: {f['ci'][0]:.3f} to {f['ci'][1]:.3f}).")
    add(f"On the {a['clean_runs']} clean runs alone: {cf['total']} in {cf['hours']:.0f} h = {cf['per_hour']:.3f} per hour "
        f"(95%: {cf['ci'][0]:.3f} to {cf['ci'][1]:.3f}).")
    if a["false_alarm_kinds"]:
        add("False alarms by kind: " + ", ".join(f"{k} {v}" for k, v in sorted(a["false_alarm_kinds"].items(),
                                                                          key=lambda kv: -kv[1])) + ".")
    add("\n## Per type (in scope)\n")
    add("| type | n | detected | rate | 95% interval | as a matching kind | median delay s |")
    add("|---|---|---|---|---|---|---|")
    for ty in TYPES:
        r = a["per_type"].get(ty)
        if r:
            add(f"| {ty} | {r['n']} | {r['detected']} | {pct(r['rate'])} | {ci(r['ci'])} | {r['compatible']} | "
                f"{'' if r['median_delay_s'] is None else r['median_delay_s']} |")
    add("\n## Per type and size (in scope)\n")
    add("| type | size | n | detected | rate | 95% interval |")
    add("|---|---|---|---|---|---|")
    for ty in TYPES:
        for bname, r in a["per_type_size"].get(ty, {}).items():
            add(f"| {ty} | {bname} | {r['n']} | {r['detected']} | {pct(r['rate'])} | {ci(r['ci'])} |")
    add("\n## Per type and size in the detector's own sigma (in scope)\n")
    add("The planted size is in units of the channel's short-term simulated noise. Slow wander makes the healthy "
        "spread, and so the detector's sigma, larger on some channels; this table gives the size the detector "
        "actually faced.\n")
    add("| type | size | n | detected | rate | 95% interval |")
    add("|---|---|---|---|---|---|")
    for ty in TYPES:
        for bname, r in a.get("per_type_eff", {}).get(ty, {}).items():
            add(f"| {ty} | {bname} | {r['n']} | {r['detected']} | {pct(r['rate'])} | {ci(r['ci'])} |")
    missed = [lb for r in results for lb in r["labels"] if not lb["detected"] and not lb["by_design"]]
    add(f"\n## Missed (in scope): {len(missed)}\n")
    for lb in sorted(missed, key=lambda x: (x["type"], x["channel"]))[:80]:
        size = (f"{lb['magnitude_sigma']:+.1f} sigma ({lb['eff_sigma']} detector sigma)"
                if lb["magnitude_sigma"] is not None
                else f"x{lb['factor']}" if lb["factor"] is not None else f"{lb['duration_s']} s")
        add(f"- seed {lb['seed']}: {lb['type']} on {lb['channel']}, {size}")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--first-seed", type=int, default=1)
    p.add_argument("--hours", type=float, default=12.0)
    p.add_argument("--calibrate-hours", type=float, default=24.0)
    p.add_argument("--profile", default="out/profile/ptdb_profile.json")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="out/stress")
    args = p.parse_args(argv)
    profile = args.profile if Path(args.profile).exists() else None
    jobs = [(s, args.hours, args.calibrate_hours, profile)
            for s in range(args.first_seed, args.first_seed + args.seeds)]
    results = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for i, r in enumerate(ex.map(_one, jobs), 1):
            results.append(r)
            print(f"  seed {r['seed']}: {sum(lb['detected'] for lb in r['labels'])}/{len(r['labels'])} detected, "
                  f"{len(r['false_alarms'])} false alarms ({i}/{len(jobs)})", flush=True)
    agg = aggregate(results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"summary": agg, "runs": results}, indent=1, default=str) + "\n",
                                      encoding="utf-8")
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as fh:
        rows = [lb for r in results for lb in r["labels"]]
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    text = render(agg, results, args)
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
