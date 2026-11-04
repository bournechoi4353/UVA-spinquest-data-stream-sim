"""Fit the simulator to a profile, simulate, profile the simulation, and compare (Phase R4).

Usage:
    python scripts/compare_sim_to_profile.py out/profile/ptdb_profile.json
    python scripts/compare_sim_to_profile.py out/profile/ptdb_profile.json --couplings configs/couplings.yaml --hours 48

Steps: fit the healthy model to the profile, simulate a healthy run (SIMULATED,
synthetic: true), write it to a SQLite fixture, profile that fixture with the
same profiler code, and print real against simulated aggregates per channel.
A channel is flagged when its noise sigma or lag-1 autocorrelation differs by
more than --threshold (default 25 percent; the lag-1 difference is taken
relative to max(|real|, 0.1) so values near zero do not blow up).

The couplings file holds a list under 'couplings:', chosen by hand from the
profile's correlation lists. Nothing is coupled automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.db_writer import write_fixture  # noqa: E402
from sim.from_profile import NOT_CAPTURED, fit_config, fit_report, load_profile  # noqa: E402
from sim.layout_scenario import generate_layout  # noqa: E402
from sim.profiler import Profiler, ProfileSettings, SQLiteAdapter  # noqa: E402
from sim.registry import load_registry  # noqa: E402


def compare(real: dict, sim: dict, threshold: float = 0.25) -> list[dict]:
    rows = []
    for table, t in real["tables"].items():
        if table not in sim["tables"]:
            continue
        st = sim["tables"][table]
        rows.append({"channel": f"{table} (spacing s)", "real_sigma": t.get("mean_spacing_s"),
                     "sim_sigma": st.get("mean_spacing_s"), "real_lag1": None, "sim_lag1": None,
                     "real_median": None, "sim_median": None, "flag": False})
        for col, rs in (t.get("channels") or {}).items():
            ss = (st.get("channels") or {}).get(col)
            if not ss:
                continue
            flag = False
            if rs.get("noise_mad_sigma") and ss.get("noise_mad_sigma") is not None:
                flag |= abs(ss["noise_mad_sigma"] - rs["noise_mad_sigma"]) > threshold * abs(rs["noise_mad_sigma"])
            if rs.get("noise_lag1") is not None and ss.get("noise_lag1") is not None:
                flag |= abs(ss["noise_lag1"] - rs["noise_lag1"]) > threshold * max(abs(rs["noise_lag1"]), 0.1)
            rows.append({"channel": f"{table}.{col}", "real_sigma": rs.get("noise_mad_sigma"),
                         "sim_sigma": ss.get("noise_mad_sigma"), "real_lag1": rs.get("noise_lag1"),
                         "sim_lag1": ss.get("noise_lag1"), "real_median": rs.get("median"),
                         "sim_median": ss.get("median"), "flag": bool(flag)})
    return rows


def _f(v) -> str:
    return "" if v is None else f"{v:.5g}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("profile")
    p.add_argument("--couplings", help="YAML file with a 'couplings:' list")
    p.add_argument("--hours", type=float, default=24.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--layout", default="configs/ptdb_layout.yaml")
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--out", default="out/compare")
    args = p.parse_args(argv)

    reg = load_registry(args.layout)
    real = load_profile(args.profile)
    couplings = None
    if args.couplings:
        couplings = (yaml.safe_load(Path(args.couplings).read_text()) or {}).get("couplings") or []
    cfg = {"seed": args.seed, "start_time": "2026-01-01T00:00:00", "duration_seconds": args.hours * 3600,
           "layout": args.layout, "synthetic": True, **fit_config(real, reg, couplings)}
    run = generate_layout(cfg, reg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fitted_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    write_fixture(run, out / "fixture.sqlite")

    s = real.get("sampling") or {}
    settings = ProfileSettings(chunks=int(s.get("chunks") or 200), chunk_rows=int(s.get("chunk_rows") or 5000),
                               timeline_probes=int(s.get("timeline_probes") or 2000),
                               window_hours=float(s.get("window_hours") or 24))
    a = SQLiteAdapter(out / "fixture.sqlite")
    try:
        sim = Profiler(a, reg, settings).run()
    finally:
        a.close()
    rows = compare(real, sim, args.threshold)
    (out / "comparison.json").write_text(json.dumps({"rows": rows, "fit": fit_report(real, reg)}, indent=2))

    print(f"SIMULATED run fitted to {args.profile} ({args.hours:g} h, seed {args.seed})")
    print(f"{'channel':62} {'sigma real':>11} {'sigma sim':>11} {'lag1 real':>10} {'lag1 sim':>10} flag")
    for r in rows:
        print(f"{r['channel']:62} {_f(r['real_sigma']):>11} {_f(r['sim_sigma']):>11} "
              f"{_f(r['real_lag1']):>10} {_f(r['sim_lag1']):>10} {'FLAG' if r['flag'] else ''}")
    flagged = [r["channel"] for r in rows if r["flag"]]
    print(f"\n{len(flagged)} channel(s) differ by more than {args.threshold:.0%}: {', '.join(flagged) or 'none'}")
    print(f"\nWhat the fit does not capture: {NOT_CAPTURED}")
    print(f"Wrote {out / 'fitted_config.yaml'}, {out / 'fixture.sqlite'}, {out / 'comparison.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
