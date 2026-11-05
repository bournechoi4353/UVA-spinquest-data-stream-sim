"""A simulated LIVE ptdb for testing the monitor (and later the Discord bot) while the logger is off.

All data is SIMULATED. It has the real table and column names and the quirks of the real data:
partial rows with NULLs, late and duplicate rows, the measured feed periods, and the healthy model
fitted to out/profile/ptdb_profile.json when it exists. The planted anomalies, mostly on QT, have
known answers.

Terminal 1, the simulated database (24 h of healthy history, then 12 h live at 60x, so 12 minutes):

    python scripts/live_sim.py --out out/live/demo --history-hours 24 --live-hours 12 --speed 60

Terminal 2, the monitor following it exactly as it would follow ptdb (start it after terminal 1
says the history is written):

    python scripts/run_monitor.py --sqlite out/live/demo/live.sqlite --follow --poll-seconds 5 \\
        --calibrate-minutes 1440 --every-minutes 30 --name live_demo

Watch out/monitor/live_demo/summary.txt change, or ask questions with the bot offline:

    python -m bot.discord_bot --monitor-dir out/monitor/live_demo --ask "anything wrong?"

Afterwards, score what the monitor found against the planted ground truth:

    python scripts/live_sim.py --score out/live/demo --monitor-dir out/monitor/live_demo

Options:
  --speed S         simulated seconds per wall second (1 = real time; 60 = one hour per minute)
  --start ISO|now   UTC time where the live part begins (default now: at --speed 1 the timestamps
                    follow the clock)
  --anomalies plan|none   the planted plan (default) or a healthy stream
  --quiet           do not print each planted anomaly as it happens (for a blind test)
Outputs in --out: live.sqlite (the database), labels.json (ground truth, never read by the
monitor) and live_info.json (settings).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.layout_scenario import datetime_to_us, generate_layout  # noqa: E402
from sim.live import LiveWriter, floor_now, run_live, score_alarms, write_ground_truth  # noqa: E402
from sim.realistic import planted_anomalies, realistic_config  # noqa: E402
from sim.registry import load_registry  # noqa: E402

UTC = timezone.utc


def build_run(reg, profile: Path | None, history_h: float, live_h: float, seed: int, live_start: datetime,
              anomalies: str):
    start = live_start - timedelta(hours=history_h)
    cfg = realistic_config(reg, profile, history_h + live_h, seed,
                           start=start.replace(tzinfo=None).isoformat(timespec="seconds"))
    if anomalies == "plan":
        cfg["anomalies"] = planted_anomalies(reg, live_h, offset_s=history_h * 3600)
    return generate_layout(cfg, reg)


def score(args) -> int:
    folder = Path(args.score)
    labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
    info = json.loads((folder / "live_info.json").read_text(encoding="utf-8"))
    mon = Path(args.monitor_dir)
    alarms = [json.loads(x) for x in (mon / "alarms.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    state = json.loads((mon / "state.json").read_text(encoding="utf-8"))
    live_start = datetime.fromisoformat(info["live_start"]).replace(tzinfo=UTC)
    end = datetime.fromisoformat(state["now"]) if state.get("now") else None
    r = score_alarms(alarms, labels, live_start, end)
    print(f"Planted in the live part (up to the monitor's last event): {r['planted']}")
    print(f"Found: {len(r['found'])}")
    for f in r["found"]:
        print(f"  {f['type']:14s} {f['channel']:34s} as {f['kind']:12s} delay {f['delay_s']} s")
    print(f"Missed: {len(r['missed'])}")
    for m in r["missed"]:
        print(f"  {m['type']:14s} {m['channel']:34s} from {m['start']}")
    print(f"False alarms: {len(r['false_alarms'])}")
    for a in r["false_alarms"][:20]:
        print(f"  {a['subject']} {a['kind']} at {a['at']}")
    return 0 if not r["missed"] else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="out/live/demo")
    p.add_argument("--layout", default="configs/ptdb_layout.yaml")
    p.add_argument("--profile", default="out/profile/ptdb_profile.json")
    p.add_argument("--history-hours", type=float, default=24.0)
    p.add_argument("--live-hours", type=float, default=12.0)
    p.add_argument("--speed", type=float, default=60.0)
    p.add_argument("--start", default="now")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--anomalies", choices=["plan", "none"], default="plan")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--score", metavar="SIM_FOLDER", help="score a monitor run against this simulation's labels")
    p.add_argument("--monitor-dir", help="with --score: the run_monitor.py output folder")
    args = p.parse_args(argv)
    if args.score:
        if not args.monitor_dir:
            p.error("--score needs --monitor-dir")
        return score(args)
    if args.speed <= 0:
        p.error("--speed must be positive")

    reg = load_registry(args.layout)
    live_start = floor_now() if args.start == "now" else datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    profile = Path(args.profile) if args.profile else None
    print(f"Building the SIMULATED run: {args.history_hours:g} h history + {args.live_hours:g} h live, "
          f"seed {args.seed}, profile {'yes' if profile and profile.exists() else 'no'}")
    run = build_run(reg, profile, args.history_hours, args.live_hours, args.seed, live_start, args.anomalies)
    out = Path(args.out)
    writer = LiveWriter(run, out / "live.sqlite", seed=args.seed)
    live_us = datetime_to_us(live_start)
    write_ground_truth(out, run, live_us, {"speed": args.speed, "history_hours": args.history_hours,
                                           "live_hours": args.live_hours, "seed": args.seed,
                                           "anomalies": args.anomalies})
    wall_min = args.live_hours * 60 / args.speed
    print(f"Database: {out / 'live.sqlite'}  ({len(writer.pending)} rows in all; the live part takes about "
          f"{wall_min:.1f} min of wall time at {args.speed:g}x)")
    print(f"Monitor: python scripts/run_monitor.py --sqlite {out / 'live.sqlite'} --follow --poll-seconds 5 "
          f"--calibrate-minutes {int(args.history_hours * 60)} --name live_{out.name}")
    try:
        run_live(writer, live_us, args.speed, announce=print,
                 labels=None if args.quiet else run.labels)
    except KeyboardInterrupt:
        print(f"\nStopped: {writer.written} of {len(writer.pending)} rows written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
