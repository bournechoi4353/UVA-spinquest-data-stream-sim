"""End-to-end test of the whole pipeline on realistic SIMULATED data with known answers.

    python scripts/run_end_to_end.py                                   # template answerer, laptop
    python scripts/run_end_to_end.py --answerer model --adapter out/model/adapter   # GPU machine

The chain is the same one that will run on real data:

    simulated ptdb-shaped SQLite fixture
      -> monitor/ingest.py PtdbReader (read-only adapter, paging by id, reorder, dedup)
      -> detectors and state store (calibrated on a separate healthy run, also read through the reader)
      -> summary text
      -> answerer (template or fine-tuned model), checked fact by fact

The data copies what the real profile showed (fitted from out/profile/ptdb_profile.json when it
exists): partial rows with many NULLs, shared timestamps, late and duplicate rows, a channel held at a
sentinel-like constant, channels that barely move, measured feed periods, and a long feed outage. On
top of that, known anomalies are planted on real channel names, including QT.

Output: <out>/report.md and report.json (detections against ground truth, false alarms, answer
checks, example summaries and answers). The exit code is 0 when every planted anomaly is detected,
there are no false alarms, and at least --min-answer-pass of the answers pass their checks.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monitor.engine import Engine  # noqa: E402
from monitor.ingest import PtdbReader  # noqa: E402
from monitor.scoring import COMPATIBLE, GRACE_S  # noqa: E402
from sim.db_writer import write_fixture  # noqa: E402
from sim.layout_scenario import generate_layout  # noqa: E402
from sim.profiler.adapters import SQLiteAdapter  # noqa: E402
from sim.realistic import planted_anomalies, realistic_config  # noqa: E402
from sim.registry import Registry, load_registry  # noqa: E402
from training.evaluate import check  # noqa: E402
from training.qa import make_examples, summary_of  # noqa: E402

def read_through_reader(cfg: dict[str, Any], reg: Registry, path: Path, stress: bool) -> list:
    run = generate_layout(cfg, reg)
    write_fixture(run, path, null_fraction=0.0, dup_fraction=0.01 if stress else 0.0,
                  late_fraction=0.01 if stress else 0.0)
    adapter = SQLiteAdapter(path)
    try:
        reader = PtdbReader(adapter, reg)
        events = list(reader.read_all())
        stats = reader.stats
    finally:
        adapter.close()
    return run, events, stats


def match(raised, labels, run_end: datetime) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    def t(iso):
        return None if iso is None else datetime.fromisoformat(iso)
    found: dict[str, Any] = {}
    false: list[dict[str, Any]] = []
    for c in raised:
        hits = [lb for lb in labels if lb["channel"] == c.subject and t(lb["start"]) <= c.detected_at
                and (c.detected_at - (t(lb["end"]) or run_end)).total_seconds() <= GRACE_S]
        good = [lb for lb in hits if lb["type"] in COMPATIBLE.get(c.kind, set())] or hits
        if good:
            lb = good[0]
            if lb["id"] not in found:
                fd = t(lb["first_detectable"])
                found[lb["id"]] = {"type": lb["type"], "channel": lb["channel"], "kind": c.kind,
                                   "detected_at": c.detected_at.isoformat(),
                                   "delay_s": None if fd is None else (c.detected_at - fd).total_seconds()}
        else:
            false.append({"subject": c.subject, "kind": c.kind, "at": c.detected_at.isoformat()})
    return found, false


def make_answerer(args):
    if args.answerer == "template":
        from bot.answerer import TemplateAnswerer
        return TemplateAnswerer()
    from bot.answerer import ModelAnswerer
    return ModelAnswerer(args.base_model, args.adapter)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", default="out/profile/ptdb_profile.json")
    p.add_argument("--layout", default="configs/ptdb_layout.yaml")
    p.add_argument("--hours", type=float, default=12.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--answerer", choices=["template", "model"], default="template")
    p.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--adapter")
    p.add_argument("--calibrate-hours", type=float, default=24.0,
                   help="length of the separate healthy calibration run (real use: a day or more)")
    p.add_argument("--snapshots", type=int, default=24)
    p.add_argument("--min-answer-pass", type=float, default=0.9)
    p.add_argument("--out", default="out/e2e")
    args = p.parse_args(argv)

    reg = load_registry(args.layout)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = realistic_config(reg, Path(args.profile) if args.profile else None, args.hours, args.seed)
    healthy_cfg = {**cfg, "seed": args.seed + 7919, "duration_seconds": args.calibrate_hours * 3600}
    test_cfg = {**cfg, "anomalies": planted_anomalies(reg, args.hours)}

    print("1. healthy calibration run -> fixture -> reader")
    _, healthy_events, _ = read_through_reader(healthy_cfg, reg, out / "healthy.sqlite", stress=False)
    print("2. test run with planted anomalies and stress rows -> fixture -> reader")
    run, events, rstats = read_through_reader(test_cfg, reg, out / "test.sqlite", stress=True)

    feeds = {t.name: [c.key for c in t.channels] for t in reg.included_tables()}
    display = {c.key: (c.display_name, c.unit) for c in reg.included_channels()}
    engine = Engine.from_healthy(healthy_events, feeds=feeds, display=display)
    # Display names in labels, so answers about a planted anomaly can be checked by name.
    name_of = {**{k: v[0] for k, v in display.items()}, **{t.name: t.name for t in reg.included_tables()}}

    print("3. detectors, state store, summaries")
    t0, t1 = events[0][0].timestamp(), events[-1][0].timestamp()
    marks = [t0 + (k + 1) * (t1 - t0) / (args.snapshots + 1) for k in range(args.snapshots)]
    for lb in run.labels:   # also look right after each anomaly becomes detectable
        if lb["first_detectable"]:
            marks.append(datetime.fromisoformat(lb["first_detectable"]).timestamp() + 300)
    marks = sorted(m for m in marks if m <= t1)
    snaps, i = [], 0
    for t, ev in events:
        engine.step(t, ev)
        while i < len(marks) and t.timestamp() >= marks[i]:
            summary, facts = summary_of(engine.store)
            snaps.append({"time": t.isoformat(), "summary": summary, "facts": facts})
            i += 1

    found, false = match(engine.raised, run.labels, events[-1][0])
    print(f"   planted {len(run.labels)}, detected {len(found)}, false alarms {len(false)}")

    print(f"4. answers ({args.answerer})")
    answerer = make_answerer(args)
    rng = random.Random(args.seed)
    qa = []
    for s in snaps:
        for ex in make_examples(s["facts"], s["summary"], rng, max_examples=4, holdout=True):
            ans = answerer(s["summary"], ex["question"])
            res = check(ans, s["summary"], ex["question"], ex["checks"])
            qa.append({"time": s["time"], "intent": ex["intent"], "question": ex["question"], "answer": ans,
                       "reference": ex["answer"], **res})
    pass_rate = sum(q["pass"] for q in qa) / max(len(qa), 1)
    halluc = sum(q["hallucinated"] for q in qa) / max(len(qa), 1)
    print(f"   {len(qa)} questions, pass {pass_rate:.1%}, hallucinated {halluc:.1%}")

    labels = [{**lb, "display": name_of.get(lb["channel"], lb["channel"]), "detected": lb["id"] in found,
               **({"detection": found[lb["id"]]} if lb["id"] in found else {})} for lb in run.labels]
    report = {
        "synthetic": True,
        "hours": args.hours, "seed": args.seed, "answerer": args.answerer, "adapter": args.adapter,
        "profile_used": bool(args.profile and Path(args.profile).exists()),
        "reader": {"rows": rstats.rows, "events": rstats.emitted, "duplicates_dropped": rstats.duplicates,
                   "too_late": rstats.too_late, "null_values": rstats.nulls},
        "labels": labels, "false_alarms": false,
        "answers": {"n": len(qa), "pass_rate": pass_rate, "hallucination_rate": halluc},
        "qa": qa, "snapshots": [{"time": s["time"], "summary": s["summary"]} for s in snaps],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "report.md").write_text(render(report), encoding="utf-8")
    ok = all(lb["detected"] for lb in labels) and not false and pass_rate >= args.min_answer_pass
    print(f"Wrote {out / 'report.md'} -> {'PASS' if ok else 'CHECK THE REPORT'}")
    return 0 if ok else 1


def render(r: dict[str, Any]) -> str:
    lines = ["# End-to-end test (SIMULATED data, real channel names)", "",
             f"{r['hours']:g} h, seed {r['seed']}, answerer: {r['answerer']}"
             + (f" ({r['adapter']})" if r["adapter"] else "")
             + f". Healthy model fitted to the real profile: {'yes' if r['profile_used'] else 'no'}.", "",
             "## Reader", "", "| rows | events | duplicates dropped | too late | NULL values |", "|---|---|---|---|---|",
             f"| {r['reader']['rows']} | {r['reader']['events']} | {r['reader']['duplicates_dropped']} | "
             f"{r['reader']['too_late']} | {r['reader']['null_values']} |", "",
             "## Planted anomalies", "", "| type | channel | start | detected | as | delay s |",
             "|---|---|---|---|---|---|"]
    for lb in r["labels"]:
        d = lb.get("detection") or {}
        delay = "" if d.get("delay_s") is None else f"{d['delay_s']:.0f}"
        lines.append(f"| {lb['type']} | {lb['display']} | {lb['start'][11:16]} | "
                     f"{'yes' if lb['detected'] else 'NO'} | {d.get('kind', '')} | {delay} |")
    lines += ["", f"False alarms: {len(r['false_alarms'])}", ""]
    for fa in r["false_alarms"][:20]:
        lines.append(f"- {fa['subject']} {fa['kind']} at {fa['at'][11:19]}")
    a = r["answers"]
    lines += ["", "## Answers", "", f"{a['n']} questions (held-out phrasings), pass {a['pass_rate']:.1%}, "
              f"hallucinated {a['hallucination_rate']:.1%}", ""]
    failed = [q for q in r["qa"] if not q["pass"] or q["hallucinated"]]
    for q in failed[:10]:
        lines += [f"- **Q:** {q['question']}", f"  - answer: {q['answer'][:300]}",
                  f"  - expected: {q['reference'][:300]}"]
    lines += ["", "## Example summaries and answers", ""]
    shown = 0
    for s in r["snapshots"]:
        if "0 abnormal" in s["summary"] and shown > 1:
            continue
        qs = [q for q in r["qa"] if q["time"] == s["time"]][:2]
        lines += [f"### {s['time'][11:16]}", "", "```", s["summary"], "```", ""]
        for q in qs:
            lines += [f"- **Q:** {q['question']}", f"  **A:** {q['answer']}", ""]
        shown += 1
        if shown >= 5:
            break
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
