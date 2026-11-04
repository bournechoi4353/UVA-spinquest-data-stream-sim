"""Build the question and answer dataset from simulated runs.

Usage:
    python -m training.make_dataset --batch out/batch --layout-runs 120 --out out/qa

For every run, the real detector pipeline runs over the stream (calibrated on
a separate healthy run of the same system), and snapshots of the state store
are taken at random times. Each snapshot gives one summary and a few
(question, answer) examples. Two kinds of runs are used:

* generic batch runs (out/batch, placeholder channel names), keeping their
  train/val/test split;
* wide-feed runs with the real ptdb names (SIMULATED), split 80/10/10 by index.

Output: <out>/{train,val,test,test_holdout}.jsonl with one example per line:
    {"messages": [system, user, assistant], "intent": ..., "checks": {...}, "source": ...}
and <out>/stats.json. test_holdout uses question phrasings never seen in training.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monitor.engine import Engine  # noqa: E402
from monitor.scoring import couplings_of, healthy_config  # noqa: E402
from sim.layout_recipes import random_anomalies  # noqa: E402
from sim.layout_scenario import generate_layout, merge_feed_events  # noqa: E402
from sim.registry import load_registry  # noqa: E402
from sim.stream import replay  # noqa: E402
from training.qa import make_examples, summary_of, to_chat  # noqa: E402

SNAPSHOTS_PER_RUN = 6


def snapshots(engine: Engine, events: Iterable, times_frac: list[float], start: float, end: float):
    """Step the engine through events and yield (summary, facts) at the requested fractions of the run."""
    marks = sorted(start + f * (end - start) for f in times_frac)
    i = 0
    for t, ev in events:
        engine.step(t, ev)
        while i < len(marks) and t.timestamp() >= marks[i]:
            yield summary_of(engine.store)
            i += 1


def generic_examples(batch: Path, rng: random.Random, limit: int | None) -> dict[str, list[dict]]:
    manifest = json.loads((batch / "manifest.json").read_text())
    out: dict[str, list[dict]] = {"train": [], "val": [], "test": [], "test_holdout": []}
    runs = manifest["runs"][:limit] if limit else manifest["runs"]
    for k, r in enumerate(runs, 1):
        run_dir = batch / r["path"]
        raw = yaml.safe_load((run_dir / "config_used.yaml").read_text())
        engine = Engine.from_healthy(replay(healthy_config(raw)), couplings=couplings_of(raw))
        events = list(replay(run_dir))
        start, end = events[0][0].timestamp(), events[-1][0].timestamp()
        fracs = [rng.uniform(0.05, 1.0) for _ in range(SNAPSHOTS_PER_RUN)]
        for summary, facts in snapshots(engine, events, fracs, start, end):
            for ex in make_examples(facts, summary, rng):
                out[r["split"]].append(_row(summary, ex, f"batch/{r['path']}"))
            if r["split"] == "test":
                for ex in make_examples(facts, summary, rng, holdout=True):
                    out["test_holdout"].append(_row(summary, ex, f"batch/{r['path']}"))
        if k % 20 == 0:
            print(f"  generic runs: {k}/{len(runs)}", flush=True)
    return out


def layout_examples(n_runs: int, seed: int, rng: random.Random) -> dict[str, list[dict]]:
    reg = load_registry(Path(__file__).resolve().parent.parent / "configs" / "ptdb_layout.yaml")
    feeds = {t.name: [c.key for c in t.channels] for t in reg.included_tables()}
    display = {c.key: (c.display_name, c.unit) for c in reg.included_channels()}
    out: dict[str, list[dict]] = {"train": [], "val": [], "test": [], "test_holdout": []}
    nprng = np.random.default_rng(seed)
    for k in range(n_runs):
        split = "train" if k % 10 < 8 else ("val" if k % 10 == 8 else "test")
        hours = float(nprng.uniform(2, 8))
        base = {"seed": seed * 1000 + k, "start_time": f"2026-0{1 + k % 9}-1{k % 9}T0{k % 10}:00:00",
                "duration_seconds": hours * 3600}
        # Per-channel levels and noise vary between runs so the model sees many number formats.
        base["channels"] = {key: {"baseline": float(f"{10 ** nprng.uniform(-2, 3):.4g}"),
                                  "sigma": 1.0, "ar1_phi": float(nprng.choice([0.0, 0.3, 0.6]))}
                            for key in display}
        for v in base["channels"].values():
            v["sigma"] = float(f"{v['baseline'] * 10 ** nprng.uniform(-3, -1.5):.3g}")
        healthy = generate_layout({**base, "seed": base["seed"] + 7919, "duration_seconds": 6 * 3600}, reg)
        engine = Engine.from_healthy(merge_feed_events(healthy.feeds), feeds=feeds, display=display)
        cfg = {**base, "anomalies": random_anomalies(base, reg, int(nprng.integers(0, 5)))}
        run = generate_layout(cfg, reg)
        events = list(merge_feed_events(run.feeds))
        start, end = events[0][0].timestamp(), events[-1][0].timestamp()
        fracs = [rng.uniform(0.05, 1.0) for _ in range(SNAPSHOTS_PER_RUN)]
        for summary, facts in snapshots(engine, events, fracs, start, end):
            for ex in make_examples(facts, summary, rng):
                out[split].append(_row(summary, ex, f"layout/{k}"))
            if split == "test":
                for ex in make_examples(facts, summary, rng, holdout=True):
                    out["test_holdout"].append(_row(summary, ex, f"layout/{k}"))
        if (k + 1) % 20 == 0:
            print(f"  layout runs: {k + 1}/{n_runs}", flush=True)
    return out


def _row(summary: str, ex: dict[str, Any], source: str) -> dict[str, Any]:
    return {"messages": to_chat(summary, ex["question"], ex["answer"]), "intent": ex["intent"],
            "checks": ex["checks"], "source": source}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch", default="out/batch")
    p.add_argument("--batch-limit", type=int, help="use only the first N batch runs")
    p.add_argument("--layout-runs", type=int, default=120)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="out/qa")
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    parts = []
    if args.batch and Path(args.batch, "manifest.json").exists():
        parts.append(generic_examples(Path(args.batch), rng, args.batch_limit))
    if args.layout_runs:
        parts.append(layout_examples(args.layout_runs, args.seed, rng))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {}
    for split in ("train", "val", "test", "test_holdout"):
        rows = [r for part in parts for r in part[split]]
        random.Random(args.seed + len(split)).shuffle(rows)   # hash() is salted per process
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        stats[split] = {"examples": len(rows), "intents": dict(Counter(r["intent"] for r in rows))}
        print(f"{split}: {len(rows)} examples")
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
