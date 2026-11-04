"""Parse a summary text (monitor/summary.py format) back into the facts structure.

The fallback answerer and the template baseline use this, so they see exactly
what the LLM sees: the summary text, nothing more.
"""

from __future__ import annotations

import re
from typing import Any

HEAD = re.compile(r"Status at (?P<now>\d{4}-\d{2}-\d{2} \d{2}:\d{2}) UTC\. (?P<n>\d+) channels monitored")
LINE = re.compile(r"^- (?P<name>.+?): (?P<body>.+)\. Normal is (?P<normal>[^ ]+?)(?: (?P<unit>\S+))?\.$")
SINCE = re.compile(r"since (?P<t>(?:\d{4}-\d{2}-\d{2} )?\d{2}:\d{2})")
DRIFT = re.compile(r"about (?P<dev>[^ ]+)(?: \S+)? (?P<dir>above|below) normal, (?P<sig>[\d.]+) sigma")
KINDS = [("running high", "drift_up"), ("running low", "drift_down"), ("stuck at a constant value", "stuck"),
         ("noisier than normal", "noisy"), ("not reporting values", "missing"), ("no longer tracking", "decoupled")]
SPIKE_ONE = re.compile(r"^(?P<name>.+?) at (?P<t>\d{2}:\d{2}) \((?P<sig>[+-][\d.]+ sigma)\)$")
SPIKE_MANY = re.compile(r"^(?P<name>.+?): (?P<n>\d+) spikes, the last at (?P<t>\d{2}:\d{2}) \(up to (?P<sig>[+-][\d.]+ sigma)\)$")


def _split_items(text: str) -> list[str]:
    """Split 'a (x), b: 3 spikes, the last at ... (y), c (z)' on top-level ', ' after a ')'."""
    return [p.strip() for p in re.split(r"(?<=\)), ", text) if p.strip()]


def facts_from_summary(summary: str) -> dict[str, Any]:
    f: dict[str, Any] = {"now": None, "n_channels": 0, "abnormal": [], "normal": [], "spikes": [],
                         "feeds": [], "feeds_down": []}
    for line in summary.splitlines():
        line = line.strip()
        m = HEAD.match(line)
        if m:
            f["now"], f["n_channels"] = m["now"], int(m["n"])
            continue
        m = LINE.match(line)
        if m:
            conds = []
            for part in m["body"].split("; "):
                kind = next((k for phrase, k in KINDS if part.startswith(phrase)), "other")
                s = SINCE.search(part)
                d = DRIFT.search(part)
                src = re.match(r"no longer tracking (\S+)", part)
                conds.append({"kind": kind, "since": s["t"] if s else "unknown time", "text": part,
                              "deviation": d["dev"] if d else None, "direction": d["dir"] if d else None,
                              "sigma": d["sig"] if d else None, "source": src.group(1) if src else None,
                              "value": None})
            f["abnormal"].append({"name": m["name"], "normal": m["normal"], "conditions": conds})
            continue
        if line.startswith("Recent spikes"):
            body = line.split(": ", 1)[1].rstrip(".")
            body = re.sub(r", and spikes on \d+ more channels$", "", body)
            for item in _split_items(body):
                a, b = SPIKE_MANY.match(item), SPIKE_ONE.match(item)
                if a:
                    f["spikes"].append({"name": a["name"], "time": a["t"], "sigma": a["sig"],
                                        "count": int(a["n"]), "text": item})
                elif b:
                    f["spikes"].append({"name": b["name"], "time": b["t"], "sigma": b["sig"], "count": 1,
                                        "text": item})
            continue
        if line.startswith("Feeds:"):
            f["feeds"] = ["(feeds)"]
            for m2 in re.finditer(r"(\S+) has sent no rows since ((?:\d{4}-\d{2}-\d{2} )?\d{2}:\d{2})", line):
                f["feeds_down"].append({"name": m2.group(1), "since": m2.group(2)})
            continue
        if line.startswith("Normal: "):
            body = line[len("Normal: "):].rstrip(".")
            if not body.startswith("the other "):
                f["normal"] = [x.strip() for x in body.split(", ")]
    return f
