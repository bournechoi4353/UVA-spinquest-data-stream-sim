"""Write the profile as JSON and as a human-readable Markdown file.

Both files start with the banner "AGGREGATES ONLY. Review before sharing."
The report is validated against the allowlist schema before either file is written.
"""

from __future__ import annotations

import json
from pathlib import Path

from .core import BANNER
from .schema import validate_profile


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def render_markdown(report: dict) -> str:
    lines = [f"# {BANNER}", "", "ptdb profile (aggregate statistics only)", ""]
    srv = report.get("server", {})
    lines += ["## Server", "", "| field | value |", "|---|---|"]
    lines += [f"| {k} | {_fmt(v)} |" for k, v in srv.items()]
    smp = report.get("sampling", {})
    lines += ["", "## Sampling", "", "| field | value |", "|---|---|"]
    lines += [f"| {k} | {_fmt(v)} |" for k, v in smp.items()]
    for name, t in report["tables"].items():
        lines += ["", f"## Table {name}", "", "| field | value |", "|---|---|"]
        for k in ("id_min", "id_max", "row_estimate", "id_min_is_1", "first_ts", "last_ts", "id_step_1_share",
                  "backward_ts_share", "duplicate_ts_share", "mean_spacing_s", "gaps_gt_5x_median", "long_step_share",
                  "sampled_hours"):
            if k in t:
                lines.append(f"| {k} | {_fmt(t[k])} |")
        if "spacing_s" in t:
            lines.append("| spacing_s | " + ", ".join(f"{k} {_fmt(v)}" for k, v in t["spacing_s"].items()) + " |")
            lines.append("| spacing_share | " + ", ".join(f"{k}: {_fmt(v)}" for k, v in t["spacing_share"].items())
                         + " |")
        if "timeline" in t:
            lines.append(f"| timeline | {t['timeline']['probes']} probes, estimated downtime "
                         f"{_fmt(t['timeline']['estimated_downtime_s'])} s |")
        if t.get("largest_gaps"):
            lines += ["", "Largest gaps (the only individual timestamps in this report):", "",
                      "| start | end | length s | exact |", "|---|---|---|---|"]
            lines += [f"| {g['start']} | {g['end']} | {_fmt(g['length_s'])} | {g['exact']} |"
                      for g in t["largest_gaps"]]
        chans = t.get("channels", {})
        if chans:
            keys = list(next(iter(chans.values())).keys())
            lines += ["", "| channel | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
            for c, s in chans.items():
                lines.append(f"| {c} | " + " | ".join(_fmt(s[k]) for k in keys) + " |")
    corr = report.get("correlations", {})
    for title, key in (("first differences", "diff_top30"), ("levels", "level_top30")):
        if corr.get(key):
            lines += ["", f"## Top channel pairs by correlation of {title}", "", "| a | b | r |", "|---|---|---|"]
            lines += [f"| {p['a']} | {p['b']} | {_fmt(p['r'])} |" for p in corr[key]]
    lines += ["", f"**{BANNER}**", ""]
    return "\n".join(lines)


def write_report(report: dict, out_dir: str | Path) -> tuple[Path, Path]:
    validate_profile(report)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jpath, mpath = out / "ptdb_profile.json", out / "ptdb_profile.md"
    jpath.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    mpath.write_text(render_markdown(report), encoding="utf-8")
    return jpath, mpath
