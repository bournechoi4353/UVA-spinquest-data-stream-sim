"""The whole pipeline on realistic simulated data: fixture -> reader -> detectors -> summary -> answers."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_end_to_end_pipeline_finds_every_planted_anomaly(tmp_path):
    out = tmp_path / "e2e"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_end_to_end.py"), "--hours", "12", "--seed", "2026",
         "--calibrate-hours", "24", "--out", str(out), "--min-answer-pass", "0.8"],
        cwd=ROOT, capture_output=True, text=True)
    report = json.loads((out / "report.json").read_text())
    assert report["synthetic"] is True
    missed = [(lb["type"], lb["channel"]) for lb in report["labels"] if not lb["detected"]]
    assert missed == []
    assert len(report["false_alarms"]) <= 1
    assert report["reader"]["too_late"] == 0 and report["reader"]["duplicates_dropped"] > 0
    assert report["answers"]["pass_rate"] >= 0.8
    assert report["answers"]["hallucination_rate"] == 0.0
    assert (out / "report.md").read_text().startswith("# End-to-end test (SIMULATED data")
    assert result.returncode in (0, 1)
