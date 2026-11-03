"""Batch generation: splits, manifest, fractions, reproducibility, categories."""

import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from sim.batch import (
    ALL_TYPES, SPLITS, assign_splits, format_summary, generate_batch, make_plan, scenario_seeds,
)
from sim.io import read_labels
from sim.scenario import parse_scenario

ROOT = Path(__file__).resolve().parent.parent
N = 40
SHORT = (0.2, 0.4)  # hours, keeps the test batch small and fast


@pytest.fixture(scope="module")
def batch(tmp_path_factory):
    out = tmp_path_factory.mktemp("batch")
    manifest = generate_batch(n=N, master_seed=3, out_dir=out, duration_range_hours=SHORT)
    return out, manifest


def tree_hash(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def labels_of(out, run):
    return read_labels(out / run["path"] / "labels.json")


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def test_split_sizes_are_80_10_10(batch):
    _, manifest = batch
    assert manifest["split_counts"] == {"train": 32, "val": 4, "test": 4}


def test_splits_are_disjoint_by_run_and_by_seed(batch):
    out, manifest = batch
    by_split = {s: [r for r in manifest["runs"] if r["split"] == s] for s in SPLITS}
    names = [set(r["run"] for r in rs) for rs in by_split.values()]
    seeds = [set(r["seed"] for r in rs) for rs in by_split.values()]
    for sets in (names, seeds):
        assert sum(len(s) for s in sets) == len(set().union(*sets)) == N
    # On disk too: each run folder name exists under exactly one split folder.
    on_disk = Counter(p.name for s in SPLITS for p in (out / s).iterdir())
    assert max(on_disk.values()) == 1 and len(on_disk) == N


def test_scenario_seeds_are_unique_and_depend_only_on_master_seed():
    _, a = scenario_seeds(11, 500)
    _, b = scenario_seeds(11, 500)
    _, c = scenario_seeds(12, 500)
    sa, sb, sc = [s for _, s in a], [s for _, s in b], [s for _, s in c]
    assert sa == sb and len(set(sa)) == 500
    assert not set(sa) & set(sc)


def test_split_assignment_is_deterministic_and_keeps_every_split_non_empty():
    first = assign_splits(30, np.random.default_rng(5), (80, 10, 10))
    again = assign_splits(30, np.random.default_rng(5), (80, 10, 10))
    other = assign_splits(30, np.random.default_rng(6), (80, 10, 10))
    assert first == again and first != other
    assert Counter(first) == {"train": 24, "val": 3, "test": 3}
    assert set(assign_splits(3, np.random.default_rng(0), (98, 1, 1))) == set(SPLITS)


# ---------------------------------------------------------------------------
# Manifest vs disk
# ---------------------------------------------------------------------------
def test_manifest_counts_match_files_on_disk(batch):
    out, manifest = batch
    assert manifest["n"] == len(manifest["runs"]) == N
    for split in SPLITS:
        dirs = sorted(p.name for p in (out / split).iterdir())
        listed = sorted(r["run"] for r in manifest["runs"] if r["split"] == split)
        assert dirs == listed
        assert manifest["split_counts"][split] == len(dirs)
    for r in manifest["runs"]:
        for name in ("data.csv", "labels.json", "config_used.yaml"):
            assert (out / r["path"] / name).is_file()
    assert json.loads((out / "manifest.json").read_text()) == manifest


def test_manifest_anomaly_types_match_the_labels_on_disk(batch):
    out, manifest = batch
    for r in manifest["runs"]:
        counts = Counter(l["type"] for l in labels_of(out, r))
        assert r["anomaly_counts"] == dict(sorted(counts.items()))
        assert r["anomaly_types"] == sorted(counts)
        assert r["n_anomalies"] == sum(counts.values())


def test_summary_counts_match_the_files(batch):
    out, manifest = batch
    expected = {t: Counter() for t in ALL_TYPES}
    for r in manifest["runs"]:
        for l in labels_of(out, r):
            expected[l["type"]][r["split"]] += 1
    for t in ALL_TYPES:
        row = manifest["summary"]["events_per_type"][t]
        assert [row[s] for s in SPLITS] == [expected[t][s] for s in SPLITS]
        assert row["total"] == sum(expected[t].values())


def test_missing_types_are_reported_not_hidden(batch):
    _, manifest = batch
    missing = manifest["summary"]["types_missing_from_split"]
    text = format_summary(manifest)
    for split, types in missing.items():
        for t in types:
            assert f"{split}: " in text and t in text
    assert ("WARNING" in text) == any(missing.values())


# ---------------------------------------------------------------------------
# Fractions
# ---------------------------------------------------------------------------
def test_clean_fraction_is_respected(batch):
    out, manifest = batch
    clean = [r for r in manifest["runs"] if r["category"] == "clean"]
    assert len(clean) / N == pytest.approx(0.25, abs=0.03)
    assert all(r["n_anomalies"] == 0 and labels_of(out, r) == [] for r in clean)
    assert all(r["n_anomalies"] > 0 for r in manifest["runs"] if r["category"] != "clean")


@pytest.mark.parametrize("n, clean, hard, multi", [(200, 0.25, 0.2, 0.25), (100, 0.5, 0.3, 0.1), (60, 0.0, 0.5, 0.5)])
def test_category_counts(n, clean, hard, multi):
    plan = make_plan(n, np.random.default_rng(1), clean, hard, multi)
    counts = Counter(p.category for p in plan)
    n_non_clean = n - counts["clean"]
    assert counts["clean"] == pytest.approx(clean * n, abs=1)
    assert counts["hard"] == pytest.approx(hard * n_non_clean, abs=1)  # hard-frac is of the NON-CLEAN runs
    assert counts["multi"] == pytest.approx(min(multi * n, n_non_clean - counts["hard"]), abs=1)
    assert sum(counts.values()) == n


def test_primary_types_are_handed_out_evenly():
    plan = make_plan(200, np.random.default_rng(1), 0.25, 0.2, 0.25)
    singles = Counter(p.primary for p in plan if p.category == "single")
    assert set(singles) == set(ALL_TYPES) and max(singles.values()) - min(singles.values()) <= 1


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def test_same_master_seed_reproduces_the_same_batch(batch, tmp_path):
    out, _ = batch
    generate_batch(n=N, master_seed=3, out_dir=tmp_path, duration_range_hours=SHORT)
    assert tree_hash(out) == tree_hash(tmp_path)


def test_different_master_seed_gives_a_different_batch(batch, tmp_path):
    out, _ = batch
    generate_batch(n=N, master_seed=4, out_dir=tmp_path, duration_range_hours=SHORT)
    assert tree_hash(out)["manifest.json"] != tree_hash(tmp_path)["manifest.json"]


def test_existing_batch_is_protected_unless_overwrite(tmp_path):
    kwargs = dict(n=6, master_seed=1, out_dir=tmp_path, duration_range_hours=SHORT)
    generate_batch(**kwargs)
    (tmp_path / "keep_me.txt").write_text("notes")
    with pytest.raises(FileExistsError, match="--overwrite"):
        generate_batch(**kwargs)
    generate_batch(**kwargs, overwrite=True)
    assert (tmp_path / "keep_me.txt").exists()  # only the batch's own files are replaced


# ---------------------------------------------------------------------------
# Scenario content
# ---------------------------------------------------------------------------
def test_every_config_is_valid_and_matches_its_manifest_entry(batch):
    import yaml

    out, manifest = batch
    for r in manifest["runs"]:
        cfg = parse_scenario(yaml.safe_load((out / r["path"] / "config_used.yaml").read_text()))
        assert cfg.seed == r["seed"]
        assert [c.name for c in cfg.channels] == r["channels"]
        assert 0.2 * 3600 * 0.9 <= cfg.duration_seconds <= 0.4 * 3600 * 1.1
        assert len(cfg.channels) >= 2


def test_multi_runs_have_overlapping_anomalies(batch):
    out, manifest = batch
    multi = [r for r in manifest["runs"] if r["category"] == "multi"]
    assert multi
    for r in multi:
        spans = [(datetime.fromisoformat(l["start"]),
                  datetime.fromisoformat(l["end"]) if l["end"] else datetime.max) for l in labels_of(out, r)]
        assert len(spans) >= 2
        assert any(a[0] <= b[1] and b[0] <= a[1] for i, a in enumerate(spans) for b in spans[i + 1:])


def test_decoupling_always_comes_with_a_source_driver_and_a_coupled_channel(batch):
    out, manifest = batch
    for r in manifest["runs"]:
        labels = labels_of(out, r)
        if any(l["type"] == "decoupling" for l in labels):
            assert {"polarization", "nmr_signal"} <= set(r["channels"])
            assert any(l["channel"] == "polarization" and l["type"] in ("linear_drift", "step_shift") for l in labels)


def test_anomalies_start_after_the_calibration_margin(batch):
    out, manifest = batch
    for r in manifest["runs"]:
        for l in labels_of(out, r):
            assert l["params"]["start_offset_seconds"] >= 0.1 * r["duration_seconds"] - 5.0


def test_hard_cases_follow_their_recipes():
    """Build many hard scenarios directly and check the recipe each kind promises."""
    from sim.batch import HARD_KINDS, PlanEntry, build_scenario

    seen = set()
    for i in range(80):
        kind = HARD_KINDS[i % 4]
        seen.add(kind)
        rng = np.random.default_rng(i)
        cfg = build_scenario(rng, i, PlanEntry(i, "hard", kind), (1.0, 3.0))
        parse_scenario(cfg)  # valid
        a = cfg["anomalies"]
        if kind == "slow_drift":
            assert len(a) == 1 and a[0]["type"] == "linear_drift" and 3 <= abs(a[0]["magnitude"]) <= 5
            assert a[0]["duration"] >= 0.39 * cfg["duration_seconds"] - 5
        elif kind == "noise_like_spikes":
            assert 1 <= len(a) <= 3 and all(x["type"] == "spike" and 3.5 <= abs(x["magnitude"]) <= 5 for x in a)
        elif kind == "spike_then_step":
            assert [x["type"] for x in a] == ["spike", "step_shift"] and a[0]["channel"] == a[1]["channel"]
            assert 0 < a[1]["start"] - a[0]["start"] <= 30
        else:
            assert len(a) == 1 and a[0]["type"] == "stuck_sensor"
    assert seen == set(HARD_KINDS)


# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------
def test_make_batch_help_documents_hard_frac_and_errors_are_readable(tmp_path):
    script = str(ROOT / "scripts" / "make_batch.py")
    help_text = subprocess.run([sys.executable, script, "--help"], capture_output=True, text=True).stdout
    assert "NON-CLEAN" in help_text and "--master-seed" in help_text and "--clean-frac" in help_text
    bad = subprocess.run([sys.executable, script, "--n", "2", "--out", str(tmp_path)], capture_output=True, text=True)
    assert bad.returncode == 2 and "at least 3" in bad.stderr
