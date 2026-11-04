"""Phase R3: the read-only profiler, tested only against local SQLite fixtures."""

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sim.db_writer import write_fixture
from sim.layout_scenario import generate_layout
from sim.profiler import (DryRunAdapter, MySQLAdapter, Profiler, ProfileSettings, ReadOnlyViolation,
                          SchemaViolation, SQLiteAdapter, check_statement, timeout_statement,
                          validate_profile, write_report)
from sim.registry import load_registry

ROOT = Path(__file__).resolve().parent.parent
REG = load_registry(ROOT / "configs" / "ptdb_layout.yaml")

KNOWN = {
    "Labjack.buffer_pressure": {"baseline": 812.5, "sigma": 2.5, "ar1_phi": 0.6},
    "Flow_Rates.main_flow": {"baseline": 3.2, "sigma": 0.05},
    "QT.ti501_ai": {"baseline": 0.35, "sigma": 0.002},
}


@pytest.fixture(scope="module")
def fixture_db(tmp_path_factory):
    cfg = {"seed": 5, "start_time": "2026-02-01T00:00:00", "duration_seconds": 8 * 3600,
           "channels": KNOWN,
           "anomalies": [{"feed": "Flow_Rates", "type": "feed_outage", "start": 10000, "duration": 1200},
                         {"channel": "Labjack.thermocouple", "type": "stuck_sensor", "start": 18000,
                          "duration": 3000}]}
    run = generate_layout(cfg, REG)
    path = tmp_path_factory.mktemp("fx") / "fixture.sqlite"
    write_fixture(run, path)
    return run, path


@pytest.fixture(scope="module")
def report(fixture_db):
    _, path = fixture_db
    a = SQLiteAdapter(path)
    try:
        return Profiler(a, REG, ProfileSettings(chunks=20, chunk_rows=500, timeline_probes=400,
                                                window_hours=2)).run()
    finally:
        a.close()


# ---- the guard ---------------------------------------------------------------------
@pytest.mark.parametrize("sql", [
    "SELECT 1", "  select id from Labjack", "SHOW TABLES", "SET time_zone = '+00:00'",
    "SET SESSION TRANSACTION READ ONLY", "SELECT 1;", "/* hi */ SELECT 1", "SELECT 'a;b'",
    "SELECT VERSION(), @@system_time_zone, NOW(), UTC_TIMESTAMP()",
])
def test_guard_allows_reads(sql):
    assert check_statement(sql) == sql


@pytest.mark.parametrize("sql", [
    "INSERT INTO Labjack VALUES (1)", "UPDATE Labjack SET x = 1", "DELETE FROM Labjack",
    "DROP TABLE Labjack", "CREATE TABLE x (a int)", "ALTER TABLE Labjack ADD c int", "TRUNCATE Labjack",
    "REPLACE INTO Labjack VALUES (1)", "GRANT ALL ON *.* TO x", "CALL p()", "LOAD DATA INFILE 'x'",
    "SELECT 1; DELETE FROM Labjack", "SELECT * FROM Labjack INTO OUTFILE '/tmp/x'",
    "SELECT * FROM Labjack FOR UPDATE", "SET GLOBAL max_connections = 1", "SET @@global.x = 1",
    "-- SELECT\nDELETE FROM Labjack", "", "   ", "PRAGMA writable_schema = 1", "WITH x AS (SELECT 1) DELETE FROM y",
])
def test_guard_rejects_everything_else(sql):
    with pytest.raises(ReadOnlyViolation):
        check_statement(sql)


def test_adapter_runs_the_guard_before_the_database(fixture_db):
    _, path = fixture_db
    a = SQLiteAdapter(path)
    with pytest.raises(ReadOnlyViolation):
        a.execute("DELETE FROM Labjack")
    assert a.execute("SELECT COUNT(*) FROM Labjack")[0][0] > 0   # nothing was deleted
    a.close()


def test_sqlite_adapter_is_read_only_even_past_the_guard(fixture_db):
    _, path = fixture_db
    a = SQLiteAdapter(path)
    with pytest.raises(Exception):
        a.con.execute("DELETE FROM Labjack")
    a.close()


# ---- timeout by server version ---------------------------------------------------------
def test_timeout_statement_mariadb_in_seconds():
    assert timeout_statement("10.6.18-MariaDB-0ubuntu0.22.04.1") == "SET SESSION max_statement_time = 30"


def test_timeout_statement_mysql_in_milliseconds():
    assert timeout_statement("8.0.36") == "SET SESSION max_execution_time = 30000"


def test_mysql_adapter_needs_every_env_var_and_never_takes_arguments():
    with pytest.raises(RuntimeError, match="PTDB_PASSWORD"):
        MySQLAdapter(env={"PTDB_HOST": "h", "PTDB_PORT": "1", "PTDB_USER": "u", "PTDB_DATABASE": "d"})


# ---- dry run -------------------------------------------------------------------------
def test_dry_run_executes_nothing_and_connects_to_nothing():
    lines = []
    a = DryRunAdapter(printer=lines.append)
    Profiler(a, REG, ProfileSettings()).dry_run()
    text = "\n".join(lines)
    assert "SELECT MIN(id), MAX(id) FROM `Labjack`" in text
    assert "SET SESSION TRANSACTION READ ONLY" in text
    assert "COUNT(" not in text
    assert all(check_statement(s) for s in a.statements)


def test_dry_run_script_never_imports_a_driver():
    code = ("import sys, runpy; sys.argv=['profile_db.py','--dry-run']; "
            "runpy.run_path('scripts/profile_db.py', run_name='__main__')"
            " if False else None\n"
            "sys.argv=['profile_db.py','--dry-run']\n"
            "import scripts_profile\n")
    del code
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys, runpy\n"
         "sys.argv = ['profile_db.py', '--dry-run']\n"
         "try:\n    runpy.run_path('scripts/profile_db.py', run_name='__main__')\n"
         "except SystemExit:\n    pass\n"
         "print('PYMYSQL_LOADED' if 'pymysql' in sys.modules else 'NO_DRIVER')\n"],
        cwd=ROOT, capture_output=True, text=True, check=True)
    assert "DRY RUN" in result.stdout
    assert result.stdout.strip().endswith("NO_DRIVER")


# ---- the statistics ------------------------------------------------------------------
def test_profile_recovers_baseline_sigma_and_spacing(report):
    lab = report["tables"]["Labjack"]
    bp = lab["channels"]["buffer_pressure"]
    assert bp["median"] == pytest.approx(812.5, abs=0.3)
    assert bp["mad_sigma"] == pytest.approx(2.5, rel=0.08)
    assert bp["noise_lag1"] == pytest.approx(0.6, abs=0.1)
    mf = report["tables"]["Flow_Rates"]["channels"]["main_flow"]
    assert mf["median"] == pytest.approx(3.2, abs=0.01)
    assert mf["noise_mad_sigma"] == pytest.approx(0.05, rel=0.1)
    ft = report["tables"]["QT"]["channels"]["ti501_ai"]
    assert ft["mad_sigma"] == pytest.approx(0.002, rel=0.1)
    for name in ("Labjack", "QT"):
        assert report["tables"][name]["mean_spacing_s"] == pytest.approx(REG.table(name).nominal_period_s, rel=0.03)
    assert lab["id_min_is_1"] is True and lab["id_step_1_share"] == 1.0


def test_profile_finds_the_outage_and_the_stuck_sensor(report):
    fr = report["tables"]["Flow_Rates"]
    assert fr["largest_gaps"], "the 1200 s outage must be reported"
    assert fr["largest_gaps"][0]["length_s"] == pytest.approx(1200, abs=60)
    assert fr["timeline"]["estimated_downtime_s"] == pytest.approx(1200, abs=60)
    # Chunks only sample part of the stuck window, but a healthy channel's longest run is about 2.
    tc = report["tables"]["Labjack"]["channels"]["thermocouple"]
    healthy = report["tables"]["Labjack"]["channels"]["buffer_pressure"]
    assert tc["longest_repeat"] >= 10 and healthy["longest_repeat"] <= 3


def test_profile_reports_correlations_with_channel_keys(report):
    corr = report["correlations"]
    assert len(corr["diff_top30"]) == 30 and len(corr["level_top30"]) == 30
    assert all("." in p["a"] and "." in p["b"] for p in corr["diff_top30"])


def test_late_and_duplicate_rows_show_up_in_ordering_checks(tmp_path):
    run = generate_layout({"seed": 2, "start_time": "2026-01-01T00:00:00", "duration_seconds": 3600}, REG)
    path = tmp_path / "f.sqlite"
    write_fixture(run, path, late_fraction=0.02)
    a = SQLiteAdapter(path)
    rep = Profiler(a, REG, ProfileSettings(chunks=5, chunk_rows=400, timeline_probes=50, window_hours=1)).run()
    a.close()
    assert rep["tables"]["Labjack"]["backward_ts_share"] > 0


# ---- output safety -------------------------------------------------------------------
def test_output_passes_the_allowlist_and_has_the_banner(report, tmp_path):
    jpath, mpath = write_report(report, tmp_path)
    data = json.loads(jpath.read_text())
    validate_profile(data)
    assert data["banner"] == "AGGREGATES ONLY. Review before sharing."
    assert mpath.read_text().startswith("# AGGREGATES ONLY. Review before sharing.")


def test_allowlist_rejects_extra_keys_and_long_lists(report):
    import copy
    bad = copy.deepcopy(report)
    bad["tables"]["Labjack"]["raw_rows"] = [1, 2, 3]
    with pytest.raises(SchemaViolation):
        validate_profile(bad)
    bad = copy.deepcopy(report)
    bad["tables"]["Labjack"]["largest_gaps"] = [{"start": None, "end": None, "length_s": 1, "exact": True}] * 21
    with pytest.raises(SchemaViolation):
        validate_profile(bad)
    bad = copy.deepcopy(report)
    bad["tables"]["Labjack"]["channels"]["buffer_pressure"]["values"] = [1.0]
    with pytest.raises(SchemaViolation):
        validate_profile(bad)


def test_no_raw_reading_appears_in_the_output(fixture_db, report, tmp_path):
    run, _ = fixture_db
    jpath, mpath = write_report(report, tmp_path)
    text = jpath.read_text() + mpath.read_text()
    rng = np.random.default_rng(0)
    # Every number the report contains on purpose (aggregates). A raw reading that happens to equal
    # one of them at the printed precision is a coincidence, not a leak.
    allowed = set()

    def collect(x):
        if isinstance(x, dict):
            for v in x.values():
                collect(v)
        elif isinstance(x, list):
            for v in x:
                collect(v)
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            allowed.add(float(x))
    collect(report)
    checked = 0
    for feed in run.feeds.values():
        for col in feed.columns:
            v = feed.values[col]
            v = v[np.isfinite(v)]
            for x in rng.choice(v, 200, replace=False):
                # Equal to a reported aggregate within the printed rounding (6 significant digits in the .md).
                if any(a is not None and abs(float(x) - a) <= 1e-5 * max(abs(a), 1e-12) for a in allowed):
                    continue
                # Full-precision forms a row dump would use (rounded forms can match aggregates).
                for rep in (repr(float(x)), repr(float(np.float32(x))), str(np.float32(x))):
                    pattern = r"(?<![\d.])" + re.escape(rep) + r"(?![\d])"   # whole numbers only
                    assert not re.search(pattern, text), f"raw reading {rep} leaked"
                checked += 1
    assert checked > 1000
    # Individual timestamps: only gap starts and ends, first_ts/last_ts, and window bounds appear.
    stamps = sum(len(t["largest_gaps"]) * 2 + 2 for t in report["tables"].values())
    assert text.count("+00:00") <= 2 * (stamps + 2 * len(report["correlations"]["windows"]))


def test_gitignore_covers_profile_output():
    entries = (ROOT / ".gitignore").read_text().splitlines()
    assert "out/profile/" in entries and "out/" in entries
