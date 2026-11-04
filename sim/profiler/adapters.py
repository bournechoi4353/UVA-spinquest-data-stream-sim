"""Database adapters for the profiler, all behind the read-only guard.

* ``SQLiteAdapter``: opens a fixture file read-only (``mode=ro``). Used in tests.
* ``MySQLAdapter``: connects to the real ptdb with credentials from the
  environment only (PTDB_HOST, PTDB_PORT, PTDB_USER, PTDB_PASSWORD,
  PTDB_DATABASE). On connect it makes the session read-only, sets the session
  time zone to UTC, and sets a per-query timeout chosen by server version.
  Only the person running the profiler uses this, by hand.
* ``DryRunAdapter``: connects to nothing. It records and prints each planned
  statement and returns no rows.

Every adapter sends every statement through ``guard.check_statement`` first.
"""

from __future__ import annotations

import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .guard import check_statement

ENV_VARS = ("PTDB_HOST", "PTDB_PORT", "PTDB_USER", "PTDB_PASSWORD", "PTDB_DATABASE")
QUERY_TIMEOUT_S = 30


def timeout_statement(version: str, seconds: int = QUERY_TIMEOUT_S) -> str:
    """The per-query timeout statement for this server.

    MariaDB: ``SET SESSION max_statement_time = <seconds>`` (seconds).
    MySQL:   ``SET SESSION max_execution_time = <milliseconds>`` (milliseconds).
    """
    if "mariadb" in version.lower():
        return f"SET SESSION max_statement_time = {int(seconds)}"
    return f"SET SESSION max_execution_time = {int(seconds) * 1000}"


def _to_us(value: Any) -> int | None:
    """A time column value (datetime or 'YYYY-MM-DD HH:MM:SS' text) as UTC microseconds."""
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(round(value.timestamp() * 1e6))


class Adapter:
    """Common interface. Subclasses implement ``_run``."""

    dialect = "generic"
    param = "?"

    def __init__(self, pause_seconds: float = 0.0):
        self.pause_seconds = pause_seconds
        self.statements: list[str] = []

    def quote(self, name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    def execute(self, sql: str, params: Sequence[Any] = ()) -> list[tuple]:
        check_statement(sql)
        self.statements.append(sql)
        rows = self._run(sql, tuple(params))
        if self.pause_seconds:
            time.sleep(self.pause_seconds)
        return rows

    def _run(self, sql: str, params: tuple) -> list[tuple]:  # pragma: no cover
        raise NotImplementedError

    def to_us(self, value: Any) -> int | None:
        return _to_us(value)

    def server_info(self) -> dict[str, Any]:
        return {"version": None, "system_time_zone": None, "now": None, "utc_now": None}

    def close(self) -> None:
        pass


class SQLiteAdapter(Adapter):
    """Read-only access to a SQLite fixture written by ``sim.db_writer``."""

    dialect = "sqlite"
    param = "?"

    def __init__(self, path: str | Path, pause_seconds: float = 0.0):
        super().__init__(pause_seconds)
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        self.con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)

    def _run(self, sql: str, params: tuple) -> list[tuple]:
        return self.con.execute(sql, params).fetchall()

    def server_info(self) -> dict[str, Any]:
        (version,) = self.execute("SELECT sqlite_version()")[0]
        return {"version": f"sqlite {version}", "system_time_zone": None, "now": None, "utc_now": None}

    def close(self) -> None:
        self.con.close()


class MySQLAdapter(Adapter):
    """The real database. Credentials come only from environment variables."""

    dialect = "mysql"
    param = "%s"

    def __init__(self, pause_seconds: float = 0.5, env: dict[str, str] | None = None):
        super().__init__(pause_seconds)
        env = os.environ if env is None else env
        missing = [v for v in ENV_VARS if not env.get(v)]
        if missing:
            raise RuntimeError(f"missing environment variable(s): {', '.join(missing)} "
                               f"(see .env.example; values are never read from files or arguments)")
        import pymysql  # imported here so tests and dry runs never need it

        self.con = pymysql.connect(
            host=env["PTDB_HOST"], port=int(env["PTDB_PORT"]), user=env["PTDB_USER"],
            password=env["PTDB_PASSWORD"], database=env["PTDB_DATABASE"],
            charset="latin1", autocommit=True, connect_timeout=15, read_timeout=60,
        )
        self.execute("SET SESSION TRANSACTION READ ONLY")
        self.execute("SET time_zone = '+00:00'")
        (version,) = self.execute("SELECT VERSION()")[0]
        self.version = str(version)
        self.execute(timeout_statement(self.version))

    def quote(self, name: str) -> str:
        return "`" + name.replace("`", "``") + "`"

    def _run(self, sql: str, params: tuple) -> list[tuple]:
        with self.con.cursor() as cur:
            cur.execute(sql, params if params else None)
            return list(cur.fetchall())

    def server_info(self) -> dict[str, Any]:
        version, tz, now, utc_now = self.execute(
            "SELECT VERSION(), @@system_time_zone, NOW(), UTC_TIMESTAMP()")[0]
        return {"version": str(version), "system_time_zone": str(tz),
                "now": str(now), "utc_now": str(utc_now)}

    def close(self) -> None:
        self.con.close()


class DryRunAdapter(Adapter):
    """Connects to nothing. Prints each planned statement and returns no rows."""

    dialect = "mysql"
    param = "%s"

    def __init__(self, printer=print):
        super().__init__(0.0)
        self.printer = printer

    def quote(self, name: str) -> str:
        return "`" + name.replace("`", "``") + "`"

    def _run(self, sql: str, params: tuple) -> list[tuple]:
        self.printer(f"  {sql}" + (f"    params={list(params)}" if params else ""))
        return []

    def server_info(self) -> dict[str, Any]:
        self.printer("  (on connect)")
        for sql in ("SET SESSION TRANSACTION READ ONLY", "SET time_zone = '+00:00'", "SELECT VERSION()",
                    timeout_statement("10.x-MariaDB"), "SELECT VERSION(), @@system_time_zone, NOW(), UTC_TIMESTAMP()"):
            self.execute(sql)
        self.printer(f"  (the timeout line is '{timeout_statement('8.0')}' if the server is MySQL)")
        return super().server_info()
