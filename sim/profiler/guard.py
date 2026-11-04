"""Read-only guard: the only gate every profiler query passes through.

A statement is allowed only if, after comments are stripped, it

* is a single statement (no second statement after a ';'),
* starts with SELECT, SHOW, or SET,
* is not a SET that reaches beyond the session (GLOBAL, PERSIST, PERSIST_ONLY, @@global.),
* does not write through a SELECT (INTO OUTFILE, INTO DUMPFILE, INTO @var) and
  does not take locks (FOR UPDATE, FOR SHARE, LOCK IN SHARE MODE).

Anything else raises ReadOnlyViolation before it reaches a database driver.
"""

from __future__ import annotations

import re

ALLOWED_PREFIXES = ("SELECT", "SHOW", "SET")

_FORBIDDEN = [
    (re.compile(r"\bINTO\s+(OUTFILE|DUMPFILE|@)", re.I), "SELECT ... INTO writes data"),
    (re.compile(r"\bINTO\b", re.I), "INTO is not allowed"),
    (re.compile(r"\bFOR\s+(UPDATE|SHARE)\b", re.I), "locking reads are not allowed"),
    (re.compile(r"\bLOCK\s+IN\s+SHARE\s+MODE\b", re.I), "locking reads are not allowed"),
    (re.compile(r"\b(GLOBAL|PERSIST|PERSIST_ONLY)\b|@@global\.", re.I), "only session settings may be SET"),
]
_COMMENTS = re.compile(r"/\*.*?\*/|--[^\n]*|#[^\n]*", re.S)
_STRINGS = re.compile(r"'(?:[^'\\]|\\.|'')*'|\"(?:[^\"\\]|\\.)*\"")


class ReadOnlyViolation(PermissionError):
    """Raised for any statement that is not plainly read-only."""


def check_statement(sql: str) -> str:
    """Return ``sql`` unchanged if it is allowed, otherwise raise ReadOnlyViolation."""
    if not isinstance(sql, str):
        raise ReadOnlyViolation(f"statement must be a string, got {type(sql).__name__}")
    # Blank out string literals first, so a ';' or keyword inside quotes cannot confuse the checks,
    # then remove comments.
    body = _COMMENTS.sub(" ", _STRINGS.sub("''", sql)).strip()
    if body.endswith(";"):
        body = body[:-1].rstrip()
    if not body:
        raise ReadOnlyViolation("empty statement")
    if ";" in body:
        raise ReadOnlyViolation("only one statement per call is allowed")
    first = body.split(None, 1)[0].upper()
    if first not in ALLOWED_PREFIXES:
        raise ReadOnlyViolation(f"{first} statements are not allowed (only SELECT, SHOW, SET)")
    for pattern, why in _FORBIDDEN:
        if pattern.search(body):
            raise ReadOnlyViolation(why)
    return sql
