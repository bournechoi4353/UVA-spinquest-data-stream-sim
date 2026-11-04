"""Read-only, aggregates-only profiler for ptdb (Phase R3). See core.py for the sampling plan."""

from .adapters import DryRunAdapter, MySQLAdapter, SQLiteAdapter, timeout_statement
from .core import BANNER, Profiler, ProfileSettings
from .guard import ReadOnlyViolation, check_statement
from .report import render_markdown, write_report
from .schema import SchemaViolation, validate_profile

__all__ = ["BANNER", "DryRunAdapter", "MySQLAdapter", "Profiler", "ProfileSettings", "ReadOnlyViolation",
           "SQLiteAdapter", "SchemaViolation", "check_statement", "render_markdown", "timeout_statement",
           "validate_profile", "write_report"]
