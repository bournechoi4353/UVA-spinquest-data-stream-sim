"""Repo-wide rule: no em dashes anywhere."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EM_DASH = chr(0x2014)
SKIP_DIRS = {".venv", ".git", "out", "__pycache__", ".pytest_cache"}
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".txt", ".json", ".csv", ".gitignore"}


def test_no_em_dashes_in_the_repo():
    offenders = []
    for path in ROOT.rglob("*"):
        if path.is_dir() or SKIP_DIRS & set(path.relative_to(ROOT).parts):
            continue
        if path.suffix in TEXT_SUFFIXES or path.name == ".gitignore":
            if EM_DASH in path.read_text(errors="ignore"):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
