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


def test_gitignore_keeps_secrets_and_real_output_out_of_git():
    entries = {line.strip() for line in (ROOT / ".gitignore").read_text().splitlines()}
    for required in (".env", "out/", "out/profile/"):
        assert required in entries, f".gitignore must contain {required}"


def test_env_example_lists_names_without_values():
    lines = [ln.strip() for ln in (ROOT / ".env.example").read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    names = [ln.split("=", 1)[0] for ln in lines]
    assert names == ["PTDB_HOST", "PTDB_PORT", "PTDB_USER", "PTDB_PASSWORD", "PTDB_DATABASE"]
    assert all(ln.endswith("=") for ln in lines), ".env.example must not contain values"
