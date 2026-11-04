#!/usr/bin/env bash
# Run the read-only ptdb profiler on Shannon (Phase R3). Run this YOURSELF, by hand.
#
#   mkdir -p ~/spinquest-discord-anomaly-checker && cd ~/spinquest-discord-anomaly-checker
#   tar -xzf ~/ptdb_profiler_bundle.tar.gz
#   bash scripts/shannon_profile.sh            # from the repo (or bundle) folder
#
#   TABLES=QT OUT=out/profile_qt bash scripts/shannon_profile.sh    # only some tables, separate output
#
# What it does:
#   1. Creates a NEW virtual environment in ./.venv-profile (never touches the system Python,
#      anaconda3, or any existing environment) and installs only what the profiler needs.
#   2. Asks for the database settings. The password is read without echo and is only ever an
#      environment variable of this one process; nothing is written to disk or history.
#   3. Prints the dry run (every planned statement, no connection) and waits for you to say yes.
#   4. Runs the profiler at low CPU and IO priority, with a pause after every query.
#   5. Writes <OUT>/ptdb_profile.json and ptdb_profile.md (aggregates only; OUT defaults to out/profile).
#
# Review out/profile/ptdb_profile.md yourself before sharing any of it, including in a chat.

set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
VENV=.venv-profile
export PIP_CACHE_DIR=$PWD/.pip-cache   # keep pip files inside this folder
OUT=${OUT:-out/profile}
TABLE_ARGS=()
[ -n "${TABLES:-}" ] && TABLE_ARGS=(--tables $TABLES)

if [ ! -x "$VENV/bin/python" ]; then
    echo "Creating a new virtual environment in $VENV (the system Python is not modified)"
    "$PY" -m venv "$VENV"
    "$VENV/bin/python" -m pip install --quiet --upgrade pip
    # Only the profiler's needs, pinned to the versions in requirements.txt.
    grep -E '^(numpy|pandas|PyYAML|PyMySQL|python-dateutil|tzdata|six)==' requirements.txt > "$VENV/req-profile.txt" || true
    grep -q '^six==' "$VENV/req-profile.txt" || echo 'six' >> "$VENV/req-profile.txt"
    "$VENV/bin/python" -m pip install --quiet -r "$VENV/req-profile.txt"
fi

echo
echo "=== Dry run: every planned statement, nothing is executed ==="
"$VENV/bin/python" scripts/profile_db.py --dry-run "${TABLE_ARGS[@]}"
echo

read -r -p "Database host (a read replica if there is one): " PTDB_HOST
read -r -p "Port [3306]: " PTDB_PORT
PTDB_PORT=${PTDB_PORT:-3306}
read -r -p "User (read-only account): " PTDB_USER
read -r -s -p "Password (not shown, not stored): " PTDB_PASSWORD
echo
read -r -p "Database [ptdb]: " PTDB_DATABASE
PTDB_DATABASE=${PTDB_DATABASE:-ptdb}
read -r -p "Pause between queries in seconds [0.5]: " PAUSE
PAUSE=${PAUSE:-0.5}

echo
echo "The profiler will run about 1,300 read-only queries against $PTDB_HOST/$PTDB_DATABASE,"
echo "with a $PAUSE s pause after each (roughly 15 to 30 minutes). Off-peak hours are best."
read -r -p "Type yes to start: " OK
if [ "$OK" != "yes" ]; then
    echo "Stopped. Nothing was run."
    exit 0
fi

export PTDB_HOST PTDB_PORT PTDB_USER PTDB_PASSWORD PTDB_DATABASE
NICE=(nice -n 10)
command -v ionice >/dev/null && NICE=(ionice -c 3 nice -n 10)
"${NICE[@]}" "$VENV/bin/python" scripts/profile_db.py --pause-seconds "$PAUSE" --out "$OUT" "${TABLE_ARGS[@]}"
unset PTDB_PASSWORD

echo
echo "Done: $OUT/ptdb_profile.json and $OUT/ptdb_profile.md"
echo "AGGREGATES ONLY. Review $OUT/ptdb_profile.md before sharing it with anyone, including in a chat."
