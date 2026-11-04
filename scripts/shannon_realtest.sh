#!/usr/bin/env bash
# Full test on REAL data, read-only, on Shannon. Run this yourself (it asks for the database password).
#
#   cd ~/spinquest-discord-anomaly-checker
#   bash scripts/shannon_realtest.sh
#
# Steps (nothing in the database is written or changed: every statement passes the read-only guard,
# and the session is SET SESSION TRANSACTION READ ONLY):
#   1. profile QT (aggregates only)                         -> out/profile_qt/
#   2. set QT's nominal period in the layout from that profile (the layout file in this folder only)
#   3. run the monitor on recent real data: calibrate on the first CAL_HOURS, then watch the next
#      TEST_HOURS, through the live reader                  -> out/monitor/real/
#   4. ask the fine-tuned model a few questions about the latest real summary
#                                                           -> out/monitor/real/answers.txt
#
# Settings: SINCE (UTC start, default 2026-08-14T19:10:00, the start of the most recent logger run),
# CAL_HOURS (default 12), TEST_HOURS (default 24), PAUSE (seconds between queries, default 0.5).
# Review out/profile_qt/ptdb_profile.md and out/monitor/real/ before sharing anything from them.

set -euo pipefail
cd "$(dirname "$0")/.."
unset PYTHONPATH
export HF_HOME=$PWD/.hf-home PIP_CACHE_DIR=$PWD/.pip-cache TRITON_CACHE_DIR=$PWD/.triton-cache TOKENIZERS_PARALLELISM=false

SINCE=${SINCE:-2026-08-14T19:10:00}
CAL_HOURS=${CAL_HOURS:-12}
TEST_HOURS=${TEST_HOURS:-24}
PAUSE=${PAUSE:-0.5}
PROFILE_PY=.venv-profile/bin/python
TRAIN_PY=.venv-train/bin/python
[ -x "$PROFILE_PY" ] || { echo "Missing .venv-profile: run bash scripts/shannon_profile.sh once first."; exit 2; }

read -r -p "Database host [twist.phys.virginia.edu]: " PTDB_HOST
PTDB_HOST=${PTDB_HOST:-twist.phys.virginia.edu}
read -r -p "Port [3306]: " PTDB_PORT
PTDB_PORT=${PTDB_PORT:-3306}
read -r -p "User [uvaspin]: " PTDB_USER
PTDB_USER=${PTDB_USER:-uvaspin}
read -r -s -p "Password (not shown, not stored): " PTDB_PASSWORD
echo
PTDB_DATABASE=${PTDB_DATABASE:-ptdb}
export PTDB_HOST PTDB_PORT PTDB_USER PTDB_PASSWORD PTDB_DATABASE
NICE=(nice -n 10)
command -v ionice >/dev/null && NICE=(ionice -c 3 nice -n 10)

echo "== 1. profile QT (read-only, aggregates only)"
"${NICE[@]}" "$PROFILE_PY" scripts/profile_db.py --tables QT --pause-seconds "$PAUSE" --out out/profile_qt

echo "== 2. QT period from the profile"
"$PROFILE_PY" - <<'PYEOF'
import json, re
from pathlib import Path
prof = json.loads(Path("out/profile_qt/ptdb_profile.json").read_text())
period = prof["tables"]["QT"].get("mean_spacing_s")
if period:
    p = Path("configs/ptdb_layout.yaml")
    s = p.read_text()
    s = re.sub(r"(  - name: QT\n(?:    .*\n)*?    nominal_period_s: )1\.0  # PROVISIONAL[^\n]*",
               lambda m: f"{m.group(1)}{period:.4g}  # measured mean spacing while running (QT profile)", s)
    p.write_text(s)
    print(f"QT nominal_period_s = {period:.4g} s")
else:
    print("QT profile has no spacing; layout unchanged")
PYEOF

echo "== 3. monitor on real data (read-only), calibrate $CAL_HOURS h then watch $TEST_HOURS h from $SINCE"
"${NICE[@]}" "$PROFILE_PY" scripts/run_monitor.py --mysql --since "$SINCE" --until-hours $((CAL_HOURS + TEST_HOURS)) \
    --calibrate-minutes $((CAL_HOURS * 60)) --every-minutes 30 --name real
unset PTDB_PASSWORD

echo "== 4. questions to the fine-tuned model about the latest real summary"
if [ -d out/model/adapter ] && [ -x "$TRAIN_PY" ]; then
    : > out/monitor/real/answers.txt
    for q in "Is anything abnormal right now?" "Which channels are normal?" "Are all feeds reporting?" \
             "Any spikes recently?" "Why is that happening?"; do
        "$TRAIN_PY" -m bot.discord_bot --monitor-dir out/monitor/real --adapter out/model/adapter --ask "$q" \
            | tail -1 | sed "s/^/Q: $q\n/" | tee -a out/monitor/real/answers.txt
    done
else
    echo "No fine-tuned adapter yet (out/model/adapter); skipping."
fi
echo
echo "Done. Read out/monitor/real/summary.txt, snapshots.jsonl and answers.txt."
echo "Review them before sharing anything, including in a chat."
