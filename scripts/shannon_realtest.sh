#!/usr/bin/env bash
# Full test on REAL data, read-only, on Shannon. Run this yourself (it asks for the database password).
#
# The database is reached through an SSH tunnel. In another terminal on Shannon, first run:
#   ssh -N -L 3307:127.0.0.1:3306 ptgroup@twist.phys.virginia.edu
# and at the prompts below enter host 127.0.0.1, port 3307.
#
#   cd ~/spinquest-discord-anomaly-checker
#   bash scripts/shannon_realtest.sh
#
# Steps (nothing in the database is written or changed: every statement passes the read-only guard,
# and the session is SET SESSION TRANSACTION READ ONLY):
#   1. profile QT (aggregates only), skipped if out/profile_qt/ptdb_profile.json exists
#                                                           -> out/profile_qt/
#   2. set QT's nominal period in the layout from that profile (the layout file in this folder only)
#   3. run the monitor on real data: calibrate on the first CAL_HOURS, then watch the next
#      TEST_HOURS, through the live reader                  -> out/monitor/$NAME/
#   4. digest of every alarm (counts, rates, durations)     -> out/monitor/$NAME/digest.txt
#   5. ask the fine-tuned model a few questions about the latest real summary
#                                                           -> out/monitor/$NAME/answers.txt
#
# Settings (environment variables):
#   SINCE       UTC start, default 2025-11-01T00:00:00. Oct 2025 to Jan 2026 is the longest stretch
#               where Flow_Rates, Labjack and QT all logged normally. (The data after 2026-08-14 is
#               only 2 h 19 min long, logged at a third of the usual rate, and ptdb ends 2026-08-17
#               21:29 UTC.)
#   CAL_HOURS   default 72     TEST_HOURS  default 144 (6 days): the test week is 2025-11-04 to 11-10,
#               the same as the first run (real_2025-11-03, 24 h calibration), so the two compare directly
#   NAME        default real_<SINCE date>
#   PROFILE_QT  set to 1 to profile QT again even if out/profile_qt exists
# Expect about 15 to 40 minutes for the defaults (about 1.6 million rows, read in pages of 5,000
# with a 0.5 s pause after each query; the detectors process about 17,000 events per second).
# Review out/monitor/$NAME/ before sharing anything from it, including in a chat.

set -euo pipefail
cd "$(dirname "$0")/.."
unset PYTHONPATH
export HF_HOME=$PWD/.hf-home PIP_CACHE_DIR=$PWD/.pip-cache TRITON_CACHE_DIR=$PWD/.triton-cache TOKENIZERS_PARALLELISM=false

SINCE=${SINCE:-2025-11-01T00:00:00}
CAL_HOURS=${CAL_HOURS:-72}
TEST_HOURS=${TEST_HOURS:-144}
PAUSE=${PAUSE:-0.5}
NAME=${NAME:-real_${SINCE%%T*}}
PROFILE_PY=.venv-profile/bin/python
TRAIN_PY=.venv-train/bin/python
OUTDIR=out/monitor/$NAME
[ -x "$PROFILE_PY" ] || { echo "Missing .venv-profile: run bash scripts/shannon_profile.sh once first."; exit 2; }
TEST_START=$("$PROFILE_PY" -c "from datetime import datetime, timedelta; print((datetime.fromisoformat('$SINCE') + timedelta(hours=$CAL_HOURS)).isoformat())")
echo "Window: calibrate $SINCE + ${CAL_HOURS} h, then test until +$((CAL_HOURS + TEST_HOURS)) h (test starts $TEST_START). Output: $OUTDIR"

read -r -p "Database host [127.0.0.1 = the tunnel]: " PTDB_HOST
PTDB_HOST=${PTDB_HOST:-127.0.0.1}
read -r -p "Port [3307 = the tunnel]: " PTDB_PORT
PTDB_PORT=${PTDB_PORT:-3307}
read -r -p "User [uvaspin]: " PTDB_USER
PTDB_USER=${PTDB_USER:-uvaspin}
read -r -s -p "Password (not shown, not stored): " PTDB_PASSWORD
echo
PTDB_DATABASE=${PTDB_DATABASE:-ptdb}
export PTDB_HOST PTDB_PORT PTDB_USER PTDB_PASSWORD PTDB_DATABASE
NICE=(nice -n 10)
command -v ionice >/dev/null && NICE=(ionice -c 3 nice -n 10)

if [ -f out/profile_qt/ptdb_profile.json ] && [ "${PROFILE_QT:-0}" != 1 ]; then
    echo "== 1. QT profile exists (out/profile_qt); skipping. Set PROFILE_QT=1 to redo it."
else
    echo "== 1. profile QT (read-only, aggregates only)"
    "${NICE[@]}" "$PROFILE_PY" scripts/profile_db.py --tables QT --pause-seconds "$PAUSE" --out out/profile_qt
fi

echo "== 2. QT period from the profile"
"$PROFILE_PY" - <<'PYEOF'
import json, re
from pathlib import Path
prof = json.loads(Path("out/profile_qt/ptdb_profile.json").read_text())
period = prof["tables"]["QT"].get("mean_spacing_s")
p = Path("configs/ptdb_layout.yaml")
s = p.read_text()
if period and "PROVISIONAL" in s:
    s = re.sub(r"(  - name: QT\n(?:    .*\n)*?    nominal_period_s: )1\.0  # PROVISIONAL[^\n]*",
               lambda m: f"{m.group(1)}{period:.4g}  # measured mean spacing while running (QT profile)", s)
    p.write_text(s)
    print(f"QT nominal_period_s = {period:.4g} s")
else:
    print("QT period already set (or no spacing in the profile); layout unchanged")
PYEOF

echo "== 3. monitor on real data (read-only)"
"${NICE[@]}" "$PROFILE_PY" scripts/run_monitor.py --mysql --since "$SINCE" --until-hours $((CAL_HOURS + TEST_HOURS)) \
    --calibrate-minutes $((CAL_HOURS * 60)) --every-minutes 60 --name "$NAME"
unset PTDB_PASSWORD

echo "== 4. alarm digest"
"$PROFILE_PY" scripts/summarize_alarms.py "$OUTDIR" --test-start "$TEST_START" --out "$OUTDIR/digest.txt"

echo "== 5. questions to the fine-tuned model about the latest real summary"
if [ -d out/model/adapter ] && [ -x "$TRAIN_PY" ]; then
    : > "$OUTDIR/answers.txt"
    for q in "Is anything abnormal right now?" "Which channels are normal?" "Are all feeds reporting?" \
             "Any spikes recently?" "Why is that happening?"; do
        "$TRAIN_PY" -m bot.discord_bot --monitor-dir "$OUTDIR" --adapter out/model/adapter --ask "$q" \
            | tail -1 | sed "s/^/Q: $q\n/" | tee -a "$OUTDIR/answers.txt"
    done
else
    echo "No fine-tuned adapter yet (out/model/adapter); skipping."
fi
echo
echo "Done. Read $OUTDIR/digest.txt, coverage.txt, summary.txt and answers.txt."
echo "Review them before sharing anything, including in a chat."
