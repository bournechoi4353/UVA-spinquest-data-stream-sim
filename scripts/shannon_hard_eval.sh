#!/usr/bin/env bash
# The hard question sets on Shannon (GPU): the fine-tuned model and the base model.
#
#   cd ~/spinquest-discord-anomaly-checker
#   nvidia-smi                                     # check nobody else is using the GPU
#   setsid nohup bash scripts/shannon_hard_eval.sh > out/logs/hard_eval.log 2>&1 < /dev/null &
#   tail -f out/logs/hard_eval.log                 # about 15 to 25 minutes; ends with HARD_DONE
#
#   test_hard   60 hand-written hard questions (training/hard_set.py)
#   test_noisy  the 802 held-out questions, scrambled (typos, shorthand names, filler words)
# Results: out/eval/summary_hard.txt, out/eval/{finetuned,base}_test_{hard,noisy}.jsonl and
# out/eval/hard_failures.txt (every failed answer of the fine-tuned model, with the failed check).
# Same isolation as shannon_train.sh: .venv-train only, caches inside this folder.

set -euo pipefail
cd "$(dirname "$0")/.."
unset PYTHONPATH
export HF_HOME=$PWD/.hf-home PIP_CACHE_DIR=$PWD/.pip-cache TRITON_CACHE_DIR=$PWD/.triton-cache TOKENIZERS_PARALLELISM=false
PY=.venv-train/bin/python
ADAPTER=${ADAPTER:-out/model/adapter}
mkdir -p out/eval out/logs

echo "== $(date) build the hard sets"
$PY -m training.hard_set --out out/qa_hard
: > out/eval/summary_hard.txt
for split in test_hard test_noisy; do
    echo "== $(date) $split: fine-tuned"
    $PY -m training.evaluate --data out/qa_hard --split "$split" --answerer model --adapter "$ADAPTER" \
        --max-new-tokens 400 --out "out/eval/finetuned_$split.jsonl" | tee -a out/eval/summary_hard.txt
    echo "== $(date) $split: fine-tuned answers through the fact guard (bot/guard.py)"
    $PY -m training.evaluate --data out/qa_hard --split "$split" --rescore "out/eval/finetuned_$split.jsonl" \
        --guard --out "out/eval/finetuned_guard_$split.jsonl" | tee -a out/eval/summary_hard.txt
    echo "== $(date) $split: base model (first 400)"
    $PY -m training.evaluate --data out/qa_hard --split "$split" --answerer model --limit 400 \
        --max-new-tokens 400 --out "out/eval/base_$split.jsonl" | tee -a out/eval/summary_hard.txt
done
$PY - <<'PYEOF' > out/eval/hard_failures.txt
import json
for split in ("test_hard", "test_noisy"):
    rows = [json.loads(x) for x in open(f"out/eval/finetuned_{split}.jsonl", encoding="utf-8")]
    bad = [r for r in rows if not r["pass"] or r["hallucinated"]]
    print(f"===== {split}: {len(bad)} of {len(rows)} failed or hallucinated")
    for r in bad:
        print(f"[{r['intent']}] Q: {r['question']}")
        print(f"   A: {r['answer']}")
        print(f"   failed: {r.get('failed')} bad: {r.get('bad')}")
PYEOF
echo "== $(date) HARD_DONE"
