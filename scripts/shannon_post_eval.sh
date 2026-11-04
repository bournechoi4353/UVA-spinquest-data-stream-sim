#!/usr/bin/env bash
# After training: test the fine-tuned model end to end, and on channel names it has not seen.
#
#   setsid nohup bash scripts/shannon_post_eval.sh > out/logs/post_eval.log 2>&1 < /dev/null &
#
#   1. end-to-end test with the fine-tuned model answering (two seeds)   -> out/e2e_model_s*/report.md
#   2. the v2 held-out set (QT channel names, never in training data v1) -> out/eval/finetuned_v2_holdout.jsonl
# Same isolation as shannon_train.sh: .venv-train only, caches inside this folder.

set -euo pipefail
cd "$(dirname "$0")/.."
unset PYTHONPATH
export HF_HOME=$PWD/.hf-home PIP_CACHE_DIR=$PWD/.pip-cache TRITON_CACHE_DIR=$PWD/.triton-cache TOKENIZERS_PARALLELISM=false
PY=.venv-train/bin/python
ADAPTER=${ADAPTER:-out/model/adapter}

echo "== $(date) 1. end-to-end with the fine-tuned model"
for seed in 2026 99; do
    $PY scripts/run_end_to_end.py --hours 12 --seed "$seed" --answerer model --adapter "$ADAPTER" \
        --out "out/e2e_model_s$seed" || true
done
echo "== $(date) 2. held-out set with QT names"
$PY -m training.evaluate --data out/qa_v2 --split test_holdout --answerer model --adapter "$ADAPTER" \
    --limit 800 --out out/eval/finetuned_v2_holdout.jsonl | tee out/eval/summary_v2.txt
$PY -m training.evaluate --data out/qa_v2 --split test_holdout --answerer template --limit 800 \
    | tee -a out/eval/summary_v2.txt
echo "== $(date) POST_DONE"
