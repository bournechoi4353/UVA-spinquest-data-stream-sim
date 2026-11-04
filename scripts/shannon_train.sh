#!/usr/bin/env bash
# Fine-tune and evaluate the answer model on Shannon. Start it only when the GPU is free.
#
#   cd ~/spinquest-discord-anomaly-checker
#   nvidia-smi                                   # check nobody else is using the GPU
#   setsid nohup bash scripts/shannon_train.sh > out/logs/full_run.log 2>&1 < /dev/null &
#   tail -f out/logs/full_run.log                # Ctrl+C stops watching, not the run
#
# Environment: the isolated .venv-train in this folder only (built from /usr/bin/python3, no
# conda, no system packages). PYTHONPATH is cleared, and the model and pip caches stay in this
# folder (.hf-home, .pip-cache), so nothing outside the folder changes.
#
# Steps (each writes to out/):
#   1. fine-tune Qwen/Qwen3.5-4B with LoRA on out/qa/train.jsonl  -> out/model/adapter
#   2. evaluate the template baseline, the base model and the fine-tuned model on
#      test and test_holdout                                     -> out/eval/*.jsonl, out/eval/summary.txt
#
# Stop it at any time with:  pkill -f "[t]raining.train_lora"; pkill -f "[t]raining.evaluate"

set -euo pipefail
cd "$(dirname "$0")/.."
unset PYTHONPATH
export HF_HOME=$PWD/.hf-home PIP_CACHE_DIR=$PWD/.pip-cache TRITON_CACHE_DIR=$PWD/.triton-cache TOKENIZERS_PARALLELISM=false
PY=.venv-train/bin/python
EPOCHS=${EPOCHS:-1}
mkdir -p out/eval out/logs

echo "== $(date) GPU before start"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv

echo "== $(date) 1. fine-tune ($EPOCHS epoch(s))"
$PY -m training.train_lora --data out/qa --out out/model --epochs "$EPOCHS"

echo "== $(date) 2. evaluate"
: > out/eval/summary.txt
for split in test test_holdout; do
    $PY -m training.evaluate --data out/qa --split "$split" --answerer template \
        --out "out/eval/template_$split.jsonl" | tee -a out/eval/summary.txt
    $PY -m training.evaluate --data out/qa --split "$split" --answerer model --limit 400 \
        --out "out/eval/base_$split.jsonl" | tee -a out/eval/summary.txt
    $PY -m training.evaluate --data out/qa --split "$split" --answerer model --adapter out/model/adapter \
        --limit 800 --out "out/eval/finetuned_$split.jsonl" | tee -a out/eval/summary.txt
done
echo "== $(date) RUN_DONE"
