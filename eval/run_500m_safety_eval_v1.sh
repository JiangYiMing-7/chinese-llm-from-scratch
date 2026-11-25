#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:?usage: bash eval/run_500m_safety_eval_v1.sh CHECKPOINT [OUTPUT_ROOT] [TASK_FILE]}"
OUTPUT_ROOT="${2:-results/evals/safety_500m_v1}"
TASK_FILE="${3:-eval/data/zh_safety_tasks_v1.jsonl}"
TOKENIZER="${TOKENIZER:-data/tokenizer/zh_bpe_hf50k_cci3hq_seed20260608.json}"
PYTHON="${PYTHON:-python}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-auto}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
LIMIT="${LIMIT:-}"

ARGS=(
  --checkpoint "$CHECKPOINT"
  --tokenizer "$TOKENIZER"
  --output-root "$OUTPUT_ROOT"
  --task-file "$TASK_FILE"
  --dtype "$DTYPE"
  --device "$DEVICE"
  --max-new-tokens "$MAX_NEW_TOKENS"
)

if [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi

"$PYTHON" eval/evaluate_safety_tasks_v1.py "${ARGS[@]}"
