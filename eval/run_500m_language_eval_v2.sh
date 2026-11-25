#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat >&2 <<'USAGE'
Usage:
  bash eval/run_500m_language_eval_v2.sh CHECKPOINT OUTPUT_ROOT [MODE]

MODE:
  smoke   quick diagnostic, skips C-Eval/CMMLU unless SKIP_BOUNDARY=0
  full    full reference eval and C-Eval/CMMLU boundary checks

Environment overrides:
  TOKENIZER=data/tokenizer/zh_bpe_hf50k_cci3hq_seed20260608.json
  VALIDATION_BIN=data/processed/cci3hq_500m/pretrain_hf50k_val.bin
  REFERENCE_TASK_JSONL=eval/data/zh_reference_tasks_500m_v2.jsonl
  CEVAL_DIR=data/eval/ceval-exam
  CMMLU_DIR=data/eval/CMMLU/data
  DEVICE=cuda
  DTYPE=bfloat16
  VALIDATION_BLOCK_SIZE=1024
  SKIP_BOUNDARY=0|1

100M compatibility example:
  TOKENIZER=data/tokenizer/zh_bpe_20k.model \
  VALIDATION_BIN=data/processed/pretrain_val_tokens.bin \
  REFERENCE_TASK_JSONL=eval/data/zh_reference_tasks_v1.jsonl \
  bash eval/run_500m_language_eval_v2.sh CHECKPOINT OUT full
USAGE
  exit 2
fi

CHECKPOINT="$1"
OUTPUT_ROOT="$2"
MODE="${3:-smoke}"
TOKENIZER="${TOKENIZER:-data/tokenizer/zh_bpe_hf50k_cci3hq_seed20260608.json}"
VALIDATION_BIN="${VALIDATION_BIN:-data/processed/cci3hq_500m/pretrain_hf50k_val.bin}"
REFERENCE_TASK_JSONL="${REFERENCE_TASK_JSONL:-eval/data/zh_reference_tasks_500m_v2.jsonl}"
CEVAL_DIR="${CEVAL_DIR:-data/eval/ceval-exam}"
CMMLU_DIR="${CMMLU_DIR:-data/eval/CMMLU/data}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
VALIDATION_BLOCK_SIZE="${VALIDATION_BLOCK_SIZE:-1024}"
PYTHON="${PYTHON:-python}"

if [[ "$MODE" == "smoke" ]]; then
  LIMIT_SMALL=(--limit 5)
  LIMIT_REF=(--limit 20)
  LIMIT_MCQ=(--limit 50)
  VAL_BATCHES=(--validation-batches 10)
  PROFILES=(--profiles greedy)
  SKIP_BOUNDARY="${SKIP_BOUNDARY:-1}"
elif [[ "$MODE" == "full" ]]; then
  LIMIT_SMALL=()
  LIMIT_REF=()
  LIMIT_MCQ=()
  VAL_BATCHES=()
  PROFILES=(--profiles greedy balanced_s1 balanced_s2)
  SKIP_BOUNDARY="${SKIP_BOUNDARY:-0}"
else
  echo "[error] unknown MODE: $MODE" >&2
  exit 2
fi

for path in "$CHECKPOINT" "$TOKENIZER" "$VALIDATION_BIN" "$REFERENCE_TASK_JSONL"; do
  if [[ ! -f "$path" ]]; then
    echo "[error] required file not found: $path" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_ROOT"
COMMON_ARGS=(
  --checkpoint "$CHECKPOINT"
  --tokenizer "$TOKENIZER"
  --output-root "$OUTPUT_ROOT"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --seed 1337
)

echo "[eval] checkpoint: $CHECKPOINT"
echo "[eval] tokenizer: $TOKENIZER"
echo "[eval] validation_bin: $VALIDATION_BIN"
echo "[eval] reference: $REFERENCE_TASK_JSONL"
echo "[eval] output_root: $OUTPUT_ROOT"
echo "[eval] mode: $MODE"

"$PYTHON" eval/evaluate_lm.py \
  "${COMMON_ARGS[@]}" \
  --suite-name zh_lm_500m_v2 \
  --validation-bin "$VALIDATION_BIN" \
  --validation-block-size "$VALIDATION_BLOCK_SIZE" \
  --validation-batch-size 16 \
  "${VAL_BATCHES[@]}"

"$PYTHON" eval/evaluate_generation.py \
  "${COMMON_ARGS[@]}" \
  --suite-name zh_generation_v1 \
  --prompt-file eval/prompts/zh_generation_v1.jsonl \
  --max-new-tokens 128 \
  "${PROFILES[@]}" \
  "${LIMIT_SMALL[@]}"

"$PYTHON" eval/evaluate_generation.py \
  "${COMMON_ARGS[@]}" \
  --suite-name zh_instruction_sanity_v1 \
  --prompt-file eval/prompts/zh_instruction_sanity_v1.jsonl \
  --template dialogue \
  --max-new-tokens 64 \
  --profiles greedy \
  "${LIMIT_SMALL[@]}"

"$PYTHON" eval/evaluate_language_tasks.py \
  "${COMMON_ARGS[@]}" \
  --suite-name zh_language_tasks_v1 \
  --prompt-file eval/prompts/zh_language_tasks_v1.jsonl \
  --template special \
  --max-new-tokens 96 \
  "${LIMIT_SMALL[@]}"

"$PYTHON" eval/evaluate_reference_tasks.py \
  "${COMMON_ARGS[@]}" \
  --suite-name zh_reference_tasks_500m_v2 \
  --data-file "$REFERENCE_TASK_JSONL" \
  --template special \
  --max-new-tokens 96 \
  "${LIMIT_REF[@]}"

if [[ "$SKIP_BOUNDARY" == "1" ]]; then
  echo "[skip] C-Eval/CMMLU boundary eval skipped"
else
  if [[ ! -d "$CEVAL_DIR" ]]; then
    echo "[error] CEVAL_DIR not found: $CEVAL_DIR" >&2
    exit 1
  fi
  if [[ ! -d "$CMMLU_DIR" ]]; then
    echo "[error] CMMLU_DIR not found: $CMMLU_DIR" >&2
    exit 1
  fi
  "$PYTHON" eval/evaluate_multiple_choice.py \
    "${COMMON_ARGS[@]}" \
    --suite-name ceval_boundary_500m_v2 \
    --benchmark ceval \
    --data-dir "$CEVAL_DIR" \
    --split "${CEVAL_SPLIT:-test}" \
    "${LIMIT_MCQ[@]}"
  "$PYTHON" eval/evaluate_multiple_choice.py \
    "${COMMON_ARGS[@]}" \
    --suite-name cmmlu_boundary_500m_v2 \
    --benchmark cmmlu \
    --data-dir "$CMMLU_DIR" \
    --split "${CMMLU_SPLIT:-test}" \
    "${LIMIT_MCQ[@]}"
fi

echo "[done] language eval v2 complete: $OUTPUT_ROOT"
