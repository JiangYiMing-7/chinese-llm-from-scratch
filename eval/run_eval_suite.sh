#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat >&2 <<'USAGE'
Usage:
  bash eval/run_eval_suite.sh CHECKPOINT TOKENIZER VALIDATION_BIN [OUTPUT_ROOT] [MODE]

Environment variables:
  DEVICE=cuda|cpu|auto          default: auto
  DTYPE=float32|bfloat16        default: float32
  CEVAL_DIR=/path/to/ceval      optional, expects split CSV files
  CMMLU_DIR=/path/to/cmmlu      optional, expects split CSV files
  EXTERNAL_JSONL=/path/file     optional, JSONL external Chinese LM data
  REFERENCE_TASK_JSONL=/path/file
                               optional, larger reference-answer language tasks
  VALIDATION_BLOCK_SIZE=256|512|1024
                               default: 256 for Tiny/Pilot/Main fair scale comparison

MODE:
  smoke                         small quick run
  full                          full validation and full prompt sets
USAGE
  exit 2
fi

CHECKPOINT="$1"
TOKENIZER="$2"
VALIDATION_BIN="$3"
OUTPUT_ROOT="${4:-results/evals}"
MODE="${5:-smoke}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-float32}"
VALIDATION_BLOCK_SIZE="${VALIDATION_BLOCK_SIZE:-256}"

COMMON_ARGS=(
  --checkpoint "$CHECKPOINT"
  --tokenizer "$TOKENIZER"
  --output-root "$OUTPUT_ROOT"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --seed 1337
)
EXT_ARGS=()
if [[ -n "${EXTERNAL_JSONL:-}" ]]; then
  EXT_ARGS=(--external-jsonl "$EXTERNAL_JSONL")
fi

if [[ "$MODE" == "smoke" ]]; then
  VAL_BATCHES=(--validation-batches 10)
  LIMIT_SMALL=(--limit 5)
  LIMIT_MCQ=(--limit 50)
  GEN_PROFILES=(--profiles greedy)
elif [[ "$MODE" == "full" ]]; then
  VAL_BATCHES=()
  LIMIT_SMALL=()
  LIMIT_MCQ=()
  GEN_PROFILES=(--profiles greedy balanced_s1 balanced_s2)
else
  echo "unknown MODE: $MODE" >&2
  exit 2
fi

python eval/evaluate_lm.py \
  "${COMMON_ARGS[@]}" \
  --suite-name zh_lm_v1 \
  --validation-bin "$VALIDATION_BIN" \
  --validation-block-size "$VALIDATION_BLOCK_SIZE" \
  --validation-batch-size 16 \
  "${VAL_BATCHES[@]}" \
  "${EXT_ARGS[@]}"

if [[ -n "${CEVAL_DIR:-}" ]]; then
  python eval/evaluate_multiple_choice.py \
    "${COMMON_ARGS[@]}" \
    --benchmark ceval \
    --data-dir "$CEVAL_DIR" \
    --split test \
    "${LIMIT_MCQ[@]}"
else
  echo "[skip] CEVAL_DIR is not set"
fi

if [[ -n "${CMMLU_DIR:-}" ]]; then
  python eval/evaluate_multiple_choice.py \
    "${COMMON_ARGS[@]}" \
    --benchmark cmmlu \
    --data-dir "$CMMLU_DIR" \
    --split test \
    "${LIMIT_MCQ[@]}"
else
  echo "[skip] CMMLU_DIR is not set"
fi

python eval/evaluate_generation.py \
  "${COMMON_ARGS[@]}" \
  --suite-name zh_generation_v1 \
  --prompt-file eval/prompts/zh_generation_v1.jsonl \
  --max-new-tokens 128 \
  "${GEN_PROFILES[@]}" \
  "${LIMIT_SMALL[@]}"

python eval/evaluate_generation.py \
  "${COMMON_ARGS[@]}" \
  --suite-name zh_instruction_sanity_v1 \
  --prompt-file eval/prompts/zh_instruction_sanity_v1.jsonl \
  --template dialogue \
  --max-new-tokens 64 \
  --profiles greedy \
  "${LIMIT_SMALL[@]}"

python eval/evaluate_language_tasks.py \
  "${COMMON_ARGS[@]}" \
  --suite-name zh_language_tasks_v1 \
  --prompt-file eval/prompts/zh_language_tasks_v1.jsonl \
  --template special \
  --max-new-tokens 96 \
  "${LIMIT_SMALL[@]}"

if [[ -n "${REFERENCE_TASK_JSONL:-}" ]]; then
  python eval/evaluate_reference_tasks.py \
    "${COMMON_ARGS[@]}" \
    --suite-name zh_reference_tasks_v1 \
    --data-file "$REFERENCE_TASK_JSONL" \
    --template special \
    --max-new-tokens 96 \
    "${LIMIT_SMALL[@]}"
else
  echo "[skip] REFERENCE_TASK_JSONL is not set"
fi

python eval/evaluate_prompt_injection_focus.py \
  "${COMMON_ARGS[@]}" \
  --suite-name zh_prompt_injection_focus_v1 \
  --prompt-file eval/prompts/zh_prompt_injection_focus_v1.jsonl \
  --max-new-tokens 96 \
  "${LIMIT_SMALL[@]}"

python eval/evaluate_generation.py \
  "${COMMON_ARGS[@]}" \
  --suite-name multilingual_probe_v1 \
  --prompt-file eval/prompts/multilingual_probe_v1.jsonl \
  --max-new-tokens 64 \
  --profiles greedy \
  "${LIMIT_SMALL[@]}"
