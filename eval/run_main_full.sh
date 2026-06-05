#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source /root/miniconda3/etc/profile.d/conda.sh
conda activate chinese-llm

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/evals/main_full_${RUN_ID}}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
VALIDATION_BLOCK_SIZE="${VALIDATION_BLOCK_SIZE:-1024}"
TOKENIZER="${TOKENIZER:-data/tokenizer/zh_bpe_20k.model}"
VALIDATION_BIN="${VALIDATION_BIN:-data/processed/pretrain_val_tokens.bin}"
CEVAL_DIR="${CEVAL_DIR:-data/eval/ceval-exam}"
CMMLU_DIR="${CMMLU_DIR:-data/eval/CMMLU/data}"

export DEVICE DTYPE VALIDATION_BLOCK_SIZE CEVAL_DIR CMMLU_DIR

mkdir -p "$OUTPUT_ROOT"/logs

MANIFEST="$OUTPUT_ROOT/run_manifest.txt"
{
  echo "run_id=$RUN_ID"
  echo "purpose=Main 0.5B-token vs Main 2B-token full-context evaluation"
  echo "created_at=$(date -Iseconds)"
  echo "cwd=$(pwd)"
  echo "device=$DEVICE"
  echo "dtype=$DTYPE"
  echo "validation_block_size=$VALIDATION_BLOCK_SIZE"
  echo "tokenizer=$TOKENIZER"
  echo "validation_bin=$VALIDATION_BIN"
  echo "ceval_dir=$CEVAL_DIR"
  echo "cmmlu_dir=$CMMLU_DIR"
  echo
  echo "[git]"
  git rev-parse HEAD 2>/dev/null || true
  git status --short 2>/dev/null || true
  echo
  echo "[data_hash_files]"
  find data/eval -maxdepth 1 -type f -name 'eval_data_sha256_*.txt' -printf '%f\n' 2>/dev/null | sort || true
} > "$MANIFEST"

run_suite() {
  local name="$1"
  local checkpoint="$2"
  local log="$OUTPUT_ROOT/logs/${name}.log"
  echo "[$(date -Iseconds)] START $name" | tee -a "$OUTPUT_ROOT/logs/run.log"
  bash eval/run_eval_suite.sh \
    "$checkpoint" \
    "$TOKENIZER" \
    "$VALIDATION_BIN" \
    "$OUTPUT_ROOT" \
    full 2>&1 | tee "$log"
  echo "[$(date -Iseconds)] END $name" | tee -a "$OUTPUT_ROOT/logs/run.log"
}

run_suite "main_0p5b_tokens" \
  "results/runs/main_20260523_005535/checkpoints/best.pt"
run_suite "main_2b_tokens" \
  "results/runs/main_2b_bf16_fromscratch_bs16_20260523_20260523_182938/checkpoints/best.pt"

python eval/summarize_eval_run.py "$OUTPUT_ROOT" | tee "$OUTPUT_ROOT/summary_index.md"

echo "output_root: $OUTPUT_ROOT"

