#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/chinese-llm-from-scratch

if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  source /root/miniconda3/etc/profile.d/conda.sh
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
else
  echo "ERROR: conda.sh not found"
  exit 1
fi
conda activate chinese-llm

MIXED_CORPUS="data/processed/mixed_corpus.jsonl"
TOKENIZER_PREFIX="data/tokenizer/zh_bpe_20k"
INPUT_SENTENCE_SIZE="${INPUT_SENTENCE_SIZE:-10000000}"
ANALYZE_SAMPLE_LINES="${ANALYZE_SAMPLE_LINES:-100000}"
NUM_THREADS="${NUM_THREADS:-16}"

mkdir -p logs/main20k results/main20k data/tokenizer

echo "========== 20K tokenizer from existing mixed corpus =========="
echo "started_at=$(date -Is)"
echo "cwd=$(pwd)"
echo "mixed_corpus=${MIXED_CORPUS}"
echo "tokenizer_prefix=${TOKENIZER_PREFIX}"
echo "input_sentence_size=${INPUT_SENTENCE_SIZE}"
echo "analyze_sample_lines=${ANALYZE_SAMPLE_LINES}"
echo "num_threads=${NUM_THREADS}"
echo

if [ ! -s "${MIXED_CORPUS}" ]; then
  echo "ERROR: missing or empty ${MIXED_CORPUS}"
  exit 1
fi

echo "[0/2] Input file"
ls -lh "${MIXED_CORPUS}"
echo

echo "[1/2] Train 20K tokenizer"
rm -f "${TOKENIZER_PREFIX}.model" "${TOKENIZER_PREFIX}.vocab" "${TOKENIZER_PREFIX}_stats.json"
/usr/bin/time -v -o logs/main20k/05_train_tokenizer_20k.time \
  python data/scripts/train_tokenizer.py \
    --input "${MIXED_CORPUS}" \
    --model-prefix "${TOKENIZER_PREFIX}" \
    --vocab-size 20000 \
    --input-sentence-size "${INPUT_SENTENCE_SIZE}" \
    --num-threads "${NUM_THREADS}" \
    --analyze-sample-lines "${ANALYZE_SAMPLE_LINES}" \
  2>&1 | tee logs/main20k/05_train_tokenizer_20k.log

echo
echo "[2/2] Tokenizer outputs"
ls -lh "${TOKENIZER_PREFIX}.model" "${TOKENIZER_PREFIX}.vocab" "${TOKENIZER_PREFIX}_stats.json"
cat "${TOKENIZER_PREFIX}_stats.json"
echo

echo "finished_at=$(date -Is)"
echo "DONE: 20K tokenizer is ready."
