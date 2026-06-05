#!/usr/bin/env bash
set -euo pipefail

cd /root/tl/chinese-llm-from-scratch

source /root/miniconda3/etc/profile.d/conda.sh
conda activate chinese-llm

mkdir -p logs/main20k results/main20k data/processed data/tokenizer

LOCAL_CORPUS="data/processed/pilot_clean.jsonl"
CCI_LOCAL="data/raw/cci3_hq_raw.jsonl"
MIXED_CORPUS="data/processed/mixed_corpus.jsonl"
TOKENIZER_PREFIX="data/tokenizer/zh_bpe_20k"

TOTAL_CHARS=6000000000
CCI_RATIO=0.98
SHUFFLE_BUFFER=20000

echo "========== 20K local-CCI data + tokenizer pipeline =========="
echo "started_at=$(date -Is)"
echo "cwd=$(pwd)"
echo "local_corpus=${LOCAL_CORPUS}"
echo "cci_local=${CCI_LOCAL}"
echo "mixed_corpus=${MIXED_CORPUS}"
echo "tokenizer_prefix=${TOKENIZER_PREFIX}"
echo "total_chars=${TOTAL_CHARS}"
echo "cci_ratio=${CCI_RATIO}"
echo "shuffle_buffer=${SHUFFLE_BUFFER}"
echo

if [ ! -s "${LOCAL_CORPUS}" ]; then
  echo "ERROR: missing or empty ${LOCAL_CORPUS}"
  exit 1
fi

if [ ! -s "${CCI_LOCAL}" ]; then
  echo "ERROR: missing or empty ${CCI_LOCAL}"
  exit 1
fi

echo "[0/3] Input files"
ls -lh "${LOCAL_CORPUS}" "${CCI_LOCAL}"
echo

echo "[1/3] Mix corpus without cross MinHash dedup"
rm -f "${MIXED_CORPUS}.tmp"
/usr/bin/time -v -o logs/main20k/04_mix_main20k_localcci_nodedup.time \
  python data/scripts/mix_datasets.py \
    --local-corpus "${LOCAL_CORPUS}" \
    --cci-local "${CCI_LOCAL}" \
    --output "${MIXED_CORPUS}" \
    --total-chars "${TOTAL_CHARS}" \
    --cci-ratio "${CCI_RATIO}" \
    --shuffle-buffer "${SHUFFLE_BUFFER}" \
    --no-cross-dedup \
  2>&1 | tee logs/main20k/04_mix_main20k_localcci_nodedup.log

echo
echo "[1/3] Mixed corpus output"
ls -lh "${MIXED_CORPUS}"
du -sh "${MIXED_CORPUS}"
echo

echo "[2/3] Train 20K tokenizer"
rm -f "${TOKENIZER_PREFIX}.model" "${TOKENIZER_PREFIX}.vocab" "${TOKENIZER_PREFIX}_stats.json"
/usr/bin/time -v -o logs/main20k/05_train_tokenizer_20k.time \
  python data/scripts/train_tokenizer.py \
    --input "${MIXED_CORPUS}" \
    --model-prefix "${TOKENIZER_PREFIX}" \
    --vocab-size 20000 \
    --input-sentence-size 10000000 \
    --analyze-sample-lines 100000 \
  2>&1 | tee logs/main20k/05_train_tokenizer_20k.log

echo
echo "[2/3] Tokenizer outputs"
ls -lh "${TOKENIZER_PREFIX}.model" "${TOKENIZER_PREFIX}.vocab" "${TOKENIZER_PREFIX}_stats.json"
cat "${TOKENIZER_PREFIX}_stats.json"
echo

echo "[3/3] Write short summary"
python - <<'PY' | tee results/main20k/mix_and_tokenizer_20k_summary.txt
import json
import pathlib
import re

print("20K local-CCI mix + tokenizer summary")

for path in [
    "logs/main20k/04_mix_main20k_localcci_nodedup.time",
    "logs/main20k/05_train_tokenizer_20k.time",
]:
    p = pathlib.Path(path)
    if not p.exists():
        continue
    text = p.read_text(errors="ignore")
    elapsed = re.search(r"Elapsed \(wall clock\) time .*: (.*)", text)
    cpu = re.search(r"Percent of CPU this job got: (.*)", text)
    rss = re.search(r"Maximum resident set size \(kbytes\): (.*)", text)
    print(
        f"- {p.name}: elapsed={elapsed.group(1) if elapsed else '?'}, "
        f"cpu={cpu.group(1) if cpu else '?'}, "
        f"max_rss_kb={rss.group(1) if rss else '?'}"
    )

for path in [
    "data/processed/mixed_corpus.jsonl",
    "data/tokenizer/zh_bpe_20k.model",
    "data/tokenizer/zh_bpe_20k.vocab",
    "data/tokenizer/zh_bpe_20k_stats.json",
]:
    p = pathlib.Path(path)
    print(f"- {path}: {p.stat().st_size if p.exists() else 'missing'} bytes")

stats_path = pathlib.Path("data/tokenizer/zh_bpe_20k_stats.json")
if stats_path.exists():
    stats = json.loads(stats_path.read_text())
    print("vocab_size=", stats.get("vocab_size"))
    print("tokens_per_char=", stats.get("tokens_per_char"))
    print("unk_rate=", stats.get("unk_rate"))
    print("roundtrip_rate=", stats.get("roundtrip_rate"))
PY

echo
echo "finished_at=$(date -Is)"
echo "DONE: mixed corpus and 20K tokenizer are ready."
