#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat >&2 <<'USAGE'
Usage:
  bash finetune/run_sft_and_eval.sh CONFIG TOKENIZER VALIDATION_BIN [OUTPUT_ROOT] [MODE]

Environment variables are forwarded to eval/run_eval_suite.sh:
  DEVICE=cuda|cpu|auto
  DTYPE=float32|bfloat16
  CEVAL_DIR=/path/to/ceval
  CMMLU_DIR=/path/to/cmmlu
  EXTERNAL_JSONL=/path/file.jsonl

Example:
  DEVICE=cuda DTYPE=bfloat16 bash finetune/run_sft_and_eval.sh \
    configs/sft_main_smoke.yaml \
    data/tokenizer/zh_bpe_20k.model \
    data/processed/pretrain_val_tokens.bin \
    results/evals \
    smoke
USAGE
  exit 2
fi

CONFIG="$1"
TOKENIZER="$2"
VALIDATION_BIN="$3"
OUTPUT_ROOT="${4:-results/evals}"
MODE="${5:-smoke}"

python finetune/train_sft.py --config "$CONFIG"

RUN_DIR="$(
  python -c 'import pathlib, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
train = cfg["train"]
if train.get("run_dir"):
    print(train["run_dir"])
else:
    root = pathlib.Path(train.get("run_dir_root", "results/sft_runs"))
    name = train.get("run_name", pathlib.Path(sys.argv[1]).stem)
    dirs = sorted(root.glob(name + "_*"), key=lambda p: p.stat().st_mtime)
    if not dirs:
        raise SystemExit("no SFT run dir found")
    print(dirs[-1])' "$CONFIG"
)"

SFT_CHECKPOINT="$(
  python -c 'import json, pathlib, sys
summary = pathlib.Path(sys.argv[1]) / "summary.json"
payload = json.loads(summary.read_text(encoding="utf-8"))
print(payload["best_checkpoint"])' "$RUN_DIR"
)"

echo "SFT checkpoint: $SFT_CHECKPOINT"
bash eval/run_eval_suite.sh "$SFT_CHECKPOINT" "$TOKENIZER" "$VALIDATION_BIN" "$OUTPUT_ROOT" "$MODE"

