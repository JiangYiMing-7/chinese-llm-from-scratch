#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/sft_500m_hf50k_normal_v2.yaml}"
LOG_ROOT="${LOG_ROOT:-logs/sft_500m}"
PYTHON="${PYTHON:-python}"
mkdir -p "$LOG_ROOT"

if [[ ! -f "$CONFIG" ]]; then
  echo "[error] config not found: $CONFIG" >&2
  exit 2
fi

"$PYTHON" - "$CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml

cfg_path = Path(sys.argv[1])
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
required = {
    "train.init_from": cfg["train"].get("init_from"),
    "data.tokenizer_path": cfg["data"].get("tokenizer_path"),
    "data.train_path": cfg["data"].get("train_path"),
    "data.val_path": cfg["data"].get("val_path"),
}
missing = [f"{name} -> {path}" for name, path in required.items() if not path or not Path(path).exists()]
if missing:
    print("[error] required SFT inputs are missing:", file=sys.stderr)
    for item in missing:
        print("  " + item, file=sys.stderr)
    raise SystemExit(1)
model = cfg["model"]
train = cfg["train"]
data = cfg["data"]
print("[config]")
print(f"  config: {cfg_path}")
print(f"  init_from: {train.get('init_from')}")
print(f"  tokenizer: {data.get('tokenizer_path')}")
print(f"  train: {data.get('train_path')}")
print(f"  val: {data.get('val_path')}")
print(f"  vocab_size: {model.get('vocab_size')}")
print(f"  layers/d_model/heads: {model.get('n_layers')}/{model.get('d_model')}/{model.get('n_heads')}")
print(f"  batch_size: {train.get('batch_size')}")
print(f"  gradient_accumulation_steps: {train.get('gradient_accumulation_steps', 1)}")
print(f"  effective_batch_size: {int(train.get('batch_size', 1)) * int(train.get('gradient_accumulation_steps', 1))}")
print(f"  precision/device: {train.get('precision')}/{train.get('device')}")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "[warn] nvidia-smi not found"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="$LOG_ROOT/$(basename "$CONFIG" .yaml)_$STAMP.log"
echo "[run] python finetune/train_sft.py --config $CONFIG"
echo "[log] $LOG_PATH"
"$PYTHON" finetune/train_sft.py --config "$CONFIG" 2>&1 | tee "$LOG_PATH"
