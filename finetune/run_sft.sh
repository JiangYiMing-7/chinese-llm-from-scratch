#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat >&2 <<'USAGE'
Usage:
  bash finetune/run_sft.sh CONFIG

Example:
  DEVICE=cuda bash finetune/run_sft.sh configs/sft_main_smoke.yaml
USAGE
  exit 2
fi

CONFIG="$1"
python finetune/train_sft.py --config "$CONFIG"

