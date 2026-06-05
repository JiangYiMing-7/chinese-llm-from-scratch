# 中文多选评测报告

## 配置

- benchmark: `ceval`
- split: `test`
- checkpoint: `results/runs/pilot_20k_20x_bf16_20260524_20260524_091754/checkpoints/best.pt`
- dtype/device: `bfloat16` / `cuda`
- length normalize: `False`

## 总体结果

| examples | correct | accuracy |
|---:|---:|---:|
| 50 | 13 | 0.2600 |

## 分学科

| subject | examples | accuracy |
|---|---:|---:|
| accountant | 50 | 0.2600 |
