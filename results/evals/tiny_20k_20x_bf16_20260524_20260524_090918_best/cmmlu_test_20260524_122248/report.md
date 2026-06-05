# 中文多选评测报告

## 配置

- benchmark: `cmmlu`
- split: `test`
- checkpoint: `results/runs/tiny_20k_20x_bf16_20260524_20260524_090918/checkpoints/best.pt`
- dtype/device: `bfloat16` / `cuda`
- length normalize: `False`

## 总体结果

| examples | correct | accuracy |
|---:|---:|---:|
| 50 | 11 | 0.2200 |

## 分学科

| subject | examples | accuracy |
|---|---:|---:|
| agronomy | 50 | 0.2200 |
