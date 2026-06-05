# 中文多选评测报告

## 配置

- benchmark: `cmmlu`
- split: `test`
- checkpoint: `results/runs/main_20260523_005535/checkpoints/best.pt`
- dtype/device: `bfloat16` / `cuda`
- length normalize: `False`

## 总体结果

| examples | correct | accuracy |
|---:|---:|---:|
| 50 | 14 | 0.2800 |

## 分学科

| subject | examples | accuracy |
|---|---:|---:|
| agronomy | 50 | 0.2800 |
