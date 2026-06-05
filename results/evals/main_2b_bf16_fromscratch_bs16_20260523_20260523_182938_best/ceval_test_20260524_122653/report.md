# 中文多选评测报告

## 配置

- benchmark: `ceval`
- split: `test`
- checkpoint: `results/runs/main_2b_bf16_fromscratch_bs16_20260523_20260523_182938/checkpoints/best.pt`
- dtype/device: `bfloat16` / `cuda`
- length normalize: `False`

## 总体结果

| examples | correct | accuracy |
|---:|---:|---:|
| 50 | 11 | 0.2200 |

## 分学科

| subject | examples | accuracy |
|---|---:|---:|
| accountant | 50 | 0.2200 |
