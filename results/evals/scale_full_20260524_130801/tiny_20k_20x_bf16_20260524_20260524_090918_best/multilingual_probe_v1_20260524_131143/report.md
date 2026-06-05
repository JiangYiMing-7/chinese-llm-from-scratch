# 中文生成与指令小测报告

## 配置

- prompt file: `eval/prompts/multilingual_probe_v1.jsonl`
- checkpoint: `results/runs/tiny_20k_20x_bf16_20260524_20260524_090918/checkpoints/best.pt`
- profiles: `greedy`

## 分组汇总

| suite | category | profile | examples | auto pass | CJK | repeat-4 | tokens | tok/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| multilingual_probe | code_mixed | greedy | 1 | n/a | 0.000 | 0.966 | 64.0 | 652.04 |
| multilingual_probe | english_completion | greedy | 1 | n/a | 0.000 | 0.760 | 64.0 | 125.02 |
| multilingual_probe | mixed_language | greedy | 1 | n/a | 0.000 | 0.927 | 64.0 | 632.36 |
| multilingual_probe | translation | greedy | 1 | 0.0% | 0.173 | 0.731 | 64.0 | 661.24 |

完整生成内容保存在 `generations.jsonl`；人工评分请填写 `manual_review.csv`。
