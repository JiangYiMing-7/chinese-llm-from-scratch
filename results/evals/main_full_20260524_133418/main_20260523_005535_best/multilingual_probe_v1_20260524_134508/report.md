# 中文生成与指令小测报告

## 配置

- prompt file: `eval/prompts/multilingual_probe_v1.jsonl`
- checkpoint: `results/runs/main_20260523_005535/checkpoints/best.pt`
- profiles: `greedy`

## 分组汇总

| suite | category | profile | examples | auto pass | CJK | repeat-4 | tokens | tok/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| multilingual_probe | code_mixed | greedy | 1 | n/a | 0.000 | 0.824 | 64.0 | 230.74 |
| multilingual_probe | english_completion | greedy | 1 | n/a | 0.000 | 0.587 | 64.0 | 89.30 |
| multilingual_probe | mixed_language | greedy | 1 | n/a | 0.945 | 0.705 | 64.0 | 224.03 |
| multilingual_probe | translation | greedy | 1 | 0.0% | 0.183 | 0.519 | 64.0 | 228.78 |

完整生成内容保存在 `generations.jsonl`；人工评分请填写 `manual_review.csv`。
