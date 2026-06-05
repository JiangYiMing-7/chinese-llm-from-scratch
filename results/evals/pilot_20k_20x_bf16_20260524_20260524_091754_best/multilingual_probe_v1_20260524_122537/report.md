# 中文生成与指令小测报告

## 配置

- prompt file: `eval/prompts/multilingual_probe_v1.jsonl`
- checkpoint: `results/runs/pilot_20k_20x_bf16_20260524_20260524_091754/checkpoints/best.pt`
- profiles: `greedy`

## 分组汇总

| suite | category | profile | examples | auto pass | CJK | repeat-4 | tokens | tok/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| multilingual_probe | code_mixed | greedy | 1 | n/a | 0.000 | 0.951 | 64.0 | 346.57 |
| multilingual_probe | english_completion | greedy | 1 | n/a | 0.000 | 0.508 | 64.0 | 106.92 |
| multilingual_probe | mixed_language | greedy | 1 | n/a | 0.934 | 0.648 | 64.0 | 345.31 |
| multilingual_probe | translation | greedy | 1 | 0.0% | 0.000 | 0.935 | 64.0 | 353.87 |

完整生成内容保存在 `generations.jsonl`；人工评分请填写 `manual_review.csv`。
