# 中文生成与指令小测报告

## 配置

- prompt file: `eval/prompts/zh_generation_v1.jsonl`
- checkpoint: `results/runs/pilot_20k_20x_bf16_20260524_20260524_091754/checkpoints/best.pt`
- profiles: `greedy`

## 分组汇总

| suite | category | profile | examples | auto pass | CJK | repeat-4 | tokens | tok/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| generation | expository | greedy | 2 | n/a | 0.910 | 0.836 | 128.0 | 347.68 |
| generation | narrative | greedy | 3 | n/a | 0.810 | 0.856 | 128.0 | 285.93 |

完整生成内容保存在 `generations.jsonl`；人工评分请填写 `manual_review.csv`。
