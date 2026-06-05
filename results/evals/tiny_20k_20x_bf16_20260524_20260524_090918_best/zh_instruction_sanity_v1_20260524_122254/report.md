# 中文生成与指令小测报告

## 配置

- prompt file: `eval/prompts/zh_instruction_sanity_v1.jsonl`
- checkpoint: `results/runs/tiny_20k_20x_bf16_20260524_20260524_090918/checkpoints/best.pt`
- profiles: `greedy`

## 分组汇总

| suite | category | profile | examples | auto pass | CJK | repeat-4 | tokens | tok/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| instruction_sanity | classification | greedy | 1 | 0.0% | 0.667 | 0.950 | 64.0 | 584.10 |
| instruction_sanity | extraction | greedy | 1 | 0.0% | 0.672 | 0.902 | 64.0 | 624.40 |
| instruction_sanity | format | greedy | 1 | 0.0% | 0.667 | 0.950 | 64.0 | 632.56 |
| instruction_sanity | json | greedy | 1 | 0.0% | 0.297 | 0.901 | 64.0 | 644.24 |
| instruction_sanity | short_answer | greedy | 1 | 0.0% | 0.250 | 0.869 | 64.0 | 131.00 |

完整生成内容保存在 `generations.jsonl`；人工评分请填写 `manual_review.csv`。
