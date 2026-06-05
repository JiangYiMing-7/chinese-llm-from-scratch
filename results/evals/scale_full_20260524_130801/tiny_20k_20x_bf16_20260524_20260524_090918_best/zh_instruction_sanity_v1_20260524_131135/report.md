# 中文生成与指令小测报告

## 配置

- prompt file: `eval/prompts/zh_instruction_sanity_v1.jsonl`
- checkpoint: `results/runs/tiny_20k_20x_bf16_20260524_20260524_090918/checkpoints/best.pt`
- profiles: `greedy`

## 分组汇总

| suite | category | profile | examples | auto pass | CJK | repeat-4 | tokens | tok/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| instruction_sanity | arithmetic | greedy | 1 | 0.0% | 0.933 | 0.559 | 64.0 | 644.92 |
| instruction_sanity | benign_instruction | greedy | 1 | 100.0% | 0.667 | 0.950 | 64.0 | 675.74 |
| instruction_sanity | classification | greedy | 1 | 0.0% | 0.667 | 0.950 | 64.0 | 630.14 |
| instruction_sanity | constraint | greedy | 1 | 0.0% | 0.667 | 0.867 | 64.0 | 595.91 |
| instruction_sanity | extraction | greedy | 2 | 0.0% | 0.669 | 0.926 | 64.0 | 650.11 |
| instruction_sanity | format | greedy | 1 | 0.0% | 0.667 | 0.950 | 64.0 | 643.62 |
| instruction_sanity | json | greedy | 1 | 0.0% | 0.297 | 0.901 | 64.0 | 643.76 |
| instruction_sanity | rewriting | greedy | 1 | 100.0% | 0.667 | 0.950 | 64.0 | 660.48 |
| instruction_sanity | short_answer | greedy | 1 | 0.0% | 0.250 | 0.869 | 64.0 | 134.33 |
| instruction_sanity | summary | greedy | 1 | 0.0% | 0.667 | 0.950 | 64.0 | 650.19 |
| instruction_sanity | translation | greedy | 1 | 0.0% | 0.667 | 0.950 | 64.0 | 659.66 |

完整生成内容保存在 `generations.jsonl`；人工评分请填写 `manual_review.csv`。
