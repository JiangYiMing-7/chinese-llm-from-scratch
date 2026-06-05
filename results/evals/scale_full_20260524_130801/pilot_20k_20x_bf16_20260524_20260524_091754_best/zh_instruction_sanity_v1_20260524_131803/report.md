# 中文生成与指令小测报告

## 配置

- prompt file: `eval/prompts/zh_instruction_sanity_v1.jsonl`
- checkpoint: `results/runs/pilot_20k_20x_bf16_20260524_20260524_091754/checkpoints/best.pt`
- profiles: `greedy`

## 分组汇总

| suite | category | profile | examples | auto pass | CJK | repeat-4 | tokens | tok/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| instruction_sanity | arithmetic | greedy | 1 | 0.0% | 0.624 | 0.756 | 64.0 | 348.14 |
| instruction_sanity | benign_instruction | greedy | 1 | 100.0% | 0.841 | 0.582 | 64.0 | 347.24 |
| instruction_sanity | classification | greedy | 1 | 0.0% | 0.704 | 0.706 | 64.0 | 342.13 |
| instruction_sanity | constraint | greedy | 1 | 0.0% | 0.889 | 0.823 | 64.0 | 310.84 |
| instruction_sanity | extraction | greedy | 2 | 50.0% | 0.848 | 0.609 | 64.0 | 345.33 |
| instruction_sanity | format | greedy | 1 | 0.0% | 0.667 | 0.933 | 64.0 | 335.01 |
| instruction_sanity | json | greedy | 1 | 100.0% | 0.667 | 0.575 | 64.0 | 344.53 |
| instruction_sanity | rewriting | greedy | 1 | 0.0% | 0.833 | 0.728 | 64.0 | 346.96 |
| instruction_sanity | short_answer | greedy | 1 | 0.0% | 0.874 | 0.821 | 64.0 | 104.01 |
| instruction_sanity | summary | greedy | 1 | 0.0% | 0.925 | 0.587 | 64.0 | 341.17 |
| instruction_sanity | translation | greedy | 1 | 0.0% | 0.186 | 0.461 | 64.0 | 336.87 |

完整生成内容保存在 `generations.jsonl`；人工评分请填写 `manual_review.csv`。
