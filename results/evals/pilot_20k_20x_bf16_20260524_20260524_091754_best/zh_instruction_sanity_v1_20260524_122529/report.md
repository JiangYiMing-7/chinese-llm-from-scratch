# 中文生成与指令小测报告

## 配置

- prompt file: `eval/prompts/zh_instruction_sanity_v1.jsonl`
- checkpoint: `results/runs/pilot_20k_20x_bf16_20260524_20260524_091754/checkpoints/best.pt`
- profiles: `greedy`

## 分组汇总

| suite | category | profile | examples | auto pass | CJK | repeat-4 | tokens | tok/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| instruction_sanity | classification | greedy | 1 | 0.0% | 0.704 | 0.706 | 64.0 | 338.93 |
| instruction_sanity | extraction | greedy | 1 | 0.0% | 0.842 | 0.745 | 64.0 | 340.66 |
| instruction_sanity | format | greedy | 1 | 0.0% | 0.667 | 0.933 | 64.0 | 338.22 |
| instruction_sanity | json | greedy | 1 | 100.0% | 0.667 | 0.575 | 64.0 | 344.77 |
| instruction_sanity | short_answer | greedy | 1 | 0.0% | 0.874 | 0.821 | 64.0 | 111.61 |

完整生成内容保存在 `generations.jsonl`；人工评分请填写 `manual_review.csv`。
