# 中文生成与指令小测报告

## 配置

- prompt file: `eval/prompts/zh_instruction_sanity_v1.jsonl`
- checkpoint: `results/runs/main_20260523_005535/checkpoints/best.pt`
- profiles: `greedy`

## 分组汇总

| suite | category | profile | examples | auto pass | CJK | repeat-4 | tokens | tok/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| instruction_sanity | arithmetic | greedy | 1 | 0.0% | 0.608 | 0.511 | 64.0 | 235.84 |
| instruction_sanity | benign_instruction | greedy | 1 | 100.0% | 0.909 | 0.741 | 64.0 | 232.30 |
| instruction_sanity | classification | greedy | 1 | 0.0% | 0.755 | 0.632 | 64.0 | 234.59 |
| instruction_sanity | constraint | greedy | 1 | 0.0% | 0.889 | 0.823 | 64.0 | 228.00 |
| instruction_sanity | extraction | greedy | 2 | 0.0% | 0.535 | 0.711 | 64.0 | 233.06 |
| instruction_sanity | format | greedy | 1 | 0.0% | 0.885 | 0.619 | 64.0 | 236.23 |
| instruction_sanity | json | greedy | 1 | 100.0% | 0.690 | 0.571 | 64.0 | 236.04 |
| instruction_sanity | rewriting | greedy | 1 | 0.0% | 0.849 | 0.614 | 64.0 | 234.48 |
| instruction_sanity | short_answer | greedy | 1 | 0.0% | 0.870 | 0.742 | 64.0 | 91.04 |
| instruction_sanity | summary | greedy | 1 | 0.0% | 0.896 | 0.484 | 64.0 | 232.18 |
| instruction_sanity | translation | greedy | 1 | 0.0% | 0.144 | 0.513 | 64.0 | 232.81 |

完整生成内容保存在 `generations.jsonl`；人工评分请填写 `manual_review.csv`。
