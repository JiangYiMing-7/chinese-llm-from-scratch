# 中文生成与指令小测报告

## 配置

- prompt file: `eval/prompts/zh_instruction_sanity_v1.jsonl`
- checkpoint: `results/runs/main_20260523_005535/checkpoints/best.pt`
- profiles: `greedy`

## 分组汇总

| suite | category | profile | examples | auto pass | CJK | repeat-4 | tokens | tok/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| instruction_sanity | classification | greedy | 1 | 0.0% | 0.755 | 0.632 | 64.0 | 234.36 |
| instruction_sanity | extraction | greedy | 1 | 0.0% | 0.720 | 0.567 | 64.0 | 235.43 |
| instruction_sanity | format | greedy | 1 | 0.0% | 0.885 | 0.619 | 64.0 | 239.64 |
| instruction_sanity | json | greedy | 1 | 100.0% | 0.690 | 0.571 | 64.0 | 238.36 |
| instruction_sanity | short_answer | greedy | 1 | 0.0% | 0.870 | 0.742 | 64.0 | 93.88 |

完整生成内容保存在 `generations.jsonl`；人工评分请填写 `manual_review.csv`。
