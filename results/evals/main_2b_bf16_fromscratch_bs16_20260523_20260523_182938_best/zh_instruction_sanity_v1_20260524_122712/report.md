# 中文生成与指令小测报告

## 配置

- prompt file: `eval/prompts/zh_instruction_sanity_v1.jsonl`
- checkpoint: `results/runs/main_2b_bf16_fromscratch_bs16_20260523_20260523_182938/checkpoints/best.pt`
- profiles: `greedy`

## 分组汇总

| suite | category | profile | examples | auto pass | CJK | repeat-4 | tokens | tok/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| instruction_sanity | classification | greedy | 1 | 0.0% | 0.690 | 0.568 | 64.0 | 229.75 |
| instruction_sanity | extraction | greedy | 1 | 0.0% | 0.720 | 0.444 | 64.0 | 233.70 |
| instruction_sanity | format | greedy | 1 | 0.0% | 0.810 | 0.679 | 64.0 | 235.96 |
| instruction_sanity | json | greedy | 1 | 100.0% | 0.745 | 0.653 | 64.0 | 234.89 |
| instruction_sanity | short_answer | greedy | 1 | 0.0% | 0.859 | 0.719 | 64.0 | 89.15 |

完整生成内容保存在 `generations.jsonl`；人工评分请填写 `manual_review.csv`。
