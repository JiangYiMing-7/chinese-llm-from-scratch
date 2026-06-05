# 中文生成与指令小测报告

## 配置

- prompt file: `eval/prompts/zh_instruction_sanity_v1.jsonl`
- checkpoint: `results/runs/main_2b_bf16_fromscratch_bs16_20260523_20260523_182938/checkpoints/best.pt`
- profiles: `greedy`

## 分组汇总

| suite | category | profile | examples | auto pass | CJK | repeat-4 | tokens | tok/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| instruction_sanity | arithmetic | greedy | 1 | 0.0% | 0.863 | 0.837 | 64.0 | 191.11 |
| instruction_sanity | benign_instruction | greedy | 1 | 100.0% | 0.909 | 0.682 | 64.0 | 225.30 |
| instruction_sanity | classification | greedy | 1 | 0.0% | 0.690 | 0.568 | 64.0 | 193.12 |
| instruction_sanity | constraint | greedy | 1 | 0.0% | 0.855 | 0.775 | 64.0 | 206.88 |
| instruction_sanity | extraction | greedy | 2 | 50.0% | 0.791 | 0.458 | 64.0 | 194.27 |
| instruction_sanity | format | greedy | 1 | 0.0% | 0.810 | 0.679 | 64.0 | 195.09 |
| instruction_sanity | json | greedy | 1 | 100.0% | 0.745 | 0.653 | 64.0 | 215.19 |
| instruction_sanity | rewriting | greedy | 1 | 0.0% | 0.849 | 0.614 | 64.0 | 217.46 |
| instruction_sanity | short_answer | greedy | 1 | 0.0% | 0.859 | 0.719 | 64.0 | 87.42 |
| instruction_sanity | summary | greedy | 1 | 0.0% | 0.896 | 0.495 | 64.0 | 207.11 |
| instruction_sanity | translation | greedy | 1 | 0.0% | 0.186 | 0.461 | 64.0 | 203.92 |

完整生成内容保存在 `generations.jsonl`；人工评分请填写 `manual_review.csv`。
