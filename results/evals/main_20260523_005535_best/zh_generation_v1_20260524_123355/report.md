# 中文生成与指令小测报告

## 配置

- prompt file: `eval/prompts/zh_generation_v1.jsonl`
- checkpoint: `results/runs/main_20260523_005535/checkpoints/best.pt`
- profiles: `greedy`

## 分组汇总

| suite | category | profile | examples | auto pass | CJK | repeat-4 | tokens | tok/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| generation | expository | greedy | 2 | n/a | 0.868 | 0.832 | 128.0 | 236.93 |
| generation | narrative | greedy | 3 | n/a | 0.800 | 0.892 | 128.0 | 203.42 |

完整生成内容保存在 `generations.jsonl`；人工评分请填写 `manual_review.csv`。
