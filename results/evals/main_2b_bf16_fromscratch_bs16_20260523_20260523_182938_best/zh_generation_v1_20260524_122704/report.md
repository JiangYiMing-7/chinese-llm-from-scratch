# 中文生成与指令小测报告

## 配置

- prompt file: `eval/prompts/zh_generation_v1.jsonl`
- checkpoint: `results/runs/main_2b_bf16_fromscratch_bs16_20260523_20260523_182938/checkpoints/best.pt`
- profiles: `greedy`

## 分组汇总

| suite | category | profile | examples | auto pass | CJK | repeat-4 | tokens | tok/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| generation | expository | greedy | 2 | n/a | 0.896 | 0.794 | 128.0 | 229.65 |
| generation | narrative | greedy | 3 | n/a | 0.836 | 0.783 | 128.0 | 195.56 |

完整生成内容保存在 `generations.jsonl`；人工评分请填写 `manual_review.csv`。
