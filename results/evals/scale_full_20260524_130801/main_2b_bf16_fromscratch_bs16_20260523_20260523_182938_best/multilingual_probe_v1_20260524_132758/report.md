# 中文生成与指令小测报告

## 配置

- prompt file: `eval/prompts/multilingual_probe_v1.jsonl`
- checkpoint: `results/runs/main_2b_bf16_fromscratch_bs16_20260523_20260523_182938/checkpoints/best.pt`
- profiles: `greedy`

## 分组汇总

| suite | category | profile | examples | auto pass | CJK | repeat-4 | tokens | tok/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| multilingual_probe | code_mixed | greedy | 1 | n/a | 0.000 | 0.854 | 64.0 | 226.06 |
| multilingual_probe | english_completion | greedy | 1 | n/a | 0.000 | 0.907 | 64.0 | 92.59 |
| multilingual_probe | mixed_language | greedy | 1 | n/a | 0.877 | 0.700 | 64.0 | 221.35 |
| multilingual_probe | translation | greedy | 1 | 0.0% | 0.183 | 0.519 | 64.0 | 226.25 |

完整生成内容保存在 `generations.jsonl`；人工评分请填写 `manual_review.csv`。
