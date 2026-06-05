# 中文生成与指令小测报告

## 配置

- prompt file: `eval/prompts/zh_generation_v1.jsonl`
- checkpoint: `results/runs/tiny_20k_20x_bf16_20260524_20260524_090918/checkpoints/best.pt`
- profiles: `greedy, balanced_s1, balanced_s2`

## 分组汇总

| suite | category | profile | examples | auto pass | CJK | repeat-4 | tokens | tok/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| generation | dialogue | balanced_s1 | 1 | n/a | 0.927 | 0.440 | 128.0 | 562.17 |
| generation | dialogue | balanced_s2 | 1 | n/a | 0.765 | 0.447 | 128.0 | 562.98 |
| generation | dialogue | greedy | 1 | n/a | 0.754 | 0.976 | 128.0 | 670.31 |
| generation | expository | balanced_s1 | 2 | n/a | 0.912 | 0.241 | 128.0 | 551.68 |
| generation | expository | balanced_s2 | 2 | n/a | 0.856 | 0.371 | 128.0 | 566.06 |
| generation | expository | greedy | 2 | n/a | 0.959 | 0.897 | 128.0 | 674.74 |
| generation | formal_writing | balanced_s1 | 2 | n/a | 0.958 | 0.705 | 128.0 | 563.97 |
| generation | formal_writing | balanced_s2 | 2 | n/a | 0.872 | 0.582 | 128.0 | 563.56 |
| generation | formal_writing | greedy | 2 | n/a | 0.899 | 0.961 | 128.0 | 670.42 |
| generation | knowledge | balanced_s1 | 2 | n/a | 0.827 | 0.391 | 128.0 | 565.81 |
| generation | knowledge | balanced_s2 | 2 | n/a | 0.620 | 0.114 | 72.5 | 543.79 |
| generation | knowledge | greedy | 2 | n/a | 0.824 | 0.911 | 128.0 | 670.20 |
| generation | long_coherence | balanced_s1 | 1 | n/a | 0.864 | 0.219 | 128.0 | 561.56 |
| generation | long_coherence | balanced_s2 | 1 | n/a | 0.909 | 0.319 | 128.0 | 558.01 |
| generation | long_coherence | greedy | 1 | n/a | 0.793 | 0.903 | 128.0 | 669.60 |
| generation | mixed_language | balanced_s1 | 1 | n/a | 0.905 | 0.376 | 128.0 | 549.43 |
| generation | mixed_language | balanced_s2 | 1 | n/a | 0.568 | 0.251 | 128.0 | 562.87 |
| generation | mixed_language | greedy | 1 | n/a | 0.800 | 0.949 | 128.0 | 671.85 |
| generation | narrative | balanced_s1 | 3 | n/a | 0.875 | 0.244 | 128.0 | 496.18 |
| generation | narrative | balanced_s2 | 3 | n/a | 0.834 | 0.255 | 128.0 | 557.16 |
| generation | narrative | greedy | 3 | n/a | 0.817 | 0.868 | 128.0 | 511.75 |
| generation | numbers | balanced_s1 | 2 | n/a | 0.842 | 0.329 | 128.0 | 562.44 |
| generation | numbers | balanced_s2 | 2 | n/a | 0.728 | 0.261 | 128.0 | 563.55 |
| generation | numbers | greedy | 2 | n/a | 0.621 | 0.866 | 128.0 | 664.84 |
| generation | punctuation | balanced_s1 | 1 | n/a | 0.831 | 0.136 | 128.0 | 560.50 |
| generation | punctuation | balanced_s2 | 1 | n/a | 0.768 | 0.118 | 128.0 | 560.40 |
| generation | punctuation | greedy | 1 | n/a | 0.800 | 0.936 | 128.0 | 667.96 |
| generation | science | balanced_s1 | 2 | n/a | 0.901 | 0.548 | 128.0 | 557.57 |
| generation | science | balanced_s2 | 2 | n/a | 0.898 | 0.334 | 128.0 | 559.25 |
| generation | science | greedy | 2 | n/a | 0.798 | 0.844 | 128.0 | 666.96 |
| generation | summary_style | balanced_s1 | 1 | n/a | 0.891 | 0.415 | 128.0 | 561.39 |
| generation | summary_style | balanced_s2 | 1 | n/a | 0.831 | 0.297 | 115.0 | 563.62 |
| generation | summary_style | greedy | 1 | n/a | 0.890 | 0.858 | 128.0 | 670.55 |
| generation | technology | balanced_s1 | 2 | n/a | 0.596 | 0.553 | 128.0 | 560.51 |
| generation | technology | balanced_s2 | 2 | n/a | 0.481 | 0.403 | 128.0 | 567.31 |
| generation | technology | greedy | 2 | n/a | 0.920 | 0.884 | 128.0 | 664.57 |

完整生成内容保存在 `generations.jsonl`；人工评分请填写 `manual_review.csv`。
