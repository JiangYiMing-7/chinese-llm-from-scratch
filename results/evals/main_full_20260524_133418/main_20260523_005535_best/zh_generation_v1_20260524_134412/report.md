# 中文生成与指令小测报告

## 配置

- prompt file: `eval/prompts/zh_generation_v1.jsonl`
- checkpoint: `results/runs/main_20260523_005535/checkpoints/best.pt`
- profiles: `greedy, balanced_s1, balanced_s2`

## 分组汇总

| suite | category | profile | examples | auto pass | CJK | repeat-4 | tokens | tok/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| generation | dialogue | balanced_s1 | 1 | n/a | 0.806 | 0.847 | 128.0 | 217.58 |
| generation | dialogue | balanced_s2 | 1 | n/a | 0.796 | 0.415 | 128.0 | 217.47 |
| generation | dialogue | greedy | 1 | n/a | 0.797 | 0.897 | 128.0 | 232.83 |
| generation | expository | balanced_s1 | 2 | n/a | 0.907 | 0.195 | 128.0 | 214.64 |
| generation | expository | balanced_s2 | 2 | n/a | 0.914 | 0.499 | 128.0 | 213.96 |
| generation | expository | greedy | 2 | n/a | 0.868 | 0.832 | 128.0 | 229.18 |
| generation | formal_writing | balanced_s1 | 2 | n/a | 0.905 | 0.571 | 128.0 | 216.87 |
| generation | formal_writing | balanced_s2 | 2 | n/a | 0.904 | 0.769 | 128.0 | 217.97 |
| generation | formal_writing | greedy | 2 | n/a | 0.891 | 0.834 | 128.0 | 231.14 |
| generation | knowledge | balanced_s1 | 2 | n/a | 0.767 | 0.150 | 128.0 | 214.14 |
| generation | knowledge | balanced_s2 | 2 | n/a | 0.840 | 0.295 | 128.0 | 214.22 |
| generation | knowledge | greedy | 2 | n/a | 0.816 | 0.843 | 128.0 | 229.77 |
| generation | long_coherence | balanced_s1 | 1 | n/a | 0.761 | 0.333 | 128.0 | 217.09 |
| generation | long_coherence | balanced_s2 | 1 | n/a | 0.889 | 0.323 | 128.0 | 216.59 |
| generation | long_coherence | greedy | 1 | n/a | 0.891 | 0.929 | 128.0 | 234.02 |
| generation | mixed_language | balanced_s1 | 1 | n/a | 0.780 | 0.431 | 128.0 | 217.39 |
| generation | mixed_language | balanced_s2 | 1 | n/a | 0.942 | 0.412 | 128.0 | 217.83 |
| generation | mixed_language | greedy | 1 | n/a | 0.893 | 0.856 | 128.0 | 234.55 |
| generation | narrative | balanced_s1 | 3 | n/a | 0.845 | 0.366 | 128.0 | 204.39 |
| generation | narrative | balanced_s2 | 3 | n/a | 0.797 | 0.423 | 128.0 | 214.47 |
| generation | narrative | greedy | 3 | n/a | 0.800 | 0.892 | 128.0 | 195.87 |
| generation | numbers | balanced_s1 | 2 | n/a | 0.691 | 0.444 | 128.0 | 216.77 |
| generation | numbers | balanced_s2 | 2 | n/a | 0.699 | 0.717 | 128.0 | 217.47 |
| generation | numbers | greedy | 2 | n/a | 0.885 | 0.925 | 128.0 | 233.95 |
| generation | punctuation | balanced_s1 | 1 | n/a | 0.882 | 0.149 | 128.0 | 216.95 |
| generation | punctuation | balanced_s2 | 1 | n/a | 0.903 | 0.469 | 128.0 | 217.91 |
| generation | punctuation | greedy | 1 | n/a | 0.884 | 0.794 | 128.0 | 233.94 |
| generation | science | balanced_s1 | 2 | n/a | 0.888 | 0.519 | 128.0 | 216.37 |
| generation | science | balanced_s2 | 2 | n/a | 0.914 | 0.273 | 128.0 | 216.37 |
| generation | science | greedy | 2 | n/a | 0.890 | 0.787 | 128.0 | 231.57 |
| generation | summary_style | balanced_s1 | 1 | n/a | 0.912 | 0.329 | 128.0 | 216.55 |
| generation | summary_style | balanced_s2 | 1 | n/a | 0.896 | 0.533 | 128.0 | 216.23 |
| generation | summary_style | greedy | 1 | n/a | 0.936 | 0.906 | 128.0 | 232.35 |
| generation | technology | balanced_s1 | 2 | n/a | 0.834 | 0.569 | 128.0 | 217.51 |
| generation | technology | balanced_s2 | 2 | n/a | 0.933 | 0.446 | 128.0 | 218.32 |
| generation | technology | greedy | 2 | n/a | 0.945 | 0.802 | 128.0 | 235.19 |

完整生成内容保存在 `generations.jsonl`；人工评分请填写 `manual_review.csv`。
