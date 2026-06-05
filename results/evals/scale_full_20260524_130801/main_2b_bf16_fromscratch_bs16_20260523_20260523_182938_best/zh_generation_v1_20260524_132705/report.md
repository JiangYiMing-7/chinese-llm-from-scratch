# 中文生成与指令小测报告

## 配置

- prompt file: `eval/prompts/zh_generation_v1.jsonl`
- checkpoint: `results/runs/main_2b_bf16_fromscratch_bs16_20260523_20260523_182938/checkpoints/best.pt`
- profiles: `greedy, balanced_s1, balanced_s2`

## 分组汇总

| suite | category | profile | examples | auto pass | CJK | repeat-4 | tokens | tok/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| generation | dialogue | balanced_s1 | 1 | n/a | 0.818 | 0.882 | 128.0 | 219.49 |
| generation | dialogue | balanced_s2 | 1 | n/a | 0.805 | 0.835 | 128.0 | 217.24 |
| generation | dialogue | greedy | 1 | n/a | 0.484 | 0.000 | 15.0 | 234.75 |
| generation | expository | balanced_s1 | 2 | n/a | 0.937 | 0.203 | 128.0 | 220.17 |
| generation | expository | balanced_s2 | 2 | n/a | 0.899 | 0.303 | 121.5 | 219.80 |
| generation | expository | greedy | 2 | n/a | 0.896 | 0.794 | 128.0 | 236.98 |
| generation | formal_writing | balanced_s1 | 2 | n/a | 0.818 | 0.365 | 85.0 | 221.08 |
| generation | formal_writing | balanced_s2 | 2 | n/a | 0.855 | 0.475 | 110.5 | 220.42 |
| generation | formal_writing | greedy | 2 | n/a | 0.658 | 0.754 | 128.0 | 237.59 |
| generation | knowledge | balanced_s1 | 2 | n/a | 0.894 | 0.207 | 128.0 | 220.29 |
| generation | knowledge | balanced_s2 | 2 | n/a | 0.863 | 0.189 | 128.0 | 221.20 |
| generation | knowledge | greedy | 2 | n/a | 0.897 | 0.771 | 128.0 | 235.75 |
| generation | long_coherence | balanced_s1 | 1 | n/a | 0.782 | 0.131 | 42.0 | 222.63 |
| generation | long_coherence | balanced_s2 | 1 | n/a | 0.850 | 0.413 | 128.0 | 222.55 |
| generation | long_coherence | greedy | 1 | n/a | 0.917 | 0.773 | 128.0 | 236.98 |
| generation | mixed_language | balanced_s1 | 1 | n/a | 0.771 | 0.149 | 128.0 | 221.59 |
| generation | mixed_language | balanced_s2 | 1 | n/a | 0.505 | 0.322 | 128.0 | 218.01 |
| generation | mixed_language | greedy | 1 | n/a | 0.306 | 0.772 | 128.0 | 237.33 |
| generation | narrative | balanced_s1 | 3 | n/a | 0.847 | 0.296 | 128.0 | 210.23 |
| generation | narrative | balanced_s2 | 3 | n/a | 0.850 | 0.116 | 128.0 | 220.88 |
| generation | narrative | greedy | 3 | n/a | 0.836 | 0.783 | 128.0 | 202.54 |
| generation | numbers | balanced_s1 | 2 | n/a | 0.855 | 0.169 | 128.0 | 219.67 |
| generation | numbers | balanced_s2 | 2 | n/a | 0.660 | 0.291 | 128.0 | 221.28 |
| generation | numbers | greedy | 2 | n/a | 0.655 | 0.769 | 128.0 | 236.15 |
| generation | punctuation | balanced_s1 | 1 | n/a | 0.905 | 0.111 | 128.0 | 222.25 |
| generation | punctuation | balanced_s2 | 1 | n/a | 0.860 | 0.629 | 128.0 | 223.86 |
| generation | punctuation | greedy | 1 | n/a | 0.896 | 0.798 | 128.0 | 239.13 |
| generation | science | balanced_s1 | 2 | n/a | 0.904 | 0.307 | 128.0 | 220.71 |
| generation | science | balanced_s2 | 2 | n/a | 0.918 | 0.156 | 128.0 | 220.82 |
| generation | science | greedy | 2 | n/a | 0.916 | 0.784 | 128.0 | 236.24 |
| generation | summary_style | balanced_s1 | 1 | n/a | 0.913 | 0.271 | 128.0 | 219.69 |
| generation | summary_style | balanced_s2 | 1 | n/a | 0.859 | 0.601 | 88.0 | 219.02 |
| generation | summary_style | greedy | 1 | n/a | 0.937 | 0.782 | 128.0 | 237.29 |
| generation | technology | balanced_s1 | 2 | n/a | 0.661 | 0.393 | 128.0 | 220.23 |
| generation | technology | balanced_s2 | 2 | n/a | 0.814 | 0.491 | 128.0 | 220.57 |
| generation | technology | greedy | 2 | n/a | 0.909 | 0.726 | 128.0 | 235.73 |

完整生成内容保存在 `generations.jsonl`；人工评分请填写 `manual_review.csv`。
