# 中文生成与指令小测报告

## 配置

- prompt file: `eval/prompts/zh_generation_v1.jsonl`
- checkpoint: `results/runs/main_2b_bf16_fromscratch_bs16_20260523_20260523_182938/checkpoints/best.pt`
- profiles: `greedy, balanced_s1, balanced_s2`

## 分组汇总

| suite | category | profile | examples | auto pass | CJK | repeat-4 | tokens | tok/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| generation | dialogue | balanced_s1 | 1 | n/a | 0.818 | 0.882 | 128.0 | 212.26 |
| generation | dialogue | balanced_s2 | 1 | n/a | 0.805 | 0.835 | 128.0 | 213.56 |
| generation | dialogue | greedy | 1 | n/a | 0.484 | 0.000 | 15.0 | 228.09 |
| generation | expository | balanced_s1 | 2 | n/a | 0.937 | 0.203 | 128.0 | 207.28 |
| generation | expository | balanced_s2 | 2 | n/a | 0.899 | 0.303 | 121.5 | 207.89 |
| generation | expository | greedy | 2 | n/a | 0.896 | 0.794 | 128.0 | 226.71 |
| generation | formal_writing | balanced_s1 | 2 | n/a | 0.818 | 0.365 | 85.0 | 212.48 |
| generation | formal_writing | balanced_s2 | 2 | n/a | 0.855 | 0.475 | 110.5 | 209.18 |
| generation | formal_writing | greedy | 2 | n/a | 0.658 | 0.754 | 128.0 | 226.83 |
| generation | knowledge | balanced_s1 | 2 | n/a | 0.894 | 0.207 | 128.0 | 209.47 |
| generation | knowledge | balanced_s2 | 2 | n/a | 0.863 | 0.189 | 128.0 | 208.19 |
| generation | knowledge | greedy | 2 | n/a | 0.897 | 0.771 | 128.0 | 227.12 |
| generation | long_coherence | balanced_s1 | 1 | n/a | 0.782 | 0.131 | 42.0 | 214.57 |
| generation | long_coherence | balanced_s2 | 1 | n/a | 0.850 | 0.413 | 128.0 | 211.65 |
| generation | long_coherence | greedy | 1 | n/a | 0.917 | 0.773 | 128.0 | 228.07 |
| generation | mixed_language | balanced_s1 | 1 | n/a | 0.771 | 0.149 | 128.0 | 213.18 |
| generation | mixed_language | balanced_s2 | 1 | n/a | 0.505 | 0.322 | 128.0 | 211.03 |
| generation | mixed_language | greedy | 1 | n/a | 0.306 | 0.772 | 128.0 | 223.81 |
| generation | narrative | balanced_s1 | 3 | n/a | 0.847 | 0.296 | 128.0 | 199.23 |
| generation | narrative | balanced_s2 | 3 | n/a | 0.850 | 0.116 | 128.0 | 210.86 |
| generation | narrative | greedy | 3 | n/a | 0.836 | 0.783 | 128.0 | 193.66 |
| generation | numbers | balanced_s1 | 2 | n/a | 0.855 | 0.169 | 128.0 | 213.60 |
| generation | numbers | balanced_s2 | 2 | n/a | 0.660 | 0.291 | 128.0 | 211.79 |
| generation | numbers | greedy | 2 | n/a | 0.655 | 0.769 | 128.0 | 225.11 |
| generation | punctuation | balanced_s1 | 1 | n/a | 0.905 | 0.111 | 128.0 | 212.50 |
| generation | punctuation | balanced_s2 | 1 | n/a | 0.860 | 0.629 | 128.0 | 210.02 |
| generation | punctuation | greedy | 1 | n/a | 0.896 | 0.798 | 128.0 | 223.16 |
| generation | science | balanced_s1 | 2 | n/a | 0.904 | 0.307 | 128.0 | 212.50 |
| generation | science | balanced_s2 | 2 | n/a | 0.918 | 0.156 | 128.0 | 209.64 |
| generation | science | greedy | 2 | n/a | 0.916 | 0.784 | 128.0 | 227.17 |
| generation | summary_style | balanced_s1 | 1 | n/a | 0.913 | 0.271 | 128.0 | 214.36 |
| generation | summary_style | balanced_s2 | 1 | n/a | 0.859 | 0.601 | 88.0 | 214.20 |
| generation | summary_style | greedy | 1 | n/a | 0.937 | 0.782 | 128.0 | 225.68 |
| generation | technology | balanced_s1 | 2 | n/a | 0.661 | 0.393 | 128.0 | 207.40 |
| generation | technology | balanced_s2 | 2 | n/a | 0.814 | 0.491 | 128.0 | 211.73 |
| generation | technology | greedy | 2 | n/a | 0.909 | 0.726 | 128.0 | 224.02 |

完整生成内容保存在 `generations.jsonl`；人工评分请填写 `manual_review.csv`。
