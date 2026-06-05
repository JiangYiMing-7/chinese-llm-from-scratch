# 中文生成与指令小测报告

## 配置

- prompt file: `eval/prompts/zh_generation_v1.jsonl`
- checkpoint: `results/runs/pilot_20k_20x_bf16_20260524_20260524_091754/checkpoints/best.pt`
- profiles: `greedy, balanced_s1, balanced_s2`

## 分组汇总

| suite | category | profile | examples | auto pass | CJK | repeat-4 | tokens | tok/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| generation | dialogue | balanced_s1 | 1 | n/a | 0.788 | 0.829 | 128.0 | 312.06 |
| generation | dialogue | balanced_s2 | 1 | n/a | 0.911 | 0.314 | 128.0 | 319.57 |
| generation | dialogue | greedy | 1 | n/a | 0.810 | 0.844 | 128.0 | 351.82 |
| generation | expository | balanced_s1 | 2 | n/a | 0.918 | 0.506 | 128.0 | 314.59 |
| generation | expository | balanced_s2 | 2 | n/a | 0.889 | 0.374 | 128.0 | 314.78 |
| generation | expository | greedy | 2 | n/a | 0.910 | 0.836 | 128.0 | 349.49 |
| generation | formal_writing | balanced_s1 | 2 | n/a | 0.892 | 0.623 | 128.0 | 317.24 |
| generation | formal_writing | balanced_s2 | 2 | n/a | 0.899 | 0.256 | 128.0 | 315.44 |
| generation | formal_writing | greedy | 2 | n/a | 0.974 | 0.876 | 128.0 | 349.38 |
| generation | knowledge | balanced_s1 | 2 | n/a | 0.761 | 0.019 | 77.5 | 319.24 |
| generation | knowledge | balanced_s2 | 2 | n/a | 0.855 | 0.208 | 128.0 | 314.29 |
| generation | knowledge | greedy | 2 | n/a | 0.859 | 0.810 | 128.0 | 337.77 |
| generation | long_coherence | balanced_s1 | 1 | n/a | 0.763 | 0.549 | 128.0 | 315.45 |
| generation | long_coherence | balanced_s2 | 1 | n/a | 0.905 | 0.091 | 128.0 | 316.86 |
| generation | long_coherence | greedy | 1 | n/a | 0.827 | 0.834 | 128.0 | 348.15 |
| generation | mixed_language | balanced_s1 | 1 | n/a | 0.730 | 0.345 | 128.0 | 313.55 |
| generation | mixed_language | balanced_s2 | 1 | n/a | 0.544 | 0.453 | 128.0 | 323.52 |
| generation | mixed_language | greedy | 1 | n/a | 0.722 | 0.870 | 128.0 | 348.28 |
| generation | narrative | balanced_s1 | 3 | n/a | 0.898 | 0.281 | 128.0 | 294.84 |
| generation | narrative | balanced_s2 | 3 | n/a | 0.830 | 0.071 | 128.0 | 316.23 |
| generation | narrative | greedy | 3 | n/a | 0.810 | 0.856 | 128.0 | 285.88 |
| generation | numbers | balanced_s1 | 2 | n/a | 0.824 | 0.519 | 128.0 | 319.43 |
| generation | numbers | balanced_s2 | 2 | n/a | 0.798 | 0.188 | 128.0 | 315.14 |
| generation | numbers | greedy | 2 | n/a | 0.584 | 0.866 | 128.0 | 349.10 |
| generation | punctuation | balanced_s1 | 1 | n/a | 0.548 | 0.527 | 128.0 | 314.11 |
| generation | punctuation | balanced_s2 | 1 | n/a | 0.830 | 0.427 | 128.0 | 319.05 |
| generation | punctuation | greedy | 1 | n/a | 0.841 | 0.804 | 128.0 | 347.88 |
| generation | science | balanced_s1 | 2 | n/a | 0.938 | 0.260 | 128.0 | 315.98 |
| generation | science | balanced_s2 | 2 | n/a | 0.896 | 0.226 | 128.0 | 315.50 |
| generation | science | greedy | 2 | n/a | 0.919 | 0.814 | 128.0 | 352.24 |
| generation | summary_style | balanced_s1 | 1 | n/a | 0.695 | 0.000 | 29.0 | 309.59 |
| generation | summary_style | balanced_s2 | 1 | n/a | 0.872 | 0.375 | 128.0 | 314.99 |
| generation | summary_style | greedy | 1 | n/a | 0.813 | 0.863 | 128.0 | 350.61 |
| generation | technology | balanced_s1 | 2 | n/a | 0.721 | 0.524 | 128.0 | 315.53 |
| generation | technology | balanced_s2 | 2 | n/a | 0.670 | 0.421 | 128.0 | 315.12 |
| generation | technology | greedy | 2 | n/a | 0.971 | 0.875 | 128.0 | 354.19 |

完整生成内容保存在 `generations.jsonl`；人工评分请填写 `manual_review.csv`。
