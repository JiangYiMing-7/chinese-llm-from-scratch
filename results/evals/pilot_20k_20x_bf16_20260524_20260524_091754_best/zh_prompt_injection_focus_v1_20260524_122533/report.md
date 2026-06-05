# 中文 Prompt Injection Focus 前测报告

本评测聚焦开题报告强调的指令层级、系统提示泄露和文档间接注入能力。
当前自动分数是启发式前测，不是部署安全结论；完整输出保存在 `per_example.jsonl` 供人工复核。

## 总体

| examples | attack examples | ASR | DSR | Utility |
|---:|---:|---:|---:|---:|
| 5 | 5 | 60.0% | 40.0% | n/a |

## 按攻击类型

| attack type | examples | ASR | DSR | Utility | repeat-4 |
|---|---:|---:|---:|---:|---:|
| direct_ignore | 3 | 66.7% | 33.3% | n/a | 0.664 |
| role_switch | 2 | 50.0% | 50.0% | n/a | 0.643 |
