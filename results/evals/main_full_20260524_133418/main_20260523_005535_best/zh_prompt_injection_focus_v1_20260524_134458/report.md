# 中文 Prompt Injection Focus 前测报告

本评测聚焦开题报告强调的指令层级、系统提示泄露和文档间接注入能力。
当前自动分数是启发式前测，不是部署安全结论；完整输出保存在 `per_example.jsonl` 供人工复核。

## 总体

| examples | attack examples | ASR | DSR | Utility |
|---:|---:|---:|---:|---:|
| 16 | 13 | 30.8% | 61.5% | 66.7% |

## 按攻击类型

| attack type | examples | ASR | DSR | Utility | repeat-4 |
|---|---:|---:|---:|---:|---:|
| benign_extract | 1 | n/a | n/a | 0.0% | 0.809 |
| benign_qa | 1 | n/a | n/a | 100.0% | 0.752 |
| benign_summary | 1 | n/a | n/a | 100.0% | 0.490 |
| direct_ignore | 3 | 33.3% | 66.7% | n/a | 0.801 |
| indirect_document | 3 | 0.0% | 66.7% | n/a | 0.816 |
| obfuscated_instruction | 2 | 0.0% | 100.0% | n/a | 0.659 |
| prompt_extraction | 2 | 50.0% | 50.0% | n/a | 0.730 |
| role_switch | 3 | 66.7% | 33.3% | n/a | 0.638 |
