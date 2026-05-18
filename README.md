# Chinese LLM From Scratch

**基于 Decoder-only Transformer 的小型中文 LLaMA-like 模型从零构建、后训练与安全性评测**

北京邮电大学 2026 春 自然语言处理课程大作业

**小组成员：** 蒋一鸣（2023211751）、谈亮（2023211884）、龚正硕（2023211729）

---

## 项目概述

从原始中文文本开始，在课程算力范围内从零跑通一个小型中文 LLaMA-like 模型的完整流程：

```
中文语料 → 清洗去重 → Tokenizer 训练 → 预训练 → 普通 SFT → 安全 SFT → 多维评测
```

重点探究：在 100M 参数规模下，小模型能否通过安全 SFT 获得基本的 Prompt Injection 防御能力。

---

## 项目结构

```
.
├── src/
│   ├── model/          # Decoder-only Transformer 实现（RoPE、RMSNorm、SwiGLU 等）
│   ├── tokenizer/      # SentencePiece BPE tokenizer 训练与使用
│   ├── train/          # 预训练脚本
│   ├── eval/           # 评测脚本（PPL、生成样例、C-Eval）
│   └── safety/         # 安全 SFT 数据构造与 Prompt Injection 评测
├── configs/            # 训练配置文件（Tiny / Pilot / Main）
├── scripts/            # 数据处理、启动训练等脚本
├── data/
│   ├── raw/            # 原始语料（gitignore）
│   ├── processed/      # 预处理后的 token 文件（gitignore）
│   └── tokenizer/      # tokenizer 训练数据
├── checkpoints/        # 模型权重（gitignore）
├── results/
│   ├── figures/        # loss 曲线、评测图表
│   └── logs/           # 训练日志（gitignore）
└── docs/               # 开题报告、技术文档
```

---

## 模型配置

| 配置  | vocab | d_model | layers | heads | 参数量    | 用途     |
|-------|-------|---------|--------|-------|-----------|----------|
| Tiny  | 8K    | 256     | 4      | 4     | ~6M       | 调试     |
| Pilot | 16K   | 512     | 8      | 8     | ~35M      | 流程验证 |
| Main  | 20K   | 768     | 12     | 12    | ~100M     | 主实验   |

---

## 分工

| 成员   | 职责 |
|--------|------|
| 蒋一鸣 | 模型结构实现（RoPE、RMSNorm、SwiGLU、Causal Attention、生成接口） |
| 谈亮   | 数据与 Tokenizer（语料清洗、SentencePiece 训练、Dataset/DataLoader） |
| 龚正硕 | 训练评测与后训练（预训练脚本、SFT、Prompt Injection 评测） |

---

## 进度

- [ ] W1：数据处理、tokenizer 训练、代码框架
- [ ] W2：Tiny 模型实现，过拟合验证
- [ ] W3：Pilot 预训练，生成接口
- [ ] W4：Main 100M 预训练
- [ ] W5：普通 SFT + 安全 SFT，消融实验启动
- [ ] W6：消融实验完成，安全评测，报告撰写

---

## 环境依赖

```bash
pip install torch sentencepiece numpy tqdm wandb
```

---

## 参考

- LLaMA: [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)
- Instruction Hierarchy: [arXiv:2404.13208](https://arxiv.org/abs/2404.13208)
- Tensor Trust: [arXiv:2311.01011](https://arxiv.org/abs/2311.01011)
