# Chinese LLM From Scratch

## 项目结构

```text
.
├── model/                 # 模型架构
│   ├── attention.py       # Causal Self-Attention
│   ├── block.py           # Transformer Block
│   ├── causal_lm.py       # Causal LM 主模型
│   ├── config.py          # 模型配置
│   ├── ffn.py             # SwiGLU FFN
│   ├── generation.py      # 自回归生成
│   ├── norm.py            # RMSNorm
│   └── rope.py            # RoPE 位置编码
├── pretrain/              # 预训练相关代码
│   ├── checkpoint.py      # 保存与加载 checkpoint
│   ├── dataset.py         # token 序列 Dataset
│   ├── optim.py           # AdamW 与学习率调度
│   ├── pretrain.py        # 预训练主循环
│   ├── train_pretrain.py  # 通用预训练入口
│   ├── train_tiny.py      # Tiny 配置快捷入口
│   └── sample.py          # 从 checkpoint 采样 token
├── data/
│   ├── scripts/           # 数据流水线脚本
│   │   ├── download_data.py
│   │   ├── clean_corpus.py
│   │   ├── mix_datasets.py
│   │   ├── train_tokenizer.py
│   │   ├── pretrain_dataset.py
│   │   ├── token_stats.py
│   │   └── verify_pipeline.py
│   ├── raw/               # 原始语料，默认不入库
│   ├── processed/         # token 数据，默认不入库
│   ├── tokenizer/         # tokenizer 模型，默认不入库
│   └── requirements.txt   # 数据处理依赖
├── configs/               # tiny / pilot / main 训练配置
├── tests/                 # 单元测试
├── utils/                 # 日志、随机种子、参数统计等工具
├── results/               # run 记录、评测结果和图表
├── checkpoints/           # 可选独立 checkpoint 目录，默认 checkpoint 随 run 保存
└── requirements.txt       # 模型训练基础依赖
```

## 环境准备

安装模型训练基础依赖：

```bash
python -m pip install -r requirements.txt
```

实际跑预训练时还需要 tokenizer 一致性检查；如果要训练 tokenizer、构建 token 数据或直接使用默认配置预训练，也安装数据处理依赖：

```bash
python -m pip install -r data/requirements.txt
```

## 测试与自检

### 1. 模型单元测试

运行单元测试，确认 attention、RoPE、RMSNorm、模型前向、训练 step 和 checkpoint 读写正常：

```bash
pytest tests -q
```

通过时应看到类似：

```text
8 passed
```

### 2. 模块导入检查

如果改过目录结构或 import 路径，可以运行：

```bash
python -c "from model.causal_lm import CausalLM; from model.config import tiny_config; from pretrain.pretrain import pretrain; print('imports ok')"
```

### 3. 语法编译检查

检查所有 Python 文件是否存在语法错误：

```bash
python -m compileall -q model pretrain utils data/scripts tests
```

### 4. 数据流水线端到端检查

该脚本会在临时目录里生成小规模合成中文语料，并串行跑完清洗、混合、tokenizer 训练、token 统计和预训练 token 数据构建。它适合在处理真实大语料前确认数据模块是通的：

```bash
python data/scripts/verify_pipeline.py \
  --workdir /tmp/chinese_llm_verify \
  --num-docs 100 \
  --verify-vocab-size 512 \
  --verify-ctx 32
```

注意：该检查需要安装 `data/requirements.txt` 中的数据处理依赖。

## 预训练流程

下面以 Tiny 小范围预训练为例。完整流程是：

```text
中文 JSONL 语料
  -> 训练 tokenizer
  -> 构建 token id 数据
  -> 启动预训练
  -> 保存 checkpoint
```

### 1. 准备语料

预训练需要 JSONL 格式中文语料，每行一个对象，至少包含 `text` 字段：

```jsonl
{"text": "这里是一段中文训练文本。"}
{"text": "这里是另一段中文训练文本。"}
```

推荐放在：

```text
data/processed/mixed_corpus.jsonl
```

Tiny 小实验建议：

- 只打通流程：1-5 MB 中文文本，词表可用 2K-4K。
- 小范围预训练：20-100 MB 中文文本，建议 8K 词表。
- `configs/tiny.yaml` 当前默认 `vocab_size: 8000`，因此 tokenizer 也应训练为 8000 词表。

如果还没有真实语料，可以先用 `data/scripts/download_data.py` 下载 debug/pilot 语料，再经过清洗和混合脚本生成 `mixed_corpus.jsonl`。

### 2. 训练 tokenizer

Tiny 默认使用 8K tokenizer：

```bash
python data/scripts/train_tokenizer.py \
  --input data/processed/mixed_corpus.jsonl \
  --model-prefix data/tokenizer/zh_bpe_8k \
  --vocab-size 8000
```

运行后会生成：

```text
data/tokenizer/zh_bpe_8k.model
data/tokenizer/zh_bpe_8k.vocab
data/tokenizer/zh_bpe_8k_stats.json
```

重要：tokenizer 的词表大小必须和配置里的 `model.vocab_size` 一致。

### 3. 构建预训练 token 数据

将 JSONL 文本编码为连续 token id 序列：

```bash
python data/scripts/pretrain_dataset.py \
  --mode build \
  --input data/processed/mixed_corpus.jsonl \
  --tokenizer data/tokenizer/zh_bpe_8k.model \
  --output data/processed/debug_tokens.pt \
  --val-output data/processed/debug_val_tokens.pt \
  --val-ratio 0.01 \
  --output-format pt \
  --context-length 64
```

运行后会生成：

```text
data/processed/debug_tokens.pt
data/processed/debug_val_tokens.pt
```

该路径与 `configs/tiny.yaml` 中的默认配置一致：

```yaml
data:
  train_path: data/processed/debug_tokens.pt
  val_path: data/processed/debug_val_tokens.pt
  tokenizer_path: data/tokenizer/zh_bpe_8k.model
  block_size: 64
```

### 4. 启动 Tiny 预训练

通用入口：

```bash
python pretrain/train_pretrain.py --config configs/tiny.yaml
```

Tiny 快捷入口：

```bash
python pretrain/train_tiny.py
```

训练时终端会打印训练 loss、滑动平均 loss、ppl、token accuracy、grad norm、learning rate、吞吐速度、显存占用和 GPU 利用率；每次验证会打印验证 loss、验证 ppl 和验证 token accuracy。

每次启动训练都会在 `results/runs/` 下创建一个新的 run 目录，例如：

```text
results/runs/tiny_20260520_153012/
```

该目录内包含：

```text
run_info.json   # 本次训练基本信息：时间、配置、超参数、数据路径、设备信息等
config.yaml     # 本次训练实际使用的配置快照
metrics.jsonl   # 训练/验证/checkpoint 事件日志
summary.json    # 训练结束摘要
```

checkpoint 默认保存到本次 run 目录内：

```text
results/runs/tiny_20260520_153012/checkpoints/ckpt.pt
```

验证集 loss 刷新最低值时，还会额外保存：

```text
results/runs/tiny_20260520_153012/checkpoints/best.pt
```

### 5. 恢复训练

如果要从已有 checkpoint 继续训练，修改 `configs/tiny.yaml`：

```yaml
train:
  resume_from: results/runs/tiny_20260520_153012/checkpoints/ckpt.pt
```

然后重新运行：

```bash
python pretrain/train_pretrain.py --config configs/tiny.yaml
```

### 6. 常见检查点

- `data/processed/mixed_corpus.jsonl` 必须存在，且每行含有 `text` 字段。
- `data/tokenizer/zh_bpe_8k.model` 必须存在。
- `data/processed/debug_tokens.pt` 必须存在。
- `data/processed/debug_val_tokens.pt` 必须存在。
- `configs/tiny.yaml` 里的 `model.vocab_size` 要等于 tokenizer 词表大小。
- `configs/tiny.yaml` 里的 `data.tokenizer_path` 要指向本次构建 token 数据时使用的 tokenizer。
- `configs/tiny.yaml` 里的 `data.block_size` 不应超过 `model.max_seq_len`。

## 训练日志与指标

每次启动训练都会生成独立 run 目录，默认路径格式为：

```text
results/runs/{run_name}_{YYYYMMDD_HHMMSS}/
```

Tiny / Pilot / Main 的默认 `run_name` 分别是 `tiny`、`pilot`、`main`。run 目录内主要文件：

- `run_info.json`：本次训练基本信息，包括启动时间、配置路径、超参数、数据路径、tokenizer、checkpoint 路径、设备信息。
- `config.yaml`：本次训练实际使用的配置快照，包含自动生成的 `run_id`、`run_dir`、`metrics_path`。
- `metrics.jsonl`：训练、验证、checkpoint 等事件日志，适合用 pandas 统计和画图。
- `summary.json`：训练结束摘要，包括结束时间、总步数、累计 tokens、best val loss、checkpoint 路径；训练中途失败时会记录 `status: failed` 和错误信息。

`metrics.jsonl` 每行是一个 JSON 事件。常见事件包括：

- `run_start`：训练启动信息，包括配置路径、device、参数量、样本数、batch size。
- `train`：训练 step 指标。
- `eval`：验证集指标。
- `checkpoint`：checkpoint 保存事件。
- `run_end`：训练结束信息。

`train` 事件中的核心字段：

- `step`：当前训练步数。
- `loss`：当前 batch 的 next-token cross entropy loss。
- `loss_ema`：训练 loss 的指数滑动平均，比单步 loss 更平滑。
- `ppl`：perplexity，由 `exp(loss)` 得到。
- `token_acc`：当前 batch 的 next-token top-1 accuracy。
- `grad_norm`：梯度范数，用于观察梯度爆炸或训练不稳定。
- `grad_clipped`：当前 step 是否触发梯度裁剪。
- `lr`：当前 learning rate。
- `tokens_seen`：累计训练 token 数。
- `tokens_per_sec`：当前 step 吞吐。
- `samples_per_sec`：当前 step 样本吞吐。
- `step_time_sec`：当前 step 耗时。
- `hw_device_type`：当前训练设备类型，如 `cuda`、`mps`、`cpu`。
- `hw_device_name`：设备名称。
- `hw_memory_allocated_mb`：PyTorch 当前已分配显存，CUDA/MPS 可用。
- `hw_memory_reserved_mb`：PyTorch 当前保留显存，CUDA/MPS 可用。
- `hw_memory_peak_allocated_mb`：本次训练的 PyTorch 峰值显存，CUDA 可用。
- `hw_memory_total_mb`：设备总显存，CUDA 可用。
- `hw_memory_allocated_ratio`：PyTorch 已分配显存占总显存比例，CUDA 可用。
- `hw_memory_reserved_ratio`：PyTorch 保留显存占总显存比例，CUDA 可用。
- `hw_gpu_utilization_percent`：GPU 利用率，NVIDIA 且存在 `nvidia-smi` 时可用。
- `hw_memory_utilization_percent`：显存带宽/显存控制器利用率，NVIDIA 且存在 `nvidia-smi` 时可用。
- `hw_memory_used_external_mb`：`nvidia-smi` 看到的整卡显存占用。
- `hw_memory_total_external_mb`：`nvidia-smi` 看到的整卡总显存。
- `hw_external_sampled`：当前行是否执行了外部硬件采样。
- `hw_external_source`：外部硬件采样来源，目前为 `nvidia-smi`。

`hardware_metrics_interval` 控制外部 GPU 利用率采样频率，默认每 10 step 采样一次。PyTorch 显存字段会随每条 `train` 事件记录；如果当前设备不支持某个指标，对应字段会是 `null`，便于后续用 pandas 统一读取。

`eval` 事件中的核心字段：

- `val_loss`：验证集平均 loss。
- `val_ppl`：验证集 perplexity。
- `val_token_acc`：验证集 next-token top-1 accuracy。
- `val_tokens`：本次验证覆盖的 token 数。
- `val_batches`：本次验证使用的 batch 数。

可以用下面的命令查看最近一次 Tiny 训练记录：

```bash
latest_run=$(ls -td results/runs/tiny_* | head -n 1)
tail -n 5 "$latest_run/metrics.jsonl"
```

查看本次训练基本信息：

```bash
cat "$latest_run/run_info.json"
```

## 配置说明

当前保留三组配置：

- `configs/tiny.yaml`：小模型快速调试，默认读取 `data/processed/debug_tokens.pt` 和 `data/processed/debug_val_tokens.pt`。
- `configs/pilot.yaml`：流程验证，默认读取 `data/processed/pilot_tokens.bin` 和 `data/processed/pilot_val_tokens.bin`。
- `configs/main.yaml`：主实验配置，默认读取 `data/processed/pretrain_tokens.bin` 和 `data/processed/pretrain_val_tokens.bin`。

checkpoint 默认写入每次训练自己的 run 目录：

```text
results/runs/{run_name}_{YYYYMMDD_HHMMSS}/checkpoints/
```

训练记录默认写入：

```text
results/runs/
```

## 后续目录预留

如果后面要继续做后训练和评测，建议新增：

```text
finetune/   # SFT、安全 SFT、后训练 Dataset
eval/       # PPL、生成样例、安全评测、C-Eval 等
```

## 本轮整理与增强记录

- 重整项目结构：删除旧的 `src/`、`llm/`、`docs/` 分散目录，将核心代码整理为 `model/`、`pretrain/`、`data/scripts/`、`utils/`、`tests/`。
- 统一训练入口和配置：预训练入口调整为 `pretrain/train_pretrain.py` 和 `pretrain/train_tiny.py`，配置集中在 `configs/`。
- 完善数据流程说明：README 补充从零下载数据、训练 tokenizer、构建 token 数据、启动 Tiny 预训练的完整流程。
- 增加固定验证集支持：`data/scripts/pretrain_dataset.py` 支持 `--val-output` 和 `--val-ratio`，可一次生成 train/val token 文件。
- 绑定 tokenizer：配置中新增 `data.tokenizer_path`，checkpoint 会保存完整配置，方便后续确认 checkpoint 对应的 tokenizer。
- 增加启动前检查：训练开始前检查 tokenizer 词表大小、token 文件 metadata、`block_size <= max_seq_len`、train/val 路径是否冲突。
- 增强训练记录管理：每次启动训练都会在 `results/runs/` 下创建独立 run 目录，包含 `run_info.json`、`config.yaml`、`metrics.jsonl`、`summary.json`；训练中途失败也会写入 `summary.json` 和 `run_error` 事件。
- 增强训练指标记录：`metrics.jsonl` 覆盖 loss、ppl、token accuracy、grad norm、learning rate、吞吐、累计 tokens、验证集指标、checkpoint 事件等。
- 增加 best checkpoint：除 `ckpt.pt` 外，验证集 loss 刷新最低值时保存 `best.pt`。
- 增加硬件监控：记录 CUDA/MPS 显存；NVIDIA 环境下通过 `nvidia-smi` 按间隔记录 GPU 利用率和显存利用率。
- 补充测试：单元测试覆盖训练 step、验证指标、checkpoint 额外状态、run 目录、失败摘要与 metrics JSONL 生成，目前 `pytest tests -q` 为 `8 passed`。







