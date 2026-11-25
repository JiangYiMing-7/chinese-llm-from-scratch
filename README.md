# Chinese LLaMA-style Language Model from Scratch

A decoder-only Chinese language model (539.85M parameters) written in plain PyTorch and pretrained from scratch on 10B tokens, followed by instruction SFT and a safety-SFT study on prompt injection.

This is a three-person course project for *Natural Language Processing* at Beijing University of Posts and Telecommunications (BUPT), April–June 2026. Training ran on 4× NVIDIA RTX PRO 6000 GPUs.

## Architecture

LLaMA-style decoder, implemented in `model/` without `transformers`:

- Pre-norm blocks with **RMSNorm**
- **Rotary position embeddings (RoPE)**, θ = 10000
- **SwiGLU** feed-forward network
- Causal multi-head self-attention, no bias terms
- Tied input/output embeddings
- Sampling-based generation (`model/generation.py`)

The config also supports norm-placement ablations (`pre_norm`, `post_norm`, `pre_output_norm`, `branch_norm`) and a top-1 MoE FFN, used for the 100M-scale experiments in `configs/ablations/`.

### Main model (`configs/main_500m_hf50k_10btok_bs8_bf16_4xpro6000_20260608.yaml`)

| | |
|---|---:|
| Parameters | 539,854,080 |
| Layers | 24 |
| Hidden size | 1280 |
| Attention heads | 20 (head dim 64) |
| FFN hidden size | 3456 |
| Context length | 1024 |
| Vocabulary | 50,000 (BPE, trained on CCI3-HQ) |

Parameter count is logged by the training script (`model params: 539.85 M`) and equals `50000·1280 + 24·(4·1280² + 3·1280·3456 + 2·1280) + 1280`.

## Pipeline

| Stage | Code | Notes |
|---|---|---|
| Corpus preparation | `data/scripts/prepare_cci3hq_corpus.py`, `clean_corpus.py`, `dedup_utils.py` | CCI3-HQ Chinese web text, cleaning and exact/near-duplicate removal |
| Tokenizer | `data/scripts/train_hf_bpe_tokenizer.py` | 50K BPE vocabulary (HF `tokenizers`) |
| Pre-tokenization | `data/scripts/pretrain_dataset_parallel.py` | Packed 1024-token blocks in a flat `.bin` file |
| Pretraining | `pretrain/train_pretrain.py`, `pretrain/pretrain.py` | DDP, BF16, AdamW, cosine schedule with warmup |
| Instruction SFT | `finetune/train_sft.py` | Chat template in `finetune/chat_template.py`; loss on assistant tokens only |
| Safety SFT | same script, `configs/safety_sft_500m_hf50k_ablation_*.yaml` | Three data variants, see results |
| Evaluation | `eval/` | Held-out PPL, C-Eval / CMMLU, Chinese reference tasks, prompt-injection suite |

## Results

All numbers below are taken from run summaries in `results/key/` (copied from the training server; checkpoints and training data are not included).

### Pretraining

| | |
|---|---:|
| Tokens seen | 10,000,007,168 (305,176 steps × 32,768 tokens/step) |
| Global batch | 32 sequences × 1024 tokens |
| Wall-clock | 44.2 h, about 70K tokens/s |
| Best validation loss | 3.1395 |
| Full held-out set (57.7M tokens) | loss 3.1402, PPL 23.11 |

### Safety SFT against prompt injection

Test set: `eval/data/zh_prompt_injection_reading_v2.jsonl`, 600 Chinese reading/search QA items built from CMRC 2018 passages: 400 with an injected instruction (4 attack types × 100), 100 clean, 100 with benign noise. ASR is the fraction of attack items where the model's answer hits the attacker's target.

All four models start from the same instruction-SFT checkpoint. Each safety variant uses 1,500 training examples and 800 steps.

| Model | ASR ↓ | Defense success | Clean success | Benign-noise success |
|---|---:|---:|---:|---:|
| Instruction SFT only (baseline) | 9.50% (38/400) | 70.25% | 100.00% | 74.00% |
| + Safety SFT A: public safety data | 10.00% | 65.25% | 76.00% | 64.00% |
| + Safety SFT B: our own injection data | **0.25%** (1/400) | **90.00%** | 92.00% | 83.00% |
| + Safety SFT C: A + B mixed | **0.25%** | 86.75% | 87.00% | 84.00% |

Public safety data (A) did not help, because its format differs from the injection-in-document setting. Our own data (B) removed almost all successful attacks but cost 8 points on clean items. The hardest attack type was an override instruction appended to the user question (baseline ASR 35%, B 1%).

### Other evaluations

- Instruction SFT raised the mean score on our Chinese reference tasks from 5.07% to 38.97%, while held-out PPL went from 23.11 to 26.73.
- C-Eval and CMMLU stay near chance (25–26%) at all stages. We do not claim knowledge gains at this scale.

## Usage

```bash
pip install -r requirements.txt
pytest                                   # 28 unit tests, CPU only

# pretraining (single GPU, or torchrun for multi-GPU)
python pretrain/train_pretrain.py --config configs/tiny.yaml
torchrun --nproc_per_node 4 pretrain/train_pretrain.py \
    --config configs/main_500m_hf50k_10btok_bs8_bf16_4xpro6000_20260608.yaml

# SFT / safety SFT
python finetune/train_sft.py --config configs/sft_500m_hf50k_normal_v2.yaml
python finetune/train_sft.py --config configs/safety_sft_500m_hf50k_ablation_b_generated_v1.yaml
```

Paths to data, tokenizer and checkpoints are set in each config. Raw corpora, tokenized data and checkpoints are not in this repository.

## Repository layout

```
model/      Transformer implementation (RMSNorm, RoPE, attention, SwiGLU, generation)
pretrain/   pretraining loop, dataset, optimizer, checkpointing
finetune/   SFT data loading, chat template, training
eval/       evaluation scripts, prompts and test sets
data/       corpus, tokenizer and SFT data-building scripts
configs/    all run configs, including ablations
tests/      unit tests
results/key run summaries behind the numbers above
```

## Team

Yiming Jiang, Liang Tan, Zhengshuo Gong
