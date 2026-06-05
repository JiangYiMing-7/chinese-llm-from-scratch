# 中文语言建模评测报告

## 配置

- checkpoint: `results/runs/tiny_20k_20x_bf16_20260524_20260524_090918/checkpoints/best.pt`
- tokenizer: `data/tokenizer/zh_bpe_20k.model`
- device: `cuda`
- dtype: `bfloat16`

## 内部固定 Validation

| batches | tokens | loss | ppl | token acc | elapsed sec |
|---:|---:|---:|---:|---:|---:|
| 10 | 40960 | 4.7844 | 119.63 | 0.2408 | 0.4 |
