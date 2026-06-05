# 中文语言建模评测报告

## 配置

- checkpoint: `results/runs/pilot_20k_20x_bf16_20260524_20260524_091754/checkpoints/best.pt`
- tokenizer: `data/tokenizer/zh_bpe_20k.model`
- device: `cuda`
- dtype: `bfloat16`

## 内部固定 Validation

| batches | tokens | loss | ppl | token acc | elapsed sec |
|---:|---:|---:|---:|---:|---:|
| 10 | 40960 | 3.9906 | 54.09 | 0.3117 | 0.4 |
