# 中文语言建模评测报告

## 配置

- checkpoint: `results/runs/main_20260523_005535/checkpoints/best.pt`
- tokenizer: `data/tokenizer/zh_bpe_20k.model`
- device: `cuda`
- dtype: `bfloat16`

## 内部固定 Validation

| batches | tokens | loss | ppl | token acc | elapsed sec |
|---:|---:|---:|---:|---:|---:|
| 10 | 40960 | 3.8797 | 48.41 | 0.3217 | 0.5 |
