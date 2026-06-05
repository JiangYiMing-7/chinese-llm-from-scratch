# 中文语言建模评测报告

## 配置

- checkpoint: `results/runs/main_2b_bf16_fromscratch_bs16_20260523_20260523_182938/checkpoints/best.pt`
- tokenizer: `data/tokenizer/zh_bpe_20k.model`
- device: `cuda`
- dtype: `bfloat16`

## 内部固定 Validation

| batches | tokens | loss | ppl | token acc | elapsed sec |
|---:|---:|---:|---:|---:|---:|
| 2453 | 40186880 | 3.4065 | 30.16 | 0.3763 | 142.5 |
