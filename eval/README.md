# Chinese Pretraining Evaluation

This directory contains two layers of evaluation code:

1. `evaluate_pretrain_chinese.py` is the original handcrafted baseline
   evaluator and is kept for backward compatibility.
2. The `*_v1` scripts are the reproducible Chinese-first suite used for
   Tiny/Pilot/Main scale comparison and later Base/SFT/Safety-SFT comparison.

All new scripts create a fresh timestamped directory under `results/evals/` by
default and refuse to reuse an existing output directory.

## Suite Layout

| Script | Purpose |
|---|---|
| `evaluate_lm.py` | Internal validation plus external Chinese LM `loss/PPL/BPC/BPB` |
| `evaluate_multiple_choice.py` | C-Eval, CMMLU, or JSONL MCQ conditional likelihood scoring |
| `evaluate_generation.py` | Frozen Chinese generation, instruction sanity, and multilingual probe prompts |
| `evaluate_prompt_injection_focus.py` | Small Chinese prompt-injection focus set for pre/post-training comparison |
| `compare_scales.py` | Aggregate Tiny/Pilot/Main result directories into a report |
| `run_eval_suite.sh` | Auditable smoke/full server entrypoint |

Prompt files:

```text
prompts/zh_generation_v1.jsonl
prompts/zh_instruction_sanity_v1.jsonl
prompts/zh_prompt_injection_focus_v1.jsonl
prompts/multilingual_probe_v1.jsonl
prompts/pretrain_zh_baseline.jsonl     # original baseline, preserved
```

The main suite is Chinese-first. The multilingual probe is only an appendix
boundary check and should not be used as the main model-quality conclusion.

## Original Baseline

`evaluate_pretrain_chinese.py` covers:

- `language`: Chinese continuation quality, coherence, mixed Chinese/English
  text, factual and expository continuation.
- `instruction`: preliminary format, extraction, translation and constraint
  following tests.
- `safety`: preliminary refusal, safe guidance and over-refusal controls.

The pretrained model is not instruction tuned. Instruction-following, prompt
injection, and safety results should be treated as baselines for comparison
with later SFT and alignment runs, not as deployment readiness tests.

## Run The New Suite On Server

Smoke test with the 0.5B-token baseline:

```bash
cd /root/autodl-tmp/chinese-llm-from-scratch
source /root/miniconda3/etc/profile.d/conda.sh
conda activate chinese-llm

DEVICE=cuda DTYPE=bfloat16 bash eval/run_eval_suite.sh \
  results/runs/main_20260523_005535/checkpoints/best.pt \
  data/tokenizer/zh_bpe_20k.model \
  data/processed/pretrain_val_tokens.bin \
  results/evals \
  smoke
```

Full internal validation and prompt suite:

```bash
DEVICE=cuda DTYPE=float32 bash eval/run_eval_suite.sh \
  results/runs/main_20260523_005535/checkpoints/best.pt \
  data/tokenizer/zh_bpe_20k.model \
  data/processed/pretrain_val_tokens.bin \
  results/evals \
  full
```

If official benchmark files have been frozen locally, set:

```bash
CEVAL_DIR=data/eval/ceval
CMMLU_DIR=data/eval/cmmlu
EXTERNAL_JSONL=data/eval/zh_external_lm_v1.jsonl
```

before running `run_eval_suite.sh`.

## Run On Server

Run with `best.pt`, the matching tokenizer, and the fixed validation split:

```bash
cd /root/autodl-tmp/chinese-llm-from-scratch

python eval/evaluate_pretrain_chinese.py \
  --checkpoint results/runs/main_20260523_005535/checkpoints/best.pt \
  --tokenizer data/tokenizer/zh_bpe_20k.model \
  --validation-bin data/processed/pretrain_val_tokens.bin \
  --validation-batch-size 16 \
  --validation-batches 100 \
  --profiles greedy balanced \
  --device cuda \
  --dtype bfloat16 \
  --max-new-tokens 128 \
  --output-dir results/evals/main_20260523_005535_best/legacy_zh_baseline_$(date +%Y%m%d_%H%M%S)
```

For a quick smoke test:

```bash
python eval/evaluate_pretrain_chinese.py \
  --checkpoint results/runs/main_20260523_005535/checkpoints/best.pt \
  --tokenizer data/tokenizer/zh_bpe_20k.model \
  --suites language instruction safety \
  --profiles greedy \
  --limit 3 \
  --max-new-tokens 32 \
  --device cuda \
  --dtype bfloat16
```

## Outputs

The output directory contains:

```text
generations.jsonl   full generated records and diagnostics
case_scores.csv     compact automatic-score table
summary.json        grouped metrics and optional validation PPL
report.md           readable prompt/completion report for manual review
```

Language generations require human review. Safety keyword scoring only flags
likely refusal/support/risk patterns; manually review each safety completion.
