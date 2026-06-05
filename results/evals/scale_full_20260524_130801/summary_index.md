# Evaluation Run Index

- root: `results/evals/scale_full_20260524_130801`
- rows: `21`

| checkpoint | suite | n/examples | tokens | loss | ppl | token acc | acc | ASR | DSR | Utility |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main_2b_bf16_fromscratch_bs16_20260523_20260523_182938_best | zh_mcq_v1 | 12342 |  |  |  |  | 0.2535 |  |  |  |
| main_2b_bf16_fromscratch_bs16_20260523_20260523_182938_best | zh_mcq_v1 | 11582 |  |  |  |  | 0.2480 |  |  |  |
| main_2b_bf16_fromscratch_bs16_20260523_20260523_182938_best | multilingual_probe_v1 | 4 |  |  |  |  |  |  |  |  |
| main_2b_bf16_fromscratch_bs16_20260523_20260523_182938_best | zh_generation_v1 | 60 |  |  |  |  |  |  |  |  |
| main_2b_bf16_fromscratch_bs16_20260523_20260523_182938_best | zh_instruction_sanity_v1 | 12 |  |  |  |  |  |  |  |  |
| main_2b_bf16_fromscratch_bs16_20260523_20260523_182938_best | internal_validation |  | 40187392 | 3.5768 | 35.7579 | 0.3553 |  |  |  |  |
| main_2b_bf16_fromscratch_bs16_20260523_20260523_182938_best | zh_prompt_injection_focus_v1 | 16 |  |  |  |  |  | 0.3846 | 0.6154 | 0.6667 |
| pilot_20k_20x_bf16_20260524_20260524_091754_best | zh_mcq_v1 | 12342 |  |  |  |  | 0.2383 |  |  |  |
| pilot_20k_20x_bf16_20260524_20260524_091754_best | zh_mcq_v1 | 11582 |  |  |  |  | 0.2515 |  |  |  |
| pilot_20k_20x_bf16_20260524_20260524_091754_best | multilingual_probe_v1 | 4 |  |  |  |  |  |  |  |  |
| pilot_20k_20x_bf16_20260524_20260524_091754_best | zh_generation_v1 | 60 |  |  |  |  |  |  |  |  |
| pilot_20k_20x_bf16_20260524_20260524_091754_best | zh_instruction_sanity_v1 | 12 |  |  |  |  |  |  |  |  |
| pilot_20k_20x_bf16_20260524_20260524_091754_best | internal_validation |  | 40187392 | 3.9417 | 51.5064 | 0.3177 |  |  |  |  |
| pilot_20k_20x_bf16_20260524_20260524_091754_best | zh_prompt_injection_focus_v1 | 16 |  |  |  |  |  | 0.3846 | 0.6154 | 0.6667 |
| tiny_20k_20x_bf16_20260524_20260524_090918_best | zh_mcq_v1 | 12342 |  |  |  |  | 0.2307 |  |  |  |
| tiny_20k_20x_bf16_20260524_20260524_090918_best | zh_mcq_v1 | 11582 |  |  |  |  | 0.2512 |  |  |  |
| tiny_20k_20x_bf16_20260524_20260524_090918_best | multilingual_probe_v1 | 4 |  |  |  |  |  |  |  |  |
| tiny_20k_20x_bf16_20260524_20260524_090918_best | zh_generation_v1 | 60 |  |  |  |  |  |  |  |  |
| tiny_20k_20x_bf16_20260524_20260524_090918_best | zh_instruction_sanity_v1 | 12 |  |  |  |  |  |  |  |  |
| tiny_20k_20x_bf16_20260524_20260524_090918_best | internal_validation |  | 40187392 | 4.7465 | 115.1790 | 0.2427 |  |  |  |  |
| tiny_20k_20x_bf16_20260524_20260524_090918_best | zh_prompt_injection_focus_v1 | 16 |  |  |  |  |  | 0.0769 | 0.9231 | 0.0000 |
