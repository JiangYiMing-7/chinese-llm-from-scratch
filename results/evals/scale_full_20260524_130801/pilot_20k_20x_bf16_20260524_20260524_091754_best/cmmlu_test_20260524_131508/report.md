# 中文多选评测报告

## 配置

- benchmark: `cmmlu`
- split: `test`
- checkpoint: `results/runs/pilot_20k_20x_bf16_20260524_20260524_091754/checkpoints/best.pt`
- dtype/device: `bfloat16` / `cuda`
- length normalize: `False`

## 总体结果

| examples | correct | accuracy |
|---:|---:|---:|
| 11582 | 2913 | 0.2515 |

## 分学科

| subject | examples | accuracy |
|---|---:|---:|
| agronomy | 169 | 0.2367 |
| anatomy | 148 | 0.2635 |
| ancient_chinese | 164 | 0.2500 |
| arts | 160 | 0.2625 |
| astronomy | 165 | 0.2485 |
| business_ethics | 209 | 0.2488 |
| chinese_civil_service_exam | 160 | 0.2812 |
| chinese_driving_rule | 131 | 0.2595 |
| chinese_food_culture | 136 | 0.2353 |
| chinese_foreign_policy | 107 | 0.2430 |
| chinese_history | 323 | 0.2260 |
| chinese_literature | 204 | 0.2843 |
| chinese_teacher_qualification | 179 | 0.2235 |
| clinical_knowledge | 237 | 0.2152 |
| college_actuarial_science | 106 | 0.2736 |
| college_education | 107 | 0.3084 |
| college_engineering_hydrology | 106 | 0.3019 |
| college_law | 108 | 0.2130 |
| college_mathematics | 105 | 0.2286 |
| college_medical_statistics | 106 | 0.2358 |
| college_medicine | 273 | 0.2381 |
| computer_science | 204 | 0.2402 |
| computer_security | 171 | 0.2515 |
| conceptual_physics | 147 | 0.2517 |
| construction_project_management | 139 | 0.2230 |
| economics | 159 | 0.2390 |
| education | 163 | 0.2454 |
| electrical_engineering | 172 | 0.2442 |
| elementary_chinese | 252 | 0.2698 |
| elementary_commonsense | 198 | 0.2475 |
| elementary_information_and_technology | 238 | 0.2689 |
| elementary_mathematics | 230 | 0.2826 |
| ethnology | 135 | 0.2444 |
| food_science | 143 | 0.2937 |
| genetics | 176 | 0.2443 |
| global_facts | 149 | 0.2416 |
| high_school_biology | 169 | 0.2604 |
| high_school_chemistry | 132 | 0.2424 |
| high_school_geography | 118 | 0.2458 |
| high_school_mathematics | 164 | 0.2744 |
| high_school_physics | 110 | 0.2364 |
| high_school_politics | 143 | 0.2517 |
| human_sexuality | 126 | 0.2698 |
| international_law | 185 | 0.2649 |
| journalism | 172 | 0.2616 |
| jurisprudence | 411 | 0.2384 |
| legal_and_moral_basis | 214 | 0.2570 |
| logical | 123 | 0.2602 |
| machine_learning | 122 | 0.2459 |
| management | 210 | 0.2476 |
| marketing | 180 | 0.2500 |
| marxist_theory | 189 | 0.2434 |
| modern_chinese | 116 | 0.2069 |
| nutrition | 145 | 0.2345 |
| philosophy | 105 | 0.2667 |
| professional_accounting | 175 | 0.2514 |
| professional_law | 211 | 0.2512 |
| professional_medicine | 376 | 0.2527 |
| professional_psychology | 232 | 0.2500 |
| public_relations | 174 | 0.2529 |
| security_study | 135 | 0.2593 |
| sociology | 226 | 0.2655 |
| sports_science | 165 | 0.2303 |
| traditional_chinese_medicine | 185 | 0.2649 |
| virology | 169 | 0.2308 |
| world_history | 161 | 0.2733 |
| world_religions | 160 | 0.2812 |
