# 中文多选评测报告

## 配置

- benchmark: `cmmlu`
- split: `test`
- checkpoint: `results/runs/tiny_20k_20x_bf16_20260524_20260524_090918/checkpoints/best.pt`
- dtype/device: `bfloat16` / `cuda`
- length normalize: `False`

## 总体结果

| examples | correct | accuracy |
|---:|---:|---:|
| 11582 | 2909 | 0.2512 |

## 分学科

| subject | examples | accuracy |
|---|---:|---:|
| agronomy | 169 | 0.2308 |
| anatomy | 148 | 0.2568 |
| ancient_chinese | 164 | 0.2561 |
| arts | 160 | 0.2562 |
| astronomy | 165 | 0.2485 |
| business_ethics | 209 | 0.2392 |
| chinese_civil_service_exam | 160 | 0.2437 |
| chinese_driving_rule | 131 | 0.2214 |
| chinese_food_culture | 136 | 0.2574 |
| chinese_foreign_policy | 107 | 0.2523 |
| chinese_history | 323 | 0.2539 |
| chinese_literature | 204 | 0.2598 |
| chinese_teacher_qualification | 179 | 0.2514 |
| clinical_knowledge | 237 | 0.2532 |
| college_actuarial_science | 106 | 0.2453 |
| college_education | 107 | 0.2991 |
| college_engineering_hydrology | 106 | 0.3019 |
| college_law | 108 | 0.2130 |
| college_mathematics | 105 | 0.2381 |
| college_medical_statistics | 106 | 0.2547 |
| college_medicine | 273 | 0.2418 |
| computer_science | 204 | 0.2500 |
| computer_security | 171 | 0.2515 |
| conceptual_physics | 147 | 0.2517 |
| construction_project_management | 139 | 0.2662 |
| economics | 159 | 0.2579 |
| education | 163 | 0.2270 |
| electrical_engineering | 172 | 0.2267 |
| elementary_chinese | 252 | 0.2738 |
| elementary_commonsense | 198 | 0.2424 |
| elementary_information_and_technology | 238 | 0.2731 |
| elementary_mathematics | 230 | 0.2783 |
| ethnology | 135 | 0.2519 |
| food_science | 143 | 0.2448 |
| genetics | 176 | 0.2500 |
| global_facts | 149 | 0.2483 |
| high_school_biology | 169 | 0.2485 |
| high_school_chemistry | 132 | 0.2500 |
| high_school_geography | 118 | 0.2542 |
| high_school_mathematics | 164 | 0.2500 |
| high_school_physics | 110 | 0.2545 |
| high_school_politics | 143 | 0.2587 |
| human_sexuality | 126 | 0.2381 |
| international_law | 185 | 0.2649 |
| journalism | 172 | 0.2558 |
| jurisprudence | 411 | 0.2457 |
| legal_and_moral_basis | 214 | 0.2570 |
| logical | 123 | 0.2602 |
| machine_learning | 122 | 0.2459 |
| management | 210 | 0.2476 |
| marketing | 180 | 0.2778 |
| marxist_theory | 189 | 0.2540 |
| modern_chinese | 116 | 0.2500 |
| nutrition | 145 | 0.2552 |
| philosophy | 105 | 0.2476 |
| professional_accounting | 175 | 0.2514 |
| professional_law | 211 | 0.2417 |
| professional_medicine | 376 | 0.2500 |
| professional_psychology | 232 | 0.2328 |
| public_relations | 174 | 0.2586 |
| security_study | 135 | 0.2370 |
| sociology | 226 | 0.2478 |
| sports_science | 165 | 0.2424 |
| traditional_chinese_medicine | 185 | 0.2324 |
| virology | 169 | 0.2485 |
| world_history | 161 | 0.2484 |
| world_religions | 160 | 0.2562 |
