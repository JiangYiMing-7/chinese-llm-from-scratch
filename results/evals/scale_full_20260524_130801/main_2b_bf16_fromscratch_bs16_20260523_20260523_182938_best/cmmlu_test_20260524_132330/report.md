# 中文多选评测报告

## 配置

- benchmark: `cmmlu`
- split: `test`
- checkpoint: `results/runs/main_2b_bf16_fromscratch_bs16_20260523_20260523_182938/checkpoints/best.pt`
- dtype/device: `bfloat16` / `cuda`
- length normalize: `False`

## 总体结果

| examples | correct | accuracy |
|---:|---:|---:|
| 11582 | 2872 | 0.2480 |

## 分学科

| subject | examples | accuracy |
|---|---:|---:|
| agronomy | 169 | 0.2130 |
| anatomy | 148 | 0.2703 |
| ancient_chinese | 164 | 0.2378 |
| arts | 160 | 0.2437 |
| astronomy | 165 | 0.2667 |
| business_ethics | 209 | 0.2488 |
| chinese_civil_service_exam | 160 | 0.2125 |
| chinese_driving_rule | 131 | 0.2366 |
| chinese_food_culture | 136 | 0.2279 |
| chinese_foreign_policy | 107 | 0.2897 |
| chinese_history | 323 | 0.2167 |
| chinese_literature | 204 | 0.2598 |
| chinese_teacher_qualification | 179 | 0.2458 |
| clinical_knowledge | 237 | 0.2321 |
| college_actuarial_science | 106 | 0.1981 |
| college_education | 107 | 0.1776 |
| college_engineering_hydrology | 106 | 0.3019 |
| college_law | 108 | 0.2500 |
| college_mathematics | 105 | 0.2476 |
| college_medical_statistics | 106 | 0.2642 |
| college_medicine | 273 | 0.2564 |
| computer_science | 204 | 0.2255 |
| computer_security | 171 | 0.2807 |
| conceptual_physics | 147 | 0.2721 |
| construction_project_management | 139 | 0.2518 |
| economics | 159 | 0.2453 |
| education | 163 | 0.2577 |
| electrical_engineering | 172 | 0.2035 |
| elementary_chinese | 252 | 0.2341 |
| elementary_commonsense | 198 | 0.2475 |
| elementary_information_and_technology | 238 | 0.2269 |
| elementary_mathematics | 230 | 0.1913 |
| ethnology | 135 | 0.2593 |
| food_science | 143 | 0.2797 |
| genetics | 176 | 0.2614 |
| global_facts | 149 | 0.2617 |
| high_school_biology | 169 | 0.2426 |
| high_school_chemistry | 132 | 0.2576 |
| high_school_geography | 118 | 0.2288 |
| high_school_mathematics | 164 | 0.2378 |
| high_school_physics | 110 | 0.2091 |
| high_school_politics | 143 | 0.2867 |
| human_sexuality | 126 | 0.2302 |
| international_law | 185 | 0.2541 |
| journalism | 172 | 0.2733 |
| jurisprudence | 411 | 0.2749 |
| legal_and_moral_basis | 214 | 0.2477 |
| logical | 123 | 0.1951 |
| machine_learning | 122 | 0.2213 |
| management | 210 | 0.2667 |
| marketing | 180 | 0.2389 |
| marxist_theory | 189 | 0.2593 |
| modern_chinese | 116 | 0.2241 |
| nutrition | 145 | 0.3103 |
| philosophy | 105 | 0.2571 |
| professional_accounting | 175 | 0.2400 |
| professional_law | 211 | 0.2986 |
| professional_medicine | 376 | 0.2420 |
| professional_psychology | 232 | 0.2845 |
| public_relations | 174 | 0.2816 |
| security_study | 135 | 0.2222 |
| sociology | 226 | 0.2611 |
| sports_science | 165 | 0.2364 |
| traditional_chinese_medicine | 185 | 0.2595 |
| virology | 169 | 0.2189 |
| world_history | 161 | 0.2671 |
| world_religions | 160 | 0.2562 |
