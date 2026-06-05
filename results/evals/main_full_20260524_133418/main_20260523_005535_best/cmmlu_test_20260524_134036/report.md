# 中文多选评测报告

## 配置

- benchmark: `cmmlu`
- split: `test`
- checkpoint: `results/runs/main_20260523_005535/checkpoints/best.pt`
- dtype/device: `bfloat16` / `cuda`
- length normalize: `False`

## 总体结果

| examples | correct | accuracy |
|---:|---:|---:|
| 11582 | 2860 | 0.2469 |

## 分学科

| subject | examples | accuracy |
|---|---:|---:|
| agronomy | 169 | 0.2959 |
| anatomy | 148 | 0.2703 |
| ancient_chinese | 164 | 0.3171 |
| arts | 160 | 0.2500 |
| astronomy | 165 | 0.2545 |
| business_ethics | 209 | 0.2632 |
| chinese_civil_service_exam | 160 | 0.2687 |
| chinese_driving_rule | 131 | 0.2519 |
| chinese_food_culture | 136 | 0.2500 |
| chinese_foreign_policy | 107 | 0.2150 |
| chinese_history | 323 | 0.2415 |
| chinese_literature | 204 | 0.2549 |
| chinese_teacher_qualification | 179 | 0.2514 |
| clinical_knowledge | 237 | 0.2405 |
| college_actuarial_science | 106 | 0.1981 |
| college_education | 107 | 0.2150 |
| college_engineering_hydrology | 106 | 0.2925 |
| college_law | 108 | 0.2963 |
| college_mathematics | 105 | 0.2095 |
| college_medical_statistics | 106 | 0.1981 |
| college_medicine | 273 | 0.2527 |
| computer_science | 204 | 0.1912 |
| computer_security | 171 | 0.2690 |
| conceptual_physics | 147 | 0.2245 |
| construction_project_management | 139 | 0.2086 |
| economics | 159 | 0.2201 |
| education | 163 | 0.2822 |
| electrical_engineering | 172 | 0.2326 |
| elementary_chinese | 252 | 0.2341 |
| elementary_commonsense | 198 | 0.2525 |
| elementary_information_and_technology | 238 | 0.2815 |
| elementary_mathematics | 230 | 0.1870 |
| ethnology | 135 | 0.2370 |
| food_science | 143 | 0.2028 |
| genetics | 176 | 0.2443 |
| global_facts | 149 | 0.2349 |
| high_school_biology | 169 | 0.2840 |
| high_school_chemistry | 132 | 0.2955 |
| high_school_geography | 118 | 0.2203 |
| high_school_mathematics | 164 | 0.2561 |
| high_school_physics | 110 | 0.2818 |
| high_school_politics | 143 | 0.2587 |
| human_sexuality | 126 | 0.2619 |
| international_law | 185 | 0.2432 |
| journalism | 172 | 0.2384 |
| jurisprudence | 411 | 0.2482 |
| legal_and_moral_basis | 214 | 0.2617 |
| logical | 123 | 0.1951 |
| machine_learning | 122 | 0.1885 |
| management | 210 | 0.2476 |
| marketing | 180 | 0.2500 |
| marxist_theory | 189 | 0.2487 |
| modern_chinese | 116 | 0.2241 |
| nutrition | 145 | 0.2276 |
| philosophy | 105 | 0.3524 |
| professional_accounting | 175 | 0.2629 |
| professional_law | 211 | 0.2464 |
| professional_medicine | 376 | 0.2314 |
| professional_psychology | 232 | 0.2241 |
| public_relations | 174 | 0.2644 |
| security_study | 135 | 0.2370 |
| sociology | 226 | 0.2522 |
| sports_science | 165 | 0.2364 |
| traditional_chinese_medicine | 185 | 0.2378 |
| virology | 169 | 0.2604 |
| world_history | 161 | 0.2609 |
| world_religions | 160 | 0.2687 |
