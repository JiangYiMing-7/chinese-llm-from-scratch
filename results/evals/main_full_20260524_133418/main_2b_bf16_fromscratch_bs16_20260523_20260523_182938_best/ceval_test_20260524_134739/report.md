# 中文多选评测报告

## 配置

- benchmark: `ceval`
- split: `test`
- checkpoint: `results/runs/main_2b_bf16_fromscratch_bs16_20260523_20260523_182938/checkpoints/best.pt`
- dtype/device: `bfloat16` / `cuda`
- length normalize: `False`

## 总体结果

| examples | correct | accuracy |
|---:|---:|---:|
| 12342 | 3129 | 0.2535 |

## 分学科

| subject | examples | accuracy |
|---|---:|---:|
| accountant | 443 | 0.2348 |
| advanced_mathematics | 173 | 0.2775 |
| art_studies | 298 | 0.2584 |
| basic_medicine | 175 | 0.2571 |
| business_administration | 301 | 0.2359 |
| chinese_language_and_literature | 209 | 0.2584 |
| civil_servant | 429 | 0.2657 |
| clinical_medicine | 200 | 0.2850 |
| college_chemistry | 224 | 0.2679 |
| college_economics | 497 | 0.2596 |
| college_physics | 176 | 0.4034 |
| college_programming | 342 | 0.2632 |
| computer_architecture | 193 | 0.2332 |
| computer_network | 171 | 0.2456 |
| discrete_mathematics | 153 | 0.2288 |
| education_science | 270 | 0.2222 |
| electrical_engineer | 339 | 0.2242 |
| environmental_impact_assessment_engineer | 281 | 0.2349 |
| fire_engineer | 282 | 0.2482 |
| high_school_biology | 175 | 0.2514 |
| high_school_chemistry | 172 | 0.2326 |
| high_school_chinese | 178 | 0.2865 |
| high_school_geography | 178 | 0.2022 |
| high_school_history | 182 | 0.2967 |
| high_school_mathematics | 166 | 0.2771 |
| high_school_physics | 175 | 0.2629 |
| high_school_politics | 176 | 0.3011 |
| ideological_and_moral_cultivation | 172 | 0.2093 |
| law | 221 | 0.2353 |
| legal_professional | 215 | 0.2279 |
| logic | 204 | 0.2108 |
| mao_zedong_thought | 219 | 0.2329 |
| marxism | 179 | 0.3128 |
| metrology_engineer | 219 | 0.2283 |
| middle_school_biology | 192 | 0.3490 |
| middle_school_chemistry | 185 | 0.2757 |
| middle_school_geography | 108 | 0.2315 |
| middle_school_history | 207 | 0.2947 |
| middle_school_mathematics | 177 | 0.2881 |
| middle_school_physics | 178 | 0.2640 |
| middle_school_politics | 193 | 0.2642 |
| modern_chinese_history | 212 | 0.2642 |
| operating_system | 179 | 0.2626 |
| physician | 443 | 0.2415 |
| plant_protection | 199 | 0.2312 |
| probability_and_statistics | 166 | 0.2771 |
| professional_tour_guide | 266 | 0.2368 |
| sports_science | 180 | 0.2167 |
| tax_accountant | 443 | 0.2325 |
| teacher_qualification | 399 | 0.2256 |
| urban_and_rural_planner | 418 | 0.2536 |
| veterinary_medicine | 210 | 0.2476 |
