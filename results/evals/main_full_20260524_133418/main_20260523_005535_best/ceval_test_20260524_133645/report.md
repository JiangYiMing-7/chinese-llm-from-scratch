# 中文多选评测报告

## 配置

- benchmark: `ceval`
- split: `test`
- checkpoint: `results/runs/main_20260523_005535/checkpoints/best.pt`
- dtype/device: `bfloat16` / `cuda`
- length normalize: `False`

## 总体结果

| examples | correct | accuracy |
|---:|---:|---:|
| 12342 | 3083 | 0.2498 |

## 分学科

| subject | examples | accuracy |
|---|---:|---:|
| accountant | 443 | 0.2438 |
| advanced_mathematics | 173 | 0.2139 |
| art_studies | 298 | 0.2114 |
| basic_medicine | 175 | 0.3257 |
| business_administration | 301 | 0.1927 |
| chinese_language_and_literature | 209 | 0.2727 |
| civil_servant | 429 | 0.2727 |
| clinical_medicine | 200 | 0.2800 |
| college_chemistry | 224 | 0.2366 |
| college_economics | 497 | 0.2394 |
| college_physics | 176 | 0.2557 |
| college_programming | 342 | 0.2544 |
| computer_architecture | 193 | 0.2332 |
| computer_network | 171 | 0.2398 |
| discrete_mathematics | 153 | 0.2549 |
| education_science | 270 | 0.2148 |
| electrical_engineer | 339 | 0.2124 |
| environmental_impact_assessment_engineer | 281 | 0.2598 |
| fire_engineer | 282 | 0.2163 |
| high_school_biology | 175 | 0.2743 |
| high_school_chemistry | 172 | 0.2791 |
| high_school_chinese | 178 | 0.2416 |
| high_school_geography | 178 | 0.2247 |
| high_school_history | 182 | 0.2253 |
| high_school_mathematics | 166 | 0.2289 |
| high_school_physics | 175 | 0.2743 |
| high_school_politics | 176 | 0.3011 |
| ideological_and_moral_cultivation | 172 | 0.2558 |
| law | 221 | 0.2172 |
| legal_professional | 215 | 0.2512 |
| logic | 204 | 0.2157 |
| mao_zedong_thought | 219 | 0.2055 |
| marxism | 179 | 0.3296 |
| metrology_engineer | 219 | 0.2420 |
| middle_school_biology | 192 | 0.2812 |
| middle_school_chemistry | 185 | 0.2919 |
| middle_school_geography | 108 | 0.2963 |
| middle_school_history | 207 | 0.2464 |
| middle_school_mathematics | 177 | 0.2881 |
| middle_school_physics | 178 | 0.2640 |
| middle_school_politics | 193 | 0.2539 |
| modern_chinese_history | 212 | 0.2689 |
| operating_system | 179 | 0.2458 |
| physician | 443 | 0.2551 |
| plant_protection | 199 | 0.2211 |
| probability_and_statistics | 166 | 0.2349 |
| professional_tour_guide | 266 | 0.2481 |
| sports_science | 180 | 0.2167 |
| tax_accountant | 443 | 0.2596 |
| teacher_qualification | 399 | 0.2832 |
| urban_and_rural_planner | 418 | 0.2512 |
| veterinary_medicine | 210 | 0.2762 |
