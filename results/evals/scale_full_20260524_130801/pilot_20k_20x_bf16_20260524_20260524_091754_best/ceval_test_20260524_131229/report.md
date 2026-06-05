# 中文多选评测报告

## 配置

- benchmark: `ceval`
- split: `test`
- checkpoint: `results/runs/pilot_20k_20x_bf16_20260524_20260524_091754/checkpoints/best.pt`
- dtype/device: `bfloat16` / `cuda`
- length normalize: `False`

## 总体结果

| examples | correct | accuracy |
|---:|---:|---:|
| 12342 | 2941 | 0.2383 |

## 分学科

| subject | examples | accuracy |
|---|---:|---:|
| accountant | 443 | 0.2280 |
| advanced_mathematics | 173 | 0.2486 |
| art_studies | 298 | 0.3020 |
| basic_medicine | 175 | 0.2400 |
| business_administration | 301 | 0.3256 |
| chinese_language_and_literature | 209 | 0.2392 |
| civil_servant | 429 | 0.2191 |
| clinical_medicine | 200 | 0.2600 |
| college_chemistry | 224 | 0.1920 |
| college_economics | 497 | 0.2173 |
| college_physics | 176 | 0.1761 |
| college_programming | 342 | 0.2368 |
| computer_architecture | 193 | 0.2176 |
| computer_network | 171 | 0.2281 |
| discrete_mathematics | 153 | 0.2484 |
| education_science | 270 | 0.2741 |
| electrical_engineer | 339 | 0.2861 |
| environmental_impact_assessment_engineer | 281 | 0.2633 |
| fire_engineer | 282 | 0.1915 |
| high_school_biology | 175 | 0.2514 |
| high_school_chemistry | 172 | 0.2209 |
| high_school_chinese | 178 | 0.2191 |
| high_school_geography | 178 | 0.2303 |
| high_school_history | 182 | 0.2198 |
| high_school_mathematics | 166 | 0.2349 |
| high_school_physics | 175 | 0.2229 |
| high_school_politics | 176 | 0.2273 |
| ideological_and_moral_cultivation | 172 | 0.2558 |
| law | 221 | 0.1855 |
| legal_professional | 215 | 0.2140 |
| logic | 204 | 0.2549 |
| mao_zedong_thought | 219 | 0.2237 |
| marxism | 179 | 0.2123 |
| metrology_engineer | 219 | 0.2603 |
| middle_school_biology | 192 | 0.2812 |
| middle_school_chemistry | 185 | 0.1784 |
| middle_school_geography | 108 | 0.2037 |
| middle_school_history | 207 | 0.2415 |
| middle_school_mathematics | 177 | 0.1864 |
| middle_school_physics | 178 | 0.2753 |
| middle_school_politics | 193 | 0.2435 |
| modern_chinese_history | 212 | 0.2358 |
| operating_system | 179 | 0.2849 |
| physician | 443 | 0.2280 |
| plant_protection | 199 | 0.3065 |
| probability_and_statistics | 166 | 0.2410 |
| professional_tour_guide | 266 | 0.2481 |
| sports_science | 180 | 0.3111 |
| tax_accountant | 443 | 0.1964 |
| teacher_qualification | 399 | 0.2331 |
| urban_and_rural_planner | 418 | 0.2512 |
| veterinary_medicine | 210 | 0.2143 |
