# 中文多选评测报告

## 配置

- benchmark: `ceval`
- split: `test`
- checkpoint: `results/runs/tiny_20k_20x_bf16_20260524_20260524_090918/checkpoints/best.pt`
- dtype/device: `bfloat16` / `cuda`
- length normalize: `False`

## 总体结果

| examples | correct | accuracy |
|---:|---:|---:|
| 12342 | 2847 | 0.2307 |

## 分学科

| subject | examples | accuracy |
|---|---:|---:|
| accountant | 443 | 0.2212 |
| advanced_mathematics | 173 | 0.2543 |
| art_studies | 298 | 0.3188 |
| basic_medicine | 175 | 0.1886 |
| business_administration | 301 | 0.3056 |
| chinese_language_and_literature | 209 | 0.2440 |
| civil_servant | 429 | 0.2145 |
| clinical_medicine | 200 | 0.2350 |
| college_chemistry | 224 | 0.2009 |
| college_economics | 497 | 0.2032 |
| college_physics | 176 | 0.1705 |
| college_programming | 342 | 0.2339 |
| computer_architecture | 193 | 0.2124 |
| computer_network | 171 | 0.2222 |
| discrete_mathematics | 153 | 0.2288 |
| education_science | 270 | 0.2741 |
| electrical_engineer | 339 | 0.2950 |
| environmental_impact_assessment_engineer | 281 | 0.2562 |
| fire_engineer | 282 | 0.2305 |
| high_school_biology | 175 | 0.2114 |
| high_school_chemistry | 172 | 0.1860 |
| high_school_chinese | 178 | 0.2528 |
| high_school_geography | 178 | 0.2416 |
| high_school_history | 182 | 0.1868 |
| high_school_mathematics | 166 | 0.2048 |
| high_school_physics | 175 | 0.1943 |
| high_school_politics | 176 | 0.1932 |
| ideological_and_moral_cultivation | 172 | 0.3023 |
| law | 221 | 0.1810 |
| legal_professional | 215 | 0.1953 |
| logic | 204 | 0.2745 |
| mao_zedong_thought | 219 | 0.2648 |
| marxism | 179 | 0.1564 |
| metrology_engineer | 219 | 0.2740 |
| middle_school_biology | 192 | 0.1823 |
| middle_school_chemistry | 185 | 0.2054 |
| middle_school_geography | 108 | 0.2037 |
| middle_school_history | 207 | 0.1739 |
| middle_school_mathematics | 177 | 0.1864 |
| middle_school_physics | 178 | 0.2697 |
| middle_school_politics | 193 | 0.2435 |
| modern_chinese_history | 212 | 0.2547 |
| operating_system | 179 | 0.2849 |
| physician | 443 | 0.2235 |
| plant_protection | 199 | 0.2613 |
| probability_and_statistics | 166 | 0.2349 |
| professional_tour_guide | 266 | 0.2368 |
| sports_science | 180 | 0.2889 |
| tax_accountant | 443 | 0.1941 |
| teacher_qualification | 399 | 0.2130 |
| urban_and_rural_planner | 418 | 0.2392 |
| veterinary_medicine | 210 | 0.2143 |
