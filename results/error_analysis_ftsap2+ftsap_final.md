# Error analysis — `ftsap2+ftsap_final` (5-fold out-of-fold predictions)


## ICD10CM

Exact match **68.4%** on 2306 terms. Candidate recall 95.8%.

Partial credit: 3-char category accuracy **79.0%**, chapter accuracy 88.0%.


**Tolerant metrics (how much of the gap is gold-standard ambiguity)**

| metric | accuracy |
|---|---|
| exact match | 68.4% |
| exact or same-name twin (identical preferred name) | 68.4% |
| exact or near-synonym (name similarity >= 90) | 71.8% |
| correct 3-character category | 79.0% |

**Failure modes (share of errors)**

| mode | errors | share |
|---|---|---|
| retrieval miss (gold not in candidates) | 96 | 13.2% |
| specificity (right 3-char category, wrong code) | 233 | 32.0% |
| right chapter, wrong category | 178 | 24.4% |
| wrong chapter | 222 | 30.5% |

**Accuracy by input slice**

| bucket | n | accuracy |
|---|---|---|
| gold code seen in training = True | 1757 | 75.1% |
| gold code seen in training = False | 549 | 46.8% |
| contains abbreviation = True | 274 | 78.1% |
| contains abbreviation = False | 2032 | 67.1% |
| OCR / typo / markup noise = True | 163 | 71.2% |
| OCR / typo / markup noise = False | 2143 | 68.2% |
| history / status-post = True | 14 | 28.6% |
| history / status-post = False | 2292 | 68.6% |
| negation cue = True | 29 | 58.6% |
| negation cue = False | 2277 | 68.5% |
| laterality = True | 236 | 72.5% |
| laterality = False | 2070 | 67.9% |
| ≥6 tokens = True | 243 | 65.0% |
| ≥6 tokens = False | 2063 | 68.8% |

**Accuracy by subconcept**

| bucket | n | accuracy |
|---|---|---|
| DiagnosisCode | 2305 | 68.4% |

**Calibration: accuracy by confidence bin**

| bucket | n | accuracy |
|---|---|---|
| (-0.001, 0.2] | 48 | 12.5% |
| (0.2, 0.4] | 207 | 22.7% |
| (0.4, 0.6] | 276 | 33.3% |
| (0.6, 0.8] | 352 | 52.3% |
| (0.8, 0.9] | 305 | 74.1% |
| (0.9, 1.0] | 1118 | 91.4% |

**Coverage vs accuracy (abstain on lowest-confidence terms)**

| coverage | accuracy |
|---|---|
| 100.0% | 68.4% |
| 90.0% | 73.7% |
| 80.0% | 79.3% |
| 70.0% | 84.0% |
| 50.0% | 90.8% |
| 30.0% | 95.5% |

**Sample errors**

| entity | gold | predicted | conf |
|---|---|---|---|
| chest tightness | R07.89 Other chest pain | R07.9 Chest pain, unspecified | 0.42 |
| glucotoxic state | E11.69 Type 2 diabetes mellitus with other specified | R73.9 Hyperglycemia, unspecified | 0.35 |
| large mixed sclerotic and lucent lesion involving the superi | M89.9 Disorder of bone, unspecified | M87.051 Idiopathic aseptic necrosis of right femur | 0.42 |
| muscle tension | M62.838 Other muscle spasm | R29.898 Other symptoms and signs involving the muscul | 0.25 |
| right ventricle is not filling his left ventricle sufficient | I50.810 Right heart failure, unspecified | I51.89 Other ill-defined heart diseases | 0.35 |
| pharmacological dvt | Z86.718 Personal history of other venous thrombosis a | I82.409 Acute embolism and thrombosis of unspecified  | 0.62 |
| nasacort allergy 24hr | Z88.8 Allergy status to other drugs, medicaments an | J30.9 Allergic rhinitis, unspecified | 0.35 |
| advanced heart failure | I50.84 End stage heart failure | I50.9 Cardiac, heart or myocardial failure NOS | 0.95 |
| acute otitis media, unspecified otitis media type | H66.90 Otitis media, unspecified, unspecified ear | H66.9 Otitis media, unspecified | 0.47 |
| acquired hypothyroidism | E03.9 Hypothyroidism, unspecified | E01.8 Other iodine-deficiency related thyroid disor | 0.67 |
| torn meniscus vitamin d deficiency | M23.209 Derangement of unspecified meniscus due to ol | E55.9 Vitamin D deficiency, unspecified | 0.61 |
| goc | Z71.89 Other specified counseling | R40.20 Unconsciousness NOS | 0.39 |
| dm hx gastric ulcer | Z87.19 Personal history of other diseases of the dig | K25.9 Gastric ulcer, unspecified as acute or chroni | 0.59 |
| watering eyes | H04.209 Unspecified epiphora, unspecified side | H57.89 Other specified disorders of eye and adnexa | 0.36 |
| idiopathic scoliosis of thoracic spine | M41.24 Other idiopathic scoliosis, thoracic region | M41.04 Infantile idiopathic scoliosis, thoracic regi | 0.36 |

## SNOMEDCT_US

Exact match **53.1%** on 7340 terms. Candidate recall 92.8%.

Partial credit: semantic-tag accuracy **74.2%**.


**Tolerant metrics (how much of the gap is gold-standard ambiguity)**

| metric | accuracy |
|---|---|
| exact match | 53.1% |
| exact or same-name twin (identical preferred name) | 53.3% |
| exact or near-synonym (name similarity >= 90) | 61.2% |

**Failure modes (share of errors)**

| mode | errors | share |
|---|---|---|
| retrieval miss (gold not in candidates) | 531 | 15.4% |
| wrong semantic tag (e.g. substance vs procedure) | 1518 | 44.1% |
| same semantic tag, wrong concept | 1394 | 40.5% |

**Accuracy by input slice**

| bucket | n | accuracy |
|---|---|---|
| gold code seen in training = True | 3999 | 63.2% |
| gold code seen in training = False | 3341 | 41.0% |
| contains abbreviation = True | 1004 | 53.1% |
| contains abbreviation = False | 6336 | 53.1% |
| OCR / typo / markup noise = True | 675 | 46.2% |
| OCR / typo / markup noise = False | 6665 | 53.8% |
| history / status-post = True | 42 | 52.4% |
| history / status-post = False | 7298 | 53.1% |
| negation cue = True | 102 | 44.1% |
| negation cue = False | 7238 | 53.2% |
| laterality = True | 377 | 62.3% |
| laterality = False | 6963 | 52.6% |
| ≥6 tokens = True | 582 | 46.6% |
| ≥6 tokens = False | 6758 | 53.7% |

**Accuracy by subconcept**

| bucket | n | accuracy |
|---|---|---|
| Device | 437 | 46.5% |
| DiagnosisCode | 2700 | 63.2% |
| EncounterType | 567 | 53.1% |
| LabService | 858 | 51.0% |
| ProcedureCode | 2777 | 44.9% |

**Calibration: accuracy by confidence bin**

| bucket | n | accuracy |
|---|---|---|
| (-0.001, 0.2] | 1208 | 13.3% |
| (0.2, 0.4] | 1754 | 31.5% |
| (0.4, 0.6] | 1399 | 52.1% |
| (0.6, 0.8] | 1205 | 68.5% |
| (0.8, 0.9] | 800 | 86.6% |
| (0.9, 1.0] | 974 | 96.1% |

**Coverage vs accuracy (abstain on lowest-confidence terms)**

| coverage | accuracy |
|---|---|
| 100.0% | 53.1% |
| 90.0% | 57.8% |
| 80.0% | 62.7% |
| 70.0% | 67.4% |
| 50.0% | 76.9% |
| 30.0% | 88.8% |

**Sample errors**

| entity | gold | predicted | conf |
|---|---|---|---|
| swollen | 249355003 Nasal turbinate finding | 65124004 Swelling | 0.41 |
| non healing wounds | 789507005 Delayed healing of wound | 781187003 Non-healing surgical wound | 0.42 |
| changes in urine frequency or amount | 170877009 Urinary symptom change | 162116003 Increased frequency of micturition | 0.29 |
| negative aspiration | 442718000 Aspiration test negative for blood during pro | 14766002 Aspiration | 0.40 |
| postural change | 258144009 Position change | 103017008 Postural dizziness | 0.25 |
| significant flattening of the femoral head | 20601007 Flattening deformity | 1142190006 Congenital dysplasia of head of femur | 0.34 |
| drug screen, urine, comprehensive | 40823001 Drug of abuse, quantitative screen, includes  | 314076009 Toxicology screening test | 0.17 |
| ssa avid metastatic deposits | 94381002 Metastatic malignant neoplasm to liver | 128462008 Metastatic malignant neoplasm | 0.40 |
| peripheral artery angioplasty | 8990000 Fluoroscopy guided percutaneous transluminal  | 709077000 Fluoroscopy guided percutaneous transluminal  | 0.25 |
| suture nerve req shortening bone extremity left | 3666002 Peripheral neurorrhaphy | 292041000119104 Entrapment neuropathy of left common peroneal | 0.30 |
| oxygen | 57485005 O2 therapy | 24099007 Oxygen | 0.54 |
| s4 | 4592006 Fourth sound gallop | 277455002 S3 - Third heart sound | 0.13 |
| other ot goal grid | 225294001 Identifying goals | 410518001 Goal | 0.29 |
| difficulty opening fist after strong grip | 422561007 Myotonic warm-up phenomenon - hand grip | 284310008 Difficulty releasing grip | 0.18 |
| home with hh | 10161009 Patient discharge, to home, with assistance | 264362003 Private dwelling | 0.12 |