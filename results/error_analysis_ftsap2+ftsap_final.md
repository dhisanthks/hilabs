# Error analysis — `ftsap2+ftsap_final` (5-fold out-of-fold predictions)


## ICD10CM

Exact match **68.3%** on 2306 terms. Candidate recall 95.8%.

Partial credit: 3-char category accuracy **78.5%**, chapter accuracy 87.7%.


**Tolerant metrics (how much of the gap is gold-standard ambiguity)**

| metric | accuracy |
|---|---|
| exact match | 68.3% |
| exact or same-name twin (identical preferred name) | 68.3% |
| exact or near-synonym (name similarity >= 90) | 71.5% |
| correct 3-character category | 78.5% |

**Failure modes (share of errors)**

| mode | errors | share |
|---|---|---|
| retrieval miss (gold not in candidates) | 96 | 13.1% |
| specificity (right 3-char category, wrong code) | 225 | 30.7% |
| right chapter, wrong category | 183 | 25.0% |
| wrong chapter | 228 | 31.1% |

**Accuracy by input slice**

| bucket | n | accuracy |
|---|---|---|
| gold code seen in training = True | 1757 | 75.1% |
| gold code seen in training = False | 549 | 46.3% |
| contains abbreviation = True | 274 | 78.1% |
| contains abbreviation = False | 2032 | 66.9% |
| OCR / typo / markup noise = True | 163 | 73.0% |
| OCR / typo / markup noise = False | 2143 | 67.9% |
| history / status-post = True | 14 | 35.7% |
| history / status-post = False | 2292 | 68.5% |
| negation cue = True | 29 | 58.6% |
| negation cue = False | 2277 | 68.4% |
| laterality = True | 236 | 70.8% |
| laterality = False | 2070 | 68.0% |
| ≥6 tokens = True | 243 | 63.0% |
| ≥6 tokens = False | 2063 | 68.9% |

**Accuracy by subconcept**

| bucket | n | accuracy |
|---|---|---|
| DiagnosisCode | 2305 | 68.2% |

**Calibration: accuracy by confidence bin**

| bucket | n | accuracy |
|---|---|---|
| (-0.001, 0.2] | 47 | 10.6% |
| (0.2, 0.4] | 206 | 21.4% |
| (0.4, 0.6] | 299 | 33.8% |
| (0.6, 0.8] | 344 | 55.5% |
| (0.8, 0.9] | 303 | 71.3% |
| (0.9, 1.0] | 1107 | 91.9% |

**Coverage vs accuracy (abstain on lowest-confidence terms)**

| coverage | accuracy |
|---|---|
| 100.0% | 68.3% |
| 90.0% | 73.9% |
| 80.0% | 79.5% |
| 70.0% | 84.2% |
| 50.0% | 91.3% |
| 30.0% | 95.7% |

**Sample errors**

| entity | gold | predicted | conf |
|---|---|---|---|
| digital ulcers | L98.499 Non-pressure chronic ulcer of skin of other s | L98.8 Other specified disorders of the skin and sub | 0.10 |
| did not sleep well | Z72.820 Sleep deprivation | G47.00 Insomnia, unspecified | 0.73 |
| abnormal rate | R94.31 Abnormal electrocardiogram [ECG] [EKG] | R00.8 Other abnormalities of heart beat | 0.32 |
| acute maxillary sinusitis, recurrence | J01.00 Acute maxillary sinusitis, unspecified | J01.01 Acute recurrent maxillary sinusitis | 0.70 |
| liver effect | T46.6X5A Adverse effect of antihyperlipidemic and anti | K76.9 Liver disease, unspecified | 0.95 |
| hypersensitivity | R20.3 Hyperesthesia | T78.40XA Allergy, unspecified, initial encounter | 0.75 |
| hfref 2/2 icm requiring lvad | I50.20 Unspecified systolic (congestive) heart failu | I50.9 Cardiac, heart or myocardial failure NOS | 0.50 |
| vaginal discharge | N89.8 Other specified noninflammatory disorders of  | R11.10 Vomiting, unspecified | 0.25 |
| multiple round and ovoid relatively hyperdense masses | R93.5 Abnormal findings on diagnostic imaging of ot | R91.8 Lung mass NOS found on diagnostic imaging of  | 0.21 |
| acute intracranial finding | R90.89 Other abnormal findings on diagnostic imaging | R93.0 Abnormal findings on diagnostic imaging of sk | 0.77 |
| additional small bilateral pulmonary nodules | R91.8 Other nonspecific abnormal finding of lung fi | R91.1 Solitary pulmonary nodule, subsegmental branc | 0.79 |
| l5 and s1 nerve root irritation | M54.16 Radiculopathy, lumbar region | G54.8 Other nerve root and plexus disorders | 0.62 |
| allergy/ sinus infection | J32.9 Chronic sinusitis, unspecified | Z88.9 Allergy status to unspecified drugs, medicame | 0.24 |
| covid + seizures | U07.1 COVID-19 | R56.9 Seizure(s) (convulsive) NOS | 0.99 |
| difficulty with sleep | G47.9 Sleep disorder, unspecified | G47.00 Insomnia, unspecified | 0.71 |

## SNOMEDCT_US

Exact match **52.2%** on 7340 terms. Candidate recall 92.8%.

Partial credit: semantic-tag accuracy **73.2%**.


**Tolerant metrics (how much of the gap is gold-standard ambiguity)**

| metric | accuracy |
|---|---|
| exact match | 52.2% |
| exact or same-name twin (identical preferred name) | 52.4% |
| exact or near-synonym (name similarity >= 90) | 60.6% |

**Failure modes (share of errors)**

| mode | errors | share |
|---|---|---|
| retrieval miss (gold not in candidates) | 531 | 15.1% |
| wrong semantic tag (e.g. substance vs procedure) | 1591 | 45.3% |
| same semantic tag, wrong concept | 1387 | 39.5% |

**Accuracy by input slice**

| bucket | n | accuracy |
|---|---|---|
| gold code seen in training = True | 3999 | 62.2% |
| gold code seen in training = False | 3341 | 40.3% |
| contains abbreviation = True | 1004 | 52.4% |
| contains abbreviation = False | 6336 | 52.2% |
| OCR / typo / markup noise = True | 675 | 44.9% |
| OCR / typo / markup noise = False | 6665 | 52.9% |
| history / status-post = True | 42 | 50.0% |
| history / status-post = False | 7298 | 52.2% |
| negation cue = True | 102 | 45.1% |
| negation cue = False | 7238 | 52.3% |
| laterality = True | 377 | 60.5% |
| laterality = False | 6963 | 51.7% |
| ≥6 tokens = True | 582 | 43.8% |
| ≥6 tokens = False | 6758 | 52.9% |

**Accuracy by subconcept**

| bucket | n | accuracy |
|---|---|---|
| Device | 437 | 45.8% |
| DiagnosisCode | 2700 | 62.7% |
| EncounterType | 567 | 53.1% |
| LabService | 858 | 47.7% |
| ProcedureCode | 2777 | 44.3% |

**Calibration: accuracy by confidence bin**

| bucket | n | accuracy |
|---|---|---|
| (-0.001, 0.2] | 1267 | 13.7% |
| (0.2, 0.4] | 1780 | 31.7% |
| (0.4, 0.6] | 1448 | 50.5% |
| (0.6, 0.8] | 1217 | 71.9% |
| (0.8, 0.9] | 750 | 85.7% |
| (0.9, 1.0] | 878 | 96.2% |

**Coverage vs accuracy (abstain on lowest-confidence terms)**

| coverage | accuracy |
|---|---|
| 100.0% | 52.2% |
| 90.0% | 56.9% |
| 80.0% | 61.6% |
| 70.0% | 66.3% |
| 50.0% | 77.2% |
| 30.0% | 87.7% |

**Sample errors**

| entity | gold | predicted | conf |
|---|---|---|---|
| shoulder procedure/surgery | 118703003 Procedure on shoulder | 78559005 Operative procedure on shoulder | 0.45 |
| activityengagement | 405081003 Leisure behavior | 257733005 Activities | 0.21 |
| functional deficits | 60641000087105 Difficulty performing activity of daily livin | 386585008 Functional disorder | 0.35 |
| phillips dreamstation device | 1284824006 Sleep apnea breathing therapy equipment | 49062001 Device | 0.07 |
| physician inpatient | 309343006 Physician | 416800000 Inpatient | 0.26 |
| breast cancer | 429740004 Family history of malignant neoplasm of breas | 254837009 Malignant neoplasm of breast | 0.91 |
| lost 6 pounds | 448765001 Unintentional weight loss | 262285001 Weight loss | 0.29 |
| hpi | 422625006 History of present illness section | 63332003 History AND physical examination | 0.08 |
| head and neck surgery | 394604002 Ear, nose and throat surgery | 763192000 Mouth care after head and neck surgery | 0.25 |
| weight bearing as tolerated to the left lower extremity | 249981005 Weight-bearing | 285692001 Does weight-bear on left leg | 0.16 |
| minutes of exercise per session | 417239007 Number of minutes | 228449008 Frequency of exercise | 0.12 |
| am-pac basic mobility score | 782487009 Assessment score | 718040006 Therapy Outcome Measure activity score | 0.22 |
| nutrition plan | 737427001 Clinical management plan | 1279742004 Development of nutrition care plan | 0.38 |
| 5lnc | 371907003 Oxygen administration by nasal cannula | 568291000005106 Hospital unit | 0.05 |
| hip extension with counter support | 229143007 Hip hitching exercise | 229153008 Hip exercises | 0.22 |