# Error analysis — `ftsap` (5-fold out-of-fold predictions)


## ICD10CM

Exact match **66.7%** on 2306 terms. Candidate recall 95.8%.

Partial credit: 3-char category accuracy **77.7%**, chapter accuracy 87.0%.


**Failure modes (share of errors)**

| mode | errors | share |
|---|---|---|
| retrieval miss (gold not in candidates) | 98 | 12.8% |
| specificity (right 3-char category, wrong code) | 242 | 31.5% |
| right chapter, wrong category | 192 | 25.0% |
| wrong chapter | 236 | 30.7% |

**Accuracy by input slice**

| bucket | n | accuracy |
|---|---|---|
| gold code seen in training = True | 1757 | 74.5% |
| gold code seen in training = False | 549 | 41.7% |
| contains abbreviation = True | 274 | 77.0% |
| contains abbreviation = False | 2032 | 65.3% |
| OCR / typo / markup noise = True | 163 | 71.8% |
| OCR / typo / markup noise = False | 2143 | 66.3% |
| history / status-post = True | 14 | 35.7% |
| history / status-post = False | 2292 | 66.9% |
| negation cue = True | 29 | 65.5% |
| negation cue = False | 2277 | 66.7% |
| laterality = True | 236 | 68.2% |
| laterality = False | 2070 | 66.5% |
| ≥6 tokens = True | 243 | 64.6% |
| ≥6 tokens = False | 2063 | 66.9% |

**Accuracy by subconcept**

| bucket | n | accuracy |
|---|---|---|
| DiagnosisCode | 2305 | 66.7% |

**Calibration: accuracy by confidence bin**

| bucket | n | accuracy |
|---|---|---|
| (-0.001, 0.2] | 89 | 13.5% |
| (0.2, 0.4] | 304 | 24.0% |
| (0.4, 0.6] | 347 | 43.2% |
| (0.6, 0.8] | 388 | 62.1% |
| (0.8, 0.9] | 280 | 81.4% |
| (0.9, 1.0] | 898 | 92.9% |

**Coverage vs accuracy (abstain on lowest-confidence terms)**

| coverage | accuracy |
|---|---|
| 100.0% | 66.7% |
| 90.0% | 71.9% |
| 80.0% | 77.5% |
| 70.0% | 82.3% |
| 50.0% | 90.6% |
| 30.0% | 94.4% |

**Sample errors**

| entity | gold | predicted | conf |
|---|---|---|---|
| sinus problems | J34.9 Disorder of nose and nasal sinuses, unspecifi | J34.89 Other specified disorders of nose and nasal s | 0.22 |
| cervical disc disorder with radiculopathy of mid-cervical | M50.12 Other cervical disc displacement, mid-cervica | M50.120 Mid-cervical disc disorder, unspecified level | 0.48 |
| cannot get up from chair without using ues reflexes | R29.898 Other symptoms and signs involving the muscul | R26.89 Other abnormalities of gait and mobility | 0.67 |
| abnormal movement | R29.898 Other symptoms and signs involving the muscul | R25.9 Unspecified abnormal involuntary movements | 0.64 |
| orthopedic restrictions | Z96.649 Presence of unspecified artificial hip joint | R29.898 Other symptoms and signs involving the muscul | 0.46 |
| brownish slightly scaly rash on the right flank | R21 Rash and other nonspecific skin eruption | L22 Psoriasiform diaper rash | 0.33 |
| hypnagogic hallucinations | G47.419 Narcolepsy without cataplexy | R44.3 Hallucinations, unspecified | 0.64 |
| ptosis of the nasal tip | M95.0 Acquired deformity of nose | J34.89 Other specified disorders of nose and nasal s | 0.43 |
| left lower extremity weakness | R29.898 Other symptoms and signs involving the muscul | R53.1 Asthenia NOS | 0.85 |
| disability | R41.9 Unspecified symptoms and signs involving cogn | Z73.6 Limitation of activities due to disability | 0.34 |
| acute maxillary sinusitis, recurrence | J01.00 Acute maxillary sinusitis, unspecified | J01.01 Acute recurrent maxillary sinusitis | 0.82 |
| perianal symptoms | R19.8 Other specified symptoms and signs involving  | L29.3 Anogenital pruritus, unspecified | 0.25 |
| functional deficits | R29.818 Other symptoms and signs involving the nervou | R68.89 Other general symptoms and signs | 0.40 |
| pulmonary vascular disease | I27.9 Pulmonary heart disease, unspecified | I28.9 Disease of pulmonary vessels, unspecified | 0.91 |
| urinary problems | R39.9 Unspecified symptoms and signs involving the  | R32 Unspecified urinary incontinence | 0.34 |

## SNOMEDCT_US

Exact match **51.1%** on 7340 terms. Candidate recall 92.5%.

Partial credit: semantic-tag accuracy **72.4%**.


**Failure modes (share of errors)**

| mode | errors | share |
|---|---|---|
| retrieval miss (gold not in candidates) | 552 | 15.4% |
| wrong semantic tag (e.g. substance vs procedure) | 1630 | 45.4% |
| same semantic tag, wrong concept | 1408 | 39.2% |

**Accuracy by input slice**

| bucket | n | accuracy |
|---|---|---|
| gold code seen in training = True | 3999 | 61.4% |
| gold code seen in training = False | 3341 | 38.8% |
| contains abbreviation = True | 1004 | 51.7% |
| contains abbreviation = False | 6336 | 51.0% |
| OCR / typo / markup noise = True | 675 | 43.6% |
| OCR / typo / markup noise = False | 6665 | 51.9% |
| history / status-post = True | 42 | 52.4% |
| history / status-post = False | 7298 | 51.1% |
| negation cue = True | 102 | 41.2% |
| negation cue = False | 7238 | 51.2% |
| laterality = True | 377 | 58.9% |
| laterality = False | 6963 | 50.7% |
| ≥6 tokens = True | 582 | 43.6% |
| ≥6 tokens = False | 6758 | 51.7% |

**Accuracy by subconcept**

| bucket | n | accuracy |
|---|---|---|
| Device | 437 | 43.5% |
| DiagnosisCode | 2700 | 62.0% |
| EncounterType | 567 | 51.3% |
| LabService | 858 | 45.6% |
| ProcedureCode | 2777 | 43.4% |

**Calibration: accuracy by confidence bin**

| bucket | n | accuracy |
|---|---|---|
| (-0.001, 0.2] | 1391 | 14.5% |
| (0.2, 0.4] | 1774 | 32.2% |
| (0.4, 0.6] | 1331 | 50.4% |
| (0.6, 0.8] | 1160 | 67.9% |
| (0.8, 0.9] | 846 | 85.7% |
| (0.9, 1.0] | 838 | 94.7% |

**Coverage vs accuracy (abstain on lowest-confidence terms)**

| coverage | accuracy |
|---|---|
| 100.0% | 51.1% |
| 90.0% | 55.6% |
| 80.0% | 60.1% |
| 70.0% | 64.8% |
| 50.0% | 74.7% |
| 30.0% | 85.9% |

**Sample errors**

| entity | gold | predicted | conf |
|---|---|---|---|
| strips) strip | 337388004 Blood glucose testing strips | 708548003 Test strip | 0.49 |
| new patient patient summary | 185387006 New patient consultation | 5491000179105 Medical record summary | 0.41 |
| femoral head | 304121006 Femoral head prosthesis | 2812003 Head of femur | 0.90 |
| psychosocial | 363910003 Characteristic of psychosocial functioning | 408919008 Psychosocial analysis | 0.28 |
| non-anion gap metabolic acidosis | 111377009 Metabolic acidosis, normal anion gap (NAG) | 59455009 Metabolic acidosis | 0.67 |
| chronic depression/ anxiety | 192080009 Chronic depression | 191708009 Chronic anxiety | 0.32 |
| ed obs admission | 448951000124107 Admission to observation unit | 50849002 Emergency room admission | 0.70 |
| intentional swelling in your legs,feet, abdomen | 267038008 Edema | 60728008 Swollen abdomen | 0.35 |
| going up/down steps | 282193002 Does walk up stairs | 282517006 Ability to run down steps | 0.12 |
| gallops | 2170000 Gallop rhythm | 49864004 Protodiastolic gallop with abnormally audible | 0.36 |
| lt shoulder op report | 78559005 Operative procedure on shoulder | 217225009 Repair of shoulder | 0.05 |
| vit d | 34713006 Vitamin D deficiency | 270990000 Serum vitamin D measurement | 0.49 |
| extreme fatigue | 84946008 Extreme exhaustion | 84229001 Fatigue | 0.89 |
| metal keratome | 708114009 Surgical blade | 470033000 Keratome, battery-powered | 0.20 |
| continue as prescribed | 306805000 Recommendation to continue with treatment | 306806004 Recommendation to continue with drug treatmen | 0.37 |