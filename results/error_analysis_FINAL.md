# Error analysis — `FINAL` (5-fold out-of-fold predictions)


## ICD10CM

Exact match **71.1%** on 2306 terms. Candidate recall 95.8%.

Partial credit: 3-char category accuracy **81.6%**, chapter accuracy 89.7%.


**Tolerant metrics (how much of the gap is gold-standard ambiguity)**

| metric | accuracy |
|---|---|
| exact match | 71.1% |
| exact or same-name twin (identical preferred name) | 71.1% |
| exact or near-synonym (name similarity >= 90) | 74.4% |
| correct 3-character category | 81.6% |

**Failure modes (share of errors)**

| mode | errors | share |
|---|---|---|
| retrieval miss (gold not in candidates) | 96 | 14.4% |
| specificity (right 3-char category, wrong code) | 227 | 34.0% |
| right chapter, wrong category | 158 | 23.7% |
| wrong chapter | 186 | 27.9% |

**Accuracy by input slice**

| bucket | n | accuracy |
|---|---|---|
| gold code seen in training = True | 1757 | 77.6% |
| gold code seen in training = False | 549 | 50.1% |
| contains abbreviation = True | 274 | 79.2% |
| contains abbreviation = False | 2032 | 70.0% |
| OCR / typo / markup noise = True | 163 | 77.9% |
| OCR / typo / markup noise = False | 2143 | 70.6% |
| history / status-post = True | 14 | 42.9% |
| history / status-post = False | 2292 | 71.2% |
| negation cue = True | 29 | 58.6% |
| negation cue = False | 2277 | 71.2% |
| laterality = True | 236 | 73.3% |
| laterality = False | 2070 | 70.8% |
| ≥6 tokens = True | 243 | 68.3% |
| ≥6 tokens = False | 2063 | 71.4% |

**Accuracy by subconcept**

| bucket | n | accuracy |
|---|---|---|
| DiagnosisCode | 2305 | 71.1% |

**Calibration: accuracy by confidence bin**

| bucket | n | accuracy |
|---|---|---|
| (-0.001, 0.2] | 22 | 13.6% |
| (0.2, 0.4] | 142 | 20.4% |
| (0.4, 0.6] | 271 | 36.5% |
| (0.6, 0.8] | 355 | 54.4% |
| (0.8, 0.9] | 252 | 67.9% |
| (0.9, 1.0] | 1264 | 90.5% |

**Coverage vs accuracy (abstain on lowest-confidence terms)**

| coverage | accuracy |
|---|---|
| 100.0% | 71.1% |
| 90.0% | 76.3% |
| 80.0% | 81.1% |
| 70.0% | 85.4% |
| 50.0% | 92.3% |
| 30.0% | 96.4% |

**Sample errors**

| entity | gold | predicted | conf |
|---|---|---|---|
| small right paracentral protrusion contacting | M51.27 Other intervertebral disc displacement, lumbo | M51.26 Other intervertebral disc displacement, lumba | 0.97 |
| irregular heartbeats | R00.8 Other and unspecified abnormalities of heart  | I49.9 Cardiac arrhythmia, unspecified | 0.60 |
| acute arterial insufficiency | I77.1 Stricture of artery | I67.81 Acute cerebrovascular insufficiency unspecifi | 0.55 |
| rtka | Z96.651 Presence of right artificial knee joint | I70.791 Other atherosclerosis of other type of bypass | 0.14 |
| blood clots | I82.90 Acute embolism and thrombosis of unspecified  | I74.9 Embolism and thrombosis of unspecified artery | 0.54 |
| difficult time at night trying to sleep and find a comfortab | G47.9 Sleep disorder, unspecified | G47.00 Insomnia, unspecified | 0.84 |
| primary osteoarthritis involving multiple joints | M15.0 Primary generalized (osteo)arthritis | M15.9 Polyosteoarthritis, unspecified | 0.53 |
| posterior dislocation | S43.026A Posterior dislocation of unspecified humerus, | S73.016A Posterior dislocation of unspecified hip, ini | 0.91 |
| tissue injury | T14.90 Injury, unspecified, initial encounter | T14.8XXA Other injury of unspecified body region, init | 0.47 |
| moderate right facet hypertrophy c2-c3 | M47.812 Spondylosis without myelopathy or radiculopat | M89.30 Hypertrophy of bone, unspecified site | 0.82 |
| fell multiple times | W19.XXXA Unspecified fall, initial encounter | R29.6 Tendency to fall | 0.87 |
| interstitial edema | J81.1 Pulmonary edema, unspecified | R60.9 Fluid retention NOS | 0.92 |
| major airway displacement | J98.4 Other disorders of lung | R09.89 Other specified symptoms and signs involving  | 0.71 |
| scarring at the right lung base | J98.4 Other disorders of lung | R91.8 Lung mass NOS found on diagnostic imaging of  | 0.76 |
| scleral injection bilaterally | H15.89 Other disorders of sclera | H15.003 Unspecified scleritis, bilateral | 0.53 |

## SNOMEDCT_US

Exact match **58.2%** on 7340 terms. Candidate recall 92.8%.

Partial credit: semantic-tag accuracy **76.9%**.


**Tolerant metrics (how much of the gap is gold-standard ambiguity)**

| metric | accuracy |
|---|---|
| exact match | 58.2% |
| exact or same-name twin (identical preferred name) | 58.5% |
| exact or near-synonym (name similarity >= 90) | 65.6% |

**Failure modes (share of errors)**

| mode | errors | share |
|---|---|---|
| retrieval miss (gold not in candidates) | 531 | 17.3% |
| wrong semantic tag (e.g. substance vs procedure) | 1316 | 42.9% |
| same semantic tag, wrong concept | 1218 | 39.7% |

**Accuracy by input slice**

| bucket | n | accuracy |
|---|---|---|
| gold code seen in training = True | 3999 | 67.7% |
| gold code seen in training = False | 3341 | 47.0% |
| contains abbreviation = True | 1004 | 57.4% |
| contains abbreviation = False | 6336 | 58.4% |
| OCR / typo / markup noise = True | 675 | 53.5% |
| OCR / typo / markup noise = False | 6665 | 58.7% |
| history / status-post = True | 42 | 50.0% |
| history / status-post = False | 7298 | 58.3% |
| negation cue = True | 102 | 54.9% |
| negation cue = False | 7238 | 58.3% |
| laterality = True | 377 | 64.7% |
| laterality = False | 6963 | 57.9% |
| ≥6 tokens = True | 582 | 52.9% |
| ≥6 tokens = False | 6758 | 58.7% |

**Accuracy by subconcept**

| bucket | n | accuracy |
|---|---|---|
| Device | 437 | 53.5% |
| DiagnosisCode | 2700 | 67.4% |
| EncounterType | 567 | 59.6% |
| LabService | 858 | 55.9% |
| ProcedureCode | 2777 | 50.5% |

**Calibration: accuracy by confidence bin**

| bucket | n | accuracy |
|---|---|---|
| (-0.001, 0.2] | 849 | 15.7% |
| (0.2, 0.4] | 1677 | 34.2% |
| (0.4, 0.6] | 1407 | 51.0% |
| (0.6, 0.8] | 1248 | 71.3% |
| (0.8, 0.9] | 861 | 83.2% |
| (0.9, 1.0] | 1298 | 96.0% |

**Coverage vs accuracy (abstain on lowest-confidence terms)**

| coverage | accuracy |
|---|---|
| 100.0% | 58.2% |
| 90.0% | 63.1% |
| 80.0% | 67.6% |
| 70.0% | 72.3% |
| 50.0% | 82.0% |
| 30.0% | 90.8% |

**Sample errors**

| entity | gold | predicted | conf |
|---|---|---|---|
| care mgt initial assessment | 371754007 Discharge planning | 135844000 Assessment of needs of caregiver completed | 0.09 |
| polyethylene | 8030004 Polyethylene glycol | 71463006 Polyethylene | 0.92 |
| medication adjustment | 309633003 Prescription dose change | 182838006 Change of medication | 0.25 |
| flex aarom | 85680003 Range of motion exercise | 298775004 Active range of shoulder flexion | 0.33 |
| increased pvc burden | 251176006 Multiple premature ventricular complexes | 251175005 Ventricular premature contraction | 0.43 |
| electrophoresis wo | 703450007 Electrophoresis | 814007 Electrophoresis | 0.34 |
| ssm st charles clinic med | 39350007 Private physicians' group office | 257585005 Clinic | 0.10 |
| hcc | 1251554004 Medical coding | 109841003 Liver carcinoma | 0.57 |
| nasacort allergy 24hr | 766426007 Triamcinolone acetonide 55 microgram/actuatio | 61582004 Allergic rhinitis | 0.15 |
| quit over 40 years ago | 2943051000005103 History of nicotine use | 8517006 Former smoker | 0.55 |
| perform transfers | 129347002 Surgical transfer - action | 302271003 Does perform wheelchair transfers | 0.24 |
| significant discomfort at the patella | 247347003 Discomfort | 239733006 Anterior knee pain | 0.27 |
| pacing thru activity steps | 363659002 Pacing - action | 74691006 Pacing up and down | 0.14 |
| oob/ sitting tolerance | 404951000 Activity tolerance | 102459008 Increased tolerance | 0.24 |
| violent/aggressive behavior | 248004009 Physical aggression | 61372001 Aggressive behavior | 0.50 |