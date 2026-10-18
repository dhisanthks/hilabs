# Error analysis — `ftsap2+ftsap+ctxfeat_final` (5-fold out-of-fold predictions)


## ICD10CM

Exact match **68.8%** on 2306 terms. Candidate recall 95.8%.

Partial credit: 3-char category accuracy **78.7%**, chapter accuracy 87.9%.


**Tolerant metrics (how much of the gap is gold-standard ambiguity)**

| metric | accuracy |
|---|---|
| exact match | 68.8% |
| exact or same-name twin (identical preferred name) | 68.9% |
| exact or near-synonym (name similarity >= 90) | 72.2% |
| correct 3-character category | 78.7% |

**Failure modes (share of errors)**

| mode | errors | share |
|---|---|---|
| retrieval miss (gold not in candidates) | 96 | 13.4% |
| specificity (right 3-char category, wrong code) | 214 | 29.8% |
| right chapter, wrong category | 184 | 25.6% |
| wrong chapter | 225 | 31.3% |

**Accuracy by input slice**

| bucket | n | accuracy |
|---|---|---|
| gold code seen in training = True | 1757 | 75.8% |
| gold code seen in training = False | 549 | 46.6% |
| contains abbreviation = True | 274 | 79.2% |
| contains abbreviation = False | 2032 | 67.4% |
| OCR / typo / markup noise = True | 163 | 72.4% |
| OCR / typo / markup noise = False | 2143 | 68.5% |
| history / status-post = True | 14 | 28.6% |
| history / status-post = False | 2292 | 69.1% |
| negation cue = True | 29 | 58.6% |
| negation cue = False | 2277 | 69.0% |
| laterality = True | 236 | 72.5% |
| laterality = False | 2070 | 68.4% |
| ≥6 tokens = True | 243 | 63.4% |
| ≥6 tokens = False | 2063 | 69.5% |

**Accuracy by subconcept**

| bucket | n | accuracy |
|---|---|---|
| DiagnosisCode | 2305 | 68.8% |

**Calibration: accuracy by confidence bin**

| bucket | n | accuracy |
|---|---|---|
| (-0.001, 0.2] | 45 | 6.7% |
| (0.2, 0.4] | 189 | 14.8% |
| (0.4, 0.6] | 289 | 40.8% |
| (0.6, 0.8] | 343 | 54.5% |
| (0.8, 0.9] | 290 | 72.1% |
| (0.9, 1.0] | 1150 | 90.6% |

**Coverage vs accuracy (abstain on lowest-confidence terms)**

| coverage | accuracy |
|---|---|
| 100.0% | 68.8% |
| 90.0% | 75.0% |
| 80.0% | 79.3% |
| 70.0% | 83.8% |
| 50.0% | 90.6% |
| 30.0% | 96.4% |

**Sample errors**

| entity | gold | predicted | conf |
|---|---|---|---|
| other wound left toe | S91.102A Unspecified open wound of left lesser toe(s)  | S91.302A Unspecified open wound, left foot, initial en | 0.67 |
| substance use disorder | F19.99 Other psychoactive substance use, unspecified | F19.10 Other psychoactive substance abuse, uncomplic | 0.75 |
| fear | F40.9 Anxiety disorder, unspecified | F41.9 Anxiety disorder, unspecified | 0.34 |
| clicking | R29.898 Other symptoms and signs involving the muscul | R09.89 Other specified symptoms and signs involving  | 0.86 |
| oe pressure | R07.89 Other chest pain | R52 Generalized pain NOS | 0.32 |
| l radiculopathy | M54.10 Radiculopathy, site unspecified | M54.16 Radiculopathy, lumbar region | 0.98 |
| sinus allergy | J30.9 Allergic rhinitis, unspecified | Z88.9 Allergy status to unspecified drugs, medicame | 0.37 |
| adrenal mass | E27.8 Other specified disorders of adrenal gland | D35.00 Benign neoplasm of unspecified adrenal gland | 0.77 |
| left atrium volume index is moderately increased | R93.1 Abnormal findings on diagnostic imaging of he | I51.7 Ventricular dilatation | 0.93 |
| bilateral edema | R60.9 Edema, unspecified | R60.0 Localized edema | 0.72 |
| decreased ue strength | R29.898 Other symptoms and signs involving the muscul | R53.1 Asthenia NOS | 0.42 |
| peripheral edema | R60.9 Edema, unspecified | R60.0 Localized edema | 0.64 |
| jaw pops on both sides intermittent | M26.69 Other specified disorders of temporomandibula | R68.84 Mandibular pain | 0.64 |
| mucous membrane abnormalities | K13.70 Unspecified lesions of oral mucosa | R89.9 Unspecified abnormal finding in specimens fro | 0.15 |
| vaginal discharge | N89.8 Other specified noninflammatory disorders of  | R11.10 Vomiting, unspecified | 0.17 |

## SNOMEDCT_US

Exact match **53.1%** on 7340 terms. Candidate recall 92.8%.

Partial credit: semantic-tag accuracy **74.5%**.


**Tolerant metrics (how much of the gap is gold-standard ambiguity)**

| metric | accuracy |
|---|---|
| exact match | 53.1% |
| exact or same-name twin (identical preferred name) | 53.4% |
| exact or near-synonym (name similarity >= 90) | 61.3% |

**Failure modes (share of errors)**

| mode | errors | share |
|---|---|---|
| retrieval miss (gold not in candidates) | 531 | 15.4% |
| wrong semantic tag (e.g. substance vs procedure) | 1499 | 43.6% |
| same semantic tag, wrong concept | 1410 | 41.0% |

**Accuracy by input slice**

| bucket | n | accuracy |
|---|---|---|
| gold code seen in training = True | 3999 | 63.0% |
| gold code seen in training = False | 3341 | 41.3% |
| contains abbreviation = True | 1004 | 52.9% |
| contains abbreviation = False | 6336 | 53.2% |
| OCR / typo / markup noise = True | 675 | 46.7% |
| OCR / typo / markup noise = False | 6665 | 53.8% |
| history / status-post = True | 42 | 52.4% |
| history / status-post = False | 7298 | 53.1% |
| negation cue = True | 102 | 46.1% |
| negation cue = False | 7238 | 53.2% |
| laterality = True | 377 | 62.1% |
| laterality = False | 6963 | 52.6% |
| ≥6 tokens = True | 582 | 44.7% |
| ≥6 tokens = False | 6758 | 53.9% |

**Accuracy by subconcept**

| bucket | n | accuracy |
|---|---|---|
| Device | 437 | 46.7% |
| DiagnosisCode | 2700 | 63.1% |
| EncounterType | 567 | 54.9% |
| LabService | 858 | 49.8% |
| ProcedureCode | 2777 | 45.2% |

**Calibration: accuracy by confidence bin**

| bucket | n | accuracy |
|---|---|---|
| (-0.001, 0.2] | 1279 | 13.1% |
| (0.2, 0.4] | 1715 | 32.2% |
| (0.4, 0.6] | 1369 | 52.7% |
| (0.6, 0.8] | 1247 | 70.3% |
| (0.8, 0.9] | 801 | 86.0% |
| (0.9, 1.0] | 929 | 96.0% |

**Coverage vs accuracy (abstain on lowest-confidence terms)**

| coverage | accuracy |
|---|---|
| 100.0% | 53.1% |
| 90.0% | 57.9% |
| 80.0% | 62.9% |
| 70.0% | 67.7% |
| 50.0% | 77.7% |
| 30.0% | 87.8% |

**Sample errors**

| entity | gold | predicted | conf |
|---|---|---|---|
| care for self in the home | 284773001 Ability to perform personal care activity | 105389006 Personal care assistance at home | 0.23 |
| leukocytes | 375701000119105 Presence of white blood cells in stool | 767002 White blood cell count | 0.28 |
| impingement syndrome | 310401000119102 Impingement syndrome of right shoulder region | 239960007 Shoulder impingement syndrome | 0.52 |
| various patterns of stepping picking up various objects from | 51998003 Therapeutic exercise | 288362004 Ability to pick up objects | 0.15 |
| education | 311401005 Patient education (procedure) | 409073007 Education procedure | 0.62 |
| electric clippers | 462539003 Hair clipper, electric | 16497000 Electric clipper | 0.89 |
| very tender to palpation | 838411007 Pain in abdominal region on palpation | 247348008 Sore to touch | 0.26 |
| oxygen supplies | 268512000 Home oxygen supply | 57485005 Oxygen therapy care | 0.25 |
| provisc | 387578003 Hyaluronate sodium | 360273002 Provision - action | 0.15 |
| get along with other people | 88598008 Social interaction disorder | 225597007 Ability to interact with others | 0.10 |
| radiopaque device | 456151000124107 Foreign body (physical object) | 419098001 Radiographic imaging contrast media | 0.64 |
| n/v/d/c/c/s | 267045008 Gastrointestinal symptom | 260413007 None  | 0.06 |
| cvs | 113257007 Cardiovascular structure | 118672003 Cardiovascular procedure | 0.49 |
| pelvic exam | 35025007 Manual pelvic examination (procedure) | 83607001 Gynecologic examination | 0.38 |
| hip and leg strengthening | 229152003 Lower limb exercises | 6397004 Muscular strength development exercise | 0.75 |