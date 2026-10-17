# Clinical Term Mapping Engine — ICD-10-CM & SNOMED CT
*HiLabs Hackathon 2026 — technical write-up*  ·  Demo video: _link_  ·  Repository: `dhisanthks/hilabs` (private)

## 1. Problem and how we scoped it
Input: a clinical entity extracted by CodEx plus its source chunk. Output: one ICD-10-CM code and one SNOMED CT
concept from the supplied dictionaries (98,222 ICD codes / 382,124 SNOMED concepts with 984,585 descriptions).

What the training data actually looks like shaped every decision:
- **9,646 labelled rows** in long format (one row per entity per system): 2,306 ICD, 7,340 SNOMED. SNOMED rows span
  diagnoses (37%), procedures (38%), labs (12%), encounter facts (8%) and devices (6%) — wider than "Problem only".
- **Every entity string occurs once.** Exact lookup is useless; the engine must generalize.
- **Long tail:** 466 of 867 ICD codes and 2,964 of 4,335 SNOMED concepts occur once; 24% (ICD) and 46% (SNOMED) of
  held-out terms have a gold code never seen in the training folds.
- **Local conventions dominate:** `can not squat` → R26.2 *Difficulty in walking*; `rashes` → *Eruption*;
  `phosphorus` (lab) → *Phosphate measurement*. Only 6% (ICD) / 15% (SNOMED) of terms match any dictionary string.
- **Label self-consistency ceiling:** among near-duplicate training terms the majority gold code agrees only
  **93.7% (ICD) / 84.6% (SNOMED)** of the time (e.g. `ble edema`→R60.9 vs `mild ble edema`→R60.0). This bounds any
  exact-match score and answers the brief's open question on inter-annotator agreement empirically.

Evaluation protocol: 5-fold cross-validation **grouped by source chunk**; every learned component (encoder
fine-tuning, cross-encoder, rerankers, kNN memory, priors) is re-trained inside each fold, so every reported number
is on terms the system never saw.

## 2. Architecture
```
 entity + chunk
   │ 1 normalize: unidecode, markup/(hcc) strip, ~350 multi-sense clinical abbreviations, spelling/OCR repair
   │   (glued words, stutter, edit-distance to dictionary vocabulary) → query variants (original kept)
   ▼
 2 candidate generation (union, ~250 ICD / ~475 SNOMED codes per term; recall ICD 95.8%, SNOMED 92.5%)
   ├ char 2–4-gram TF-IDF and BM25 over all dictionary descriptions
   ├ fine-tuned SapBERT bi-encoder over descriptions (+ top-15 inside each of 13 SNOMED semantic tags)
   ├ base SapBERT over descriptions
   └ kNN over training terms (fine-tuned, base, char) → "similar terms were coded as X"
   ▼
 3 R1  LightGBM LambdaRank, ~95 features: retriever scores/ranks (exact back-fill), fuzzy similarity, specificity
       (extra / missing qualifier words), laterality agreement, ICD structure, SNOMED tag, code/category priors,
       P(tag | subconcept)
   ▼
 4 cross-encoder (SapBERT, listwise) on R1 top-30, reading term + ±120 chars of chunk context → R2
   ▼
 5 ICD↔SNOMED coupling: each system's R2 top-5 → co-occurrence + name-similarity features for the other → R3
   ▼
   code + display + calibrated confidence + needs_review flag
```

## 3. Results (5-fold CV, exact match)
| Stage | ICD-10-CM | SNOMED CT |
|---|---|---|
| char TF-IDF + kNN baseline | 41.2 | 32.8 |
| + normalization + LambdaRank over lexical retrievers | 55.9 | 43.9 |
| + base SapBERT dense retrieval and kNN | 63.1 | — |
| + per-fold fine-tuned SapBERT + tag-stratified retrieval | 66.7 | 51.1 |
| + cross-encoder with chunk context | 67.5 | 51.6 |
| + ICD↔SNOMED coupling | 67.7 | 51.8 |
| + round-2 fine-tuning (iterative hard negatives) + ensemble with round-1 encoder (**final**) | **68.3** | **52.2** |
| final, **diagnosis/condition terms only** (brief's scope) | 68.2 | **62.7** |

Partial credit: ICD correct 3-character category **78.5%**, chapter 87.7%; SNOMED correct semantic tag 73.2%.
Long tail (gold code unseen in the training folds — 24% of ICD, 46% of SNOMED terms): ICD **46.3%**, SNOMED **40.3%**
(seen codes: 75.1% / 62.2%).
**Abstention:** confidence is calibrated — confidence > 0.9 is right 91.9% (ICD, 48% of terms) and 96.2% (SNOMED).
| coverage (abstain on lowest confidence) | 100% | 90% | 70% | 50% | 30% |
|---|---|---|---|---|---|
| ICD accuracy | 68.3 | 73.9 | 84.2 | 91.3 | 95.7 |
| SNOMED accuracy | 52.2 | 56.9 | 66.3 | 77.2 | 87.7 |

## 4. Design decisions
- **Retrieve-then-rerank with many cheap retrievers + one learned stacker.** Each retriever is good at a different
  failure mode (typos, paraphrase, local convention, priors); LightGBM LambdaRank learns how to weigh them.
  Reranker hyper-parameters moved accuracy ≤1 point (7 settings tested) — gains come from signals, not tuning.
- **Encoder choice by benchmark, not by default.** 7 encoders benchmarked zero-shot (SapBERT, BioLORD-2023,
  gte-base, MedCPT, BGE, MedEmbed, e5); SapBERT best all-round → fine-tuned per fold with in-batch +
  hard negatives. Biggest single gain (+3.6 ICD, +7.2 SNOMED); iterative hard negatives add +5–6 points to the
  encoder's own top-1.
- **Specificity calibration is explicit**, not accidental: features count qualifier words the candidate adds
  (over-coding) or misses (under-coding); "unspecified"/"other"/leaf flags; specificity errors are balanced
  (46 over vs 60 under), i.e. no systematic bias.
- **Both systems predicted independently, then coupled.** A learned crosswalk recovers only 37% of ICD codes even
  from the *gold* SNOMED code, so we never derive one system from the other; agreement is a feature.
- **Context** enters through the cross-encoder (term + chunk window) and abbreviation fan-out (`MS` → multiple
  sclerosis / mitral stenosis / mental status are all retrieved; context decides).
- **Determinism & deployability:** fixed seeds, LightGBM deterministic mode, no sampling; all models local
  (no chart data leaves the machine); runs on a 4 GB laptop GPU; ~_x_ ms/term at inference.

## 5. Error analysis (final model)
| Failure mode (share of errors) | ICD (732 errors) | SNOMED (3,509 errors) |
|---|---|---|
| gold never retrieved (recall) | 13% | 15% |
| right ICD category, wrong specificity | 31% | — |
| right chapter, wrong category / same SNOMED tag, wrong concept | 25% | 40% |
| wrong chapter / wrong SNOMED semantic tag | 31% | 45% |

By slice: SNOMED diagnoses 62.7% vs procedures 44.3%, labs 47.7%, devices 45.8% (non-diagnosis SNOMED rows follow
idiosyncratic conventions, e.g. `phosphorus` → *Phosphate measurement*, `roll supine` → *Positioning subject in supine
position*); OCR-noisy terms 44.9% (SNOMED) vs 52.9%; negated 45.1%; long phrases (≥6 tokens) 43.8%; abbreviations are
*easier* than average after expansion (ICD 78.1%). Many residual errors are near-synonym ties where the gold standard
itself is inconsistent (label self-consistency 93.7% / 84.6%), e.g. `acute resp failure` gold *Acute respiratory failure*
vs predicted *Acute hypoxemic respiratory failure*; J96.01 vs J96.21 for two spellings of the same phrase.

## 6. What we tried and abandoned
| Idea | Result | Why dropped |
|---|---|---|
| Local LLM adjudicator (Qwen3-4B-Instruct, 4-bit, few-shot with retrieved training examples) | alone 59% ICD / 39% SNOMED vs reranker 68% / 51%; fixes 35% of reranker's recoverable errors but breaks more; blend +0.25 | gain within noise; 0.3–0.7 s/term |
| Concept-type classifier (chapter / semantic tag) | 80% chapter acc., reranker unchanged | redundant with reranker |
| Context-kNN (MedCAT-style concept context embeddings) | ICD 66.9 → 66.7 | redundant |
| Bigger / differently-tuned LightGBM | ±0.1–1 point | noise |
| Longer bi-encoder training (4 epochs) | no change | harder negatives matter, not epochs |

## 7. Innovations
Tag-stratified dense retrieval for SNOMED; iterative hard-negative fine-tuning on the engine's own OOF confusions;
explicit specificity features; joint ICD↔SNOMED coupling with name similarity (generalizes to unseen code pairs);
calibrated confidence with abstention curve; measured label-consistency ceiling.

## 8. Remaining gap and next steps
_Fill at end._ Ideas: ensemble of differently-initialized fine-tuned encoders (n2c2 winner style), longer
cross-encoder training (xMEN uses ~20 epochs), section-header-aware context, SNOMED IS-A relations (not in the
supplied dictionary) for hierarchical partial credit and post-processing, active-learning loop on low-confidence terms.

## 9. External resources
SapBERT (cambridgeltl/SapBERT-from-PubMedBERT-fulltext, Apache-2.0); benchmarked: BioLORD-2023, gte-base, MedCPT,
BGE-base-en-v1.5, MedEmbed-base-v0.1, e5-base-v2; Qwen3-4B-Instruct-2507 (Apache-2.0, tested, not shipped);
dictionaries supplied by the challenge; hand-curated clinical abbreviation list; PyTorch, sentence-transformers,
transformers, LightGBM, scikit-learn, RapidFuzz.
