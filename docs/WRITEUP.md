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
| + round-2 fine-tuning with iterative hard negatives | _TBD_ | _TBD_ |

Partial credit (ICD): correct 3-character category 77.7%, chapter 87.0%. SNOMED semantic-tag accuracy 72.4%.
Long tail (codes unseen in training folds): ICD _..._, SNOMED _..._.
**Abstention:** confidence is well calibrated (ICD terms with confidence > 0.9: 92.9% correct). Abstaining on the
least-confident half gives **90.6% accuracy at 50% coverage** (ICD).

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
_To be filled with final numbers — see results/error_analysis_final.md._
Failure modes (fine-tuned model): ICD — retrieval miss 12.8%, specificity within category 31.5%, wrong category same
chapter 25.0%, wrong chapter 30.7%. SNOMED — retrieval miss 15.4%, wrong semantic tag 45.4%, same tag wrong concept 39.2%.
Hardest slices: SNOMED procedures (40%) and labs (48%) vs diagnoses (62%); OCR-noisy terms; negated mentions.

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
