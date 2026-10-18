# Work Log — HiLabs Clinical Term Mapping Engine

Running log of every step, experiment, number and decision. Source material for the write-up, README and demo.

## 2026-10-16 — Day 0: Data understanding

### Environment
- Windows 11, Python 3.13, NVIDIA RTX 3050 Laptop (4 GB VRAM), CUDA 13.1 driver.
- Project venv `.venv` with CUDA PyTorch (cu128) + sentence-transformers, faiss, bm25s, rapidfuzz, lightgbm.

### Training data (`hackathon_train.csv`)
- 9,646 rows. Columns: `entity, chunk_text, coding_system, code, display, subconcept`.
  **Long format** (one row per entity per coding system) — differs from the brief's `chunk_id, clinical_entity` schema; engine will accept both.
- Coding systems: SNOMEDCT_US 7,340 rows, ICD10CM 2,306 rows.
- Entity+chunk groups: 7,517 → both systems 2,124; SNOMED-only 5,213; ICD-only 180.
- `subconcept`: DiagnosisCode 5,005; ProcedureCode 2,778; LabService 858; EncounterType 567; Device 437; Medication 1.
  ICD rows are ~all DiagnosisCode; SNOMED rows span all subconcepts (wider than the brief's "Problem only").
- Labels are internally consistent: each entity maps to exactly one code per system.
- **Every entity string appears in exactly one chunk** → eval terms will be new strings; exact-lookup memorization is useless, generalization is everything.
- Long tail: ICD 867 distinct gold codes, 466 seen once; SNOMED 4,335 distinct, 2,964 seen once.
- Gold ICD code length (no dot): 3 chars 119, 4 chars 1,069, 5 chars 775, 6 chars 235, 7 chars 108 → 3-char categories are legal gold targets.
- Chunk text: median 168 chars, max 23k; often table-ish (`header_1: ... header_2: ...`).

### Dictionaries
- ICD-10-CM: 159,306 rows (code, description), 98,222 distinct codes, incl. range headers (e.g. `A00-B99`) and multiple descriptions per code.
- SNOMED CT US: 984,585 descriptions, 382,124 concepts; `description_order`; FSN semantic tags available for all concepts.
  Notes sheet: HiLabs' own Term Map index uses `gte_base_fp32`.
- Gold coverage: SNOMED 100% in dictionary; ICD 99.9% (one gold label `L6.0` is not a valid code → label noise).
- `display` column = one of the dictionary descriptions of the gold code (SNOMED 100%, ICD 94.7%).

### Baseline signal
- Normalized entity exact-matches some dictionary description: ICD 6.4% (gold among matches 65%), SNOMED 15.4% (gold 79%).
  → String lookup alone is hopeless; the coders' conventions (e.g. "can not squat" → R26.2 Difficulty in walking) must be learned.
- SNOMED semantic tag of gold vs subconcept: DiagnosisCode → finding/disorder; ProcedureCode → procedure/regime; Device → physical object. Strong reranking feature.

### Plan
normalize → multi-source candidate generation (BM25, char-TF-IDF, dense encoder(s), kNN over training terms, cross-system links) → learned reranker (LightGBM; test cross-encoder & local LLM) → code + confidence.
Validation: 5-fold GroupKFold by chunk; metrics per system: exact match, long-tail (code unseen in train fold), ICD category accuracy.

## Exp01 — Lexical baselines (5-fold CV grouped by chunk) — `experiments/exp01_lexical.py`
| Method | ICD10CM | SNOMEDCT_US |
|---|---|---|
| char(2-4) TF-IDF over dictionary, top-1 | 23.4% | 30.1% |
| ... recall@10 / @50 / @100 | 46.4 / 58.0 / 61.5% | 53.4 / 66.4 / 71.1% |
| kNN (char TF-IDF) over training terms, top-1 | 40.0% | 22.9% |
| kNN if sim≥θ else dictionary | 41.2% (θ=.5) | 32.8% (θ=.8) |
| Ceiling for pure memorization: gold code seen in train fold | 76.2% | 54.5% |

Takeaways: data is far harder than the brief's "mid-80s string baseline" — coder conventions map
lay/qualified phrasing to abstract codes ("can not squat"→R26.2). Lexical recall@100 is only 61–71%,
so **candidate recall is the first bottleneck**; semantic encoders + training-term kNN must fill it.
Over half of SNOMED eval codes will be unseen in training → dictionary retrieval must carry the long tail.

## Normalization layer — `src/normalize.py`, `src/abbrev.py`, `src/spell.py`
- `clean`: unidecode, lowercase, strip `<br>`/HTML, `(hcc)`-style billing tags, stray punctuation.
- `abbrev.expand`: ~350 curated clinical abbreviations (general chart-documentation knowledge, not fitted to data).
  Ambiguous ones fan out to every sense (ms → multiple sclerosis / mitral stenosis / mental status; ra; pid; pe; ed; ca);
  the reranker decides with context. Expansions are **added as query variants**, the original string is always kept.
- `Speller`: vocabulary = 47k frequent tokens from all dictionary descriptions. OOV tokens (≥5 chars) are
  (a) split if two glued words (`chronickidney`→`chronic kidney`, `valveprolapse`→`valve prolapse`),
  (b) cut to in-vocab prefix if OCR stutter (`spondylosispondylosisondylosis...`→`spondylosis`),
  (c) else nearest frequent word within edit distance 1–2 (`cardiomyopahty`→`cardiomyopathy`, `exertinoal`→`exertional`).
- Mined OOV tokens in training entities: 1,606 types (abbrev: sob, t2dm, osa, pna, hfref, b/l, s/p; typos; OCR).

## Cross-system analysis
- 2,124 mentions have both gold codes. SNOMED per ICD code: 1.82; ICD per SNOMED: 1.13.
- Learned crosswalk (majority pairing in train folds) given the **gold** other code: ICD 37.0%, SNOMED 27.8%
  → deriving one system from the other is a dead end (confirms the brief); use as a reranker feature only.
- The two systems sometimes encode the same mention differently on purpose:
  "history of degenerative disc disease" → ICD Z87.39 (personal history) but SNOMED 77547008 (the disease).
- ICD chapter mix: R 30%, M 13%, I 13%, J 6%, E 6%, K 4%, L 4%, Z 3.5%. 35% of ICD gold codes are "unspecified";
  98% are leaf codes.

## Exp03 — Normalization impact on lexical recall — `experiments/exp03_lexical_norm.py`
Query = all variants (original + abbreviation expansions + spelling repair), max score per code.
| System / retriever | R@1 clean → variants | R@10 | R@100 |
|---|---|---|---|
| ICD char TF-IDF | 23.4 → 25.1 | 46.4 → 50.8 | 61.5 → 66.1 |
| ICD BM25 | 18.0 → 19.8 | 38.7 → 42.7 | 52.6 → 57.5 |
| SNOMED char TF-IDF | 30.1 → 32.7 | 53.4 → 56.9 | 71.1 → 74.4 |
| SNOMED BM25 | 26.3 → 29.1 | 45.5 → 48.8 | 62.2 → 67.0 |
Normalization is worth +3–5 points of recall everywhere; char n-grams beat BM25 (typo/OCR robustness).

## Exp04a — Lexical-only candidates + LambdaRank reranker (ICD) — `experiments/exp04_rerank_cv.py lex`
- Candidates = union(char top-60, BM25 top-60, char-kNN over training terms top-25) ≈ 110/row.
- Candidate recall (gold anywhere in pool): **84.4%** (vs 66% for char alone).
- LightGBM LambdaRank, 5-fold CV, ~74 features (retriever scores/ranks with exact back-fill, fuzzy ratios,
  query flags, ICD code meta, code/category/chapter priors from train fold, laterality agreement, P(chapter|subconcept)).
- **ICD exact match 55.9%** (seen codes 64.0%, unseen codes 29.9%; n_unseen=549/2306). Up from 41.2% baseline.
- Bug caught: the ranker's own `score` column was initially included as a feature (leaked earlier folds' scores;
  gave a fake 54.2% with 'score' as top feature). Fixed by excluding it in `ranker.NON_FEATURES`.
- Top features: char_sim_gap, char_sim, code_prior, bm25_rank, fuzzy ratios, chapter, leaf flag, unspecified flag.

## Exp04b — Lexical-only reranker, SNOMED
- Candidate recall 82.7%; **SNOMED exact match 43.9%** (seen 55.7%, unseen 29.8%; n_unseen=3,341/7,340). Up from 32.8%.
- Top features: char_rank, kchar_rank/kchar_sum (training-term kNN), **p_grp_sub** (P(semantic tag | subconcept)),
  bm25_rank, code_prior, c_tag.

## Exp02 — Off-the-shelf dense encoder benchmark (zero-shot) — `experiments/exp02_encoders.py`
Dictionary retrieval recall (code-level, max over synonyms) + kNN@1 over training terms. Same folds.
| Encoder | ICD R@1 | R@10 | R@100 | ICD kNN@1 | SNOMED R@1 | R@10 | R@100 | SNOMED kNN@1 | SNOMED enc. time |
|---|---|---|---|---|---|---|---|---|---|
| char TF-IDF (ref) | 23.4 | 46.4 | 61.5 | 40.0 | 30.1 | 53.4 | 71.1 | 22.9 | — |
| SapBERT (PubMedBERT, UMLS-synonym trained) | 32.8 | 61.0 | 76.5 | 48.5 | 38.6 | 63.2 | 79.2 | 26.7 | 327s |
| gte-base (HiLabs' current Term Map encoder) | 28.3 | 57.7 | 75.3 | 47.1 | 38.0 | 63.8 | 80.2 | 25.8 | 379s |

## Error analysis #1 (lexical reranker) — what fails
- SNOMED: 4,117 errors = 1,269 gold-not-in-pool + 2,848 ranking errors. **2,322 errors (56%) pick the wrong semantic tag**
  (e.g. `phosphorus` [LabService] gold *Phosphate measurement* (procedure) → predicted *Phosphorus* (substance);
  `roll supine` gold *Positioning subject in supine position* → predicted *Supine position*).
  Accuracy by subconcept: Diagnosis 53%, Encounter 47%, Device 42%, Lab 38%, Procedure 36%.
- ICD: 1,018 errors = 359 not-in-pool + 659 ranking; 236 are same 3-char category (specificity), 537 same chapter.
  Typical: `ckd 3` → should be N18.30, `tricompartmental osteoarthritis` → M17.9 (knee implied), `muscle soreness` → M79.10.
- **Fix implemented: tag-stratified dense retrieval** — besides the global top-k, take top-15 codes inside each of 13 SNOMED
  semantic tags (computed from one GPU score matrix, no extra memory) + feature `dense_*_grank` (rank within own tag).

## Ops note
- Disk ran out (pip cache held 8.5 GB of wheels) → `pip cache purge` → 18.8 GB free. Dense embedding caches are fp16.

## Exp04c — + SapBERT (dense dictionary retrieval + dense kNN over training terms), ICD
- Candidates ≈ 197/row; **candidate recall 93.1%** (lexical-only 84.4%).
- **ICD exact match 63.1%** (seen 70.8%, unseen 38.6%) — up from 55.9%.
- Top features: knn_sapbert_rank, dense_sapbert_rank/sim/gap, knn_sapbert_sum. → Bottleneck is now **ranking**, not recall.

## Exp02 (complete) — 7 off-the-shelf encoders, zero-shot
| Encoder | ICD R@1 | ICD R@10 | ICD R@100 | ICD kNN@1 | SNOMED R@1 | SNOMED R@10 | SNOMED R@100 | SNOMED kNN@1 |
|---|---|---|---|---|---|---|---|---|
| SapBERT (cambridgeltl/SapBERT-from-PubMedBERT-fulltext) | 32.8 | 61.0 | 76.5 | **48.5** | **38.6** | 63.2 | 79.2 | **26.7** |
| BioLORD-2023 (FremyCompany/BioLORD-2023) | **33.6** | **64.0** | **81.1** | 46.5 | 36.7 | 59.9 | 75.4 | 25.3 |
| gte-base (thenlper/gte-base) | 28.3 | 57.7 | 75.3 | 47.1 | 38.0 | **63.8** | **80.2** | 25.8 |
| MedCPT query enc. (ncbi/MedCPT-Query-Encoder) | 25.3 | 60.0 | 79.3 | 47.2 | 35.2 | 60.8 | 77.6 | 25.0 |
| BGE-base-en-v1.5 | 25.2 | 54.3 | 72.8 | 45.7 | 37.5 | 63.1 | 79.8 | 25.6 |
| MedEmbed-base-v0.1 | 25.3 | 52.7 | 71.4 | 45.6 | 37.2 | 63.0 | 79.7 | 25.6 |
| e5-base-v2 | 24.3 | 51.2 | 70.8 | 44.9 | 38.0 | 62.7 | 79.1 | 25.0 |
**Decision:** SapBERT = best all-rounder (best SNOMED R@1, best kNN on both) → fine-tuning base.
BioLORD (best ICD recall) and gte (best SNOMED recall) kept as candidate complementary views.
General-domain models (BGE/e5) are competitive on SNOMED but clearly weaker on ICD.
Engine change: dense indexes are paged onto the GPU per encoder (4 GB VRAM can't hold two SNOMED indexes).

## Label-consistency ceiling (important for the write-up)
Grouped training terms that are identical after normalization + stop/severity-word removal + token sort.
| | groups (rows) | groups with conflicting gold | majority-label agreement |
|---|---|---|---|
| ICD10CM | 76 (159) | 10 | **93.7%** |
| SNOMEDCT_US | 192 (416) | 59 | **84.6%** |
Examples: `acute hypoxic and hypercapnic respiratory failure`→J96.21 vs `acute hypoxic/hypercapnic respiratory failure`→J96.01;
`ble edema`→R60.9 vs `mild ble edema`→R60.0; `phq-9`→720433000 vs `phq 9`→715252007;
`impaired activity tolerance`→Activity intolerance vs `impaired<br>activity tolerance`→Impaired exercise tolerance.
→ The gold standard's own self-consistency on near-duplicates is below the 95% bar (esp. SNOMED). This is an
empirical proxy for inter-annotator agreement and bounds achievable exact-match accuracy.
Also: ICD specificity errors are balanced (46 over-specific vs 60 under-specific out of 239 same-category errors) → no
systematic bias to correct by rule.

## Exp05 — Fine-tuned SapBERT (per fold) — early read, ICD fold 0 only
Fine-tune: MultipleNegativesRanking loss + 1 mined hard negative (lexically closest wrong code), 2 epochs, lr 2e-5,
batch 64, shared model for ICD+SNOMED, trained only on the 4 training folds. Base SapBERT features kept alongside.
| ICD fold 0 | base SapBERT | fine-tuned SapBERT |
|---|---|---|
| dense dictionary retrieval top-1 | 34.6% | **41.6%** |
| kNN over training terms top-1 | 49.9% | **53.0%** |
| candidate recall (pool ≈250/row) | 93.7% | **96.1%** |
Full 5-fold reranked result pending (folds 1–4 queued; ~25–30 min per fold, dominated by re-encoding 984k SNOMED descriptions).

## Exp05 — fold 0 complete (both systems)
- SNOMED fold 0: candidates ≈475/row (tag-stratified), **candidate recall 92.2%** (lexical-only 82.7%).
  Dense top-1: base 40.8% → fine-tuned 42.1% (smaller fine-tune gain than ICD's 34.6 → 41.6).
- Fine-tune cost: 454 steps, 238 s on RTX 3050; re-encoding 984k SNOMED descriptions ≈13 min/fold (the bottleneck).
- Quick inner-CV on fold 0 only (ranker sees ~1/3 of usual data → conservative), same protocol for all tables:
| fold 0 | ICD | SNOMED |
|---|---|---|
| lexical + reranker | 56.5% | 41.7% |
| + base SapBERT | 63.0% | — |
| + fine-tuned SapBERT + tag-stratified retrieval | **66.3%** | **50.2%** |

## Subconcept robustness check
The brief's eval schema (`chunk_id, clinical_entity`) may not include `subconcept`. Ranker retrained without
`p_grp_sub`/`subconcept` features: ICD 63.1% → 64.0% (no dependence; all ICD rows are DiagnosisCode),
SNOMED 43.9% → 42.9% (−1.0). → Engine trains both variants and picks by the input's columns.

## Priority note
Headline metric = min(ICD, SNOMED) → SNOMED is the bottleneck; next steps are ordered by expected SNOMED gain:
cross-encoder (ranking errors dominate), ICD↔SNOMED coupling, second encoder (gte had best SNOMED recall).

## SNOMED error anatomy (fine-tuned, fold 0, 735 errors / 1,476)
Gold rank among errors: 2nd 168, 3rd 79, 4–5th 79, 6–10th 80, 11–100th 176, >100 38, not retrieved 115 (7.8%).
→ ~1/3 of errors are near-misses between near-synonyms (`acute resp failure`: gold *Acute respiratory failure*,
picked *Acute hypoxemic respiratory failure*) → target for a cross-encoder + specificity features.
By subconcept: Diagnosis 62.4%, Encounter 54.5%, Lab 48.1%, Device 43.8%, Procedure 39.9%.
Context-dependent / noisy labels exist (`exercise regularly` → gold *Gets no exercise*).

## Specificity-calibration features — `features.specificity_features`
sp_extra_n/frac = candidate content words not in any query variant (over-specific);
sp_miss_n/frac = query content words not in the description (under-specific); min over the code's descriptions.
Fold-0 inner CV: ICD 66.3 → 66.3, SNOMED 50.2 → 50.6 (neutral/slightly positive) → kept for interpretability.

## Exp05 — RESULT: fine-tuned SapBERT, full 5-fold CV (`results/exp05_ftsap.json`)
| 5-fold CV | ICD10CM | SNOMEDCT_US |
|---|---|---|
| lexical + reranker (Exp04a/b) | 55.9% | 43.9% |
| + base SapBERT (Exp04c) | 63.1% | — |
| **+ fine-tuned SapBERT (per fold) + base SapBERT + tag-stratified SNOMED retrieval** | **66.7%** | **51.1%** |
| candidate recall | 95.8% | 92.5% |
| acc on seen / unseen codes | 74.5 / 41.7% | 61.4 / 38.8% |
| acc given gold in pool | 69.7% | 55.2% |
Top features now: dense_ft_sim_gap, dense_ft_rank, knn_ft_rank (fine-tuned encoder dominates).
Long tail improved strongly (SNOMED unseen 29.8 → 38.8%). Per-fold cost ≈25–30 min (SNOMED re-encode dominates).

## Exp08a — ICD↔SNOMED coupling, co-occurrence features only (`results/exp08_ftsap_nosim.json`)
Stage-2 reranker with cp_cooc / cp_cooc_w / cp_has from the other system's OOF top-5 (fold-aware pair counts).
ICD 66.9 → **67.4%** (+0.5); SNOMED 50.7 → **51.2%** (+0.5).
Coverage explains the small gain: 92% of ICD mentions have a SNOMED sibling, only 29% of SNOMED mentions have an
ICD sibling (most SNOMED rows are labs/procedures/devices). Name-similarity variant (cp_sim) pending GPU.

## Exp07 — cross-encoder (in progress)
First attempt (15 negatives, 128 tokens, term+context): ~53 min/fold → too slow for the budget; restarted with
9 negatives, 96 tokens, 6 groups/batch → ~22 min/fold. Joint ICD+SNOMED model per fold, listwise CE loss.

## Error analysis #2 — fine-tuned model, full OOF (`experiments/error_analysis.py ftsap` → `results/error_analysis_ftsap.md`)
**ICD10CM 66.7%** — partial credit: 3-char category 77.7%, chapter 87.0%.
Errors: retrieval miss 12.8% | specificity (right category) 31.5% | right chapter wrong category 25.0% | wrong chapter 30.7%.
Slices: seen 74.5% / unseen 41.7%; abbreviation 77.0% (expansion works); OCR/noise 71.8%; history 35.7% (n=14).
**SNOMEDCT_US 51.1%** — semantic-tag accuracy 72.4%.
Errors: retrieval miss 15.4% | wrong semantic tag 45.4% | same tag, wrong concept 39.2%.
Slices: seen 61.4% / unseen 38.8%; OCR/noise 43.6%; negation 41.2%; ≥6 tokens 43.6%; laterality 58.9%.
**Calibration is good → abstention works** (ICD): conf>0.9 → 92.9% correct (39% of terms); conf 0.8–0.9 → 81.4%;
conf<0.2 → 13.5%. Coverage/accuracy: 100%→66.7, 90%→71.9, 80%→77.5, 70%→82.3, **50%→90.6**, 30%→94.4.
→ Next: concept-type classifier (Exp09) targets the 45% wrong-tag SNOMED errors and 31% wrong-chapter ICD errors.

## Exp07 — cross-encoder, fold-0 read
Cross-encoder alone (rerank top-30): ICD 42.2%, SNOMED 40.7% (weak standalone; gold in top-30: 93.2% / 85.5%).
As a **feature in LightGBM** (fold-0 inner CV): ICD 66.3 → **69.8%** (+3.5), SNOMED 50.6 → **52.0%** (+1.4).
→ Complementary signal; kept. Remaining folds running.

## Exp09 — concept-type classifier (negative result)
Logistic regression on SapBERT embedding + char n-grams (+ subconcept) predicting ICD chapter / SNOMED semantic tag.
OOF type accuracy: ICD chapter 80.2% (majority 29.9%), SNOMED tag 66.7% (majority 25.5%) — but the SNOMED tag
accuracy is *below* the reranker's own 72.4%, and adding p_type to the ICD reranker gave 66.9 → 66.7%.
→ The reranker already captures type information; dropped.

## Research — see `docs/RESEARCH.md`
xMEN (char TF-IDF + SapBERT + trained cross-encoder) ≈ our architecture; DrivenData SNOMED challenge winner relied on
training-annotation dictionaries + section context; n2c2 2019 top = 85.3% with a 5-model BERT ensemble; MedCAT uses
concept context embeddings → implemented as context-kNN features (Exp10, running).

## Exp10 — context-kNN (MedCAT idea), ICD
Features: cosine of the mention's surrounding chunk text (entity removed) / entity+context to training mentions of each
candidate code (max, mean, gap), fold-aware. ICD: 66.9 → 66.7% (no gain; features get importance but are redundant).
SNOMED run deferred (RAM: 3.5M-row table + cross-encoder process exceeded 15 GB).
**Pattern so far:** extra hand-crafted features (type classifier, context-kNN, bigger ranker) → ~0; new *models*
(fine-tuned encoder +3.6/+7.2, cross-encoder +3.5/+1.4 on fold 0) → real gains. Remaining GPU time goes to models.

## Exp06b — reranker capacity on SNOMED (fine-tuned table)
default (31 leaves, lr .05, 600) 50.7% | 63 leaves 50.6% | 127 leaves lr .03 900 rounds 50.7% | 63 leaves trunc30 lr .03 1000 50.7%.
→ No gain; LightGBM defaults kept. (Same conclusion as ICD Exp06: ±1 point, noise level.)

## Exp07 — RESULT: cross-encoder stacked, full 5-fold CV (`results/exp07_ftsap_cectx.json`)
Cross-encoder (SapBERT init, listwise, 1 epoch, 9 hard negatives, query = term + ±120-char chunk context, 96 tokens),
scores R1's top-30; features ce / ce_rank / ce_gap + r1 into R2.
| | ICD | SNOMED |
|---|---|---|
| R1 (fine-tuned stack) | 66.9% | 50.7% |
| **R2 (+cross-encoder)** | **67.5%** | **51.6%** |
| cross-encoder alone (top-30) | 40.6% | 40.0% |
| unseen-code accuracy R1 → R2 | 44.3 → 46.5% | 37.8 → 39.6% |
Lesson: the fold-0 inner-CV estimate (+3.5/+1.4) overstated the gain — that protocol trains the ranker on 1/3 of a fold;
only full 5-fold numbers are reported as results from here on. CE training: ~22–30 min/fold on RTX 3050.

## Exp08b — coupling with name similarity, on top of the cross-encoder (ICD)
cp_sim / cp_sim_w (SapBERT similarity between candidate name and the other system's top-5 names) + co-occurrence:
ICD R2 67.5 → **67.7%**. SNOMED run crashed (RAM, concurrent LLM job) → rerun pending.

## Exp11 — local LLM adjudicator (Qwen3-4B-Instruct-2507, 4-bit bnb, Apache-2.0), prototype
Prompt: term + chunk context + 5 retrieved *training* examples (fold-aware few-shot of coder conventions) + R2 top-10
lettered candidates; next-token letter log-probs (deterministic, one forward pass).
300 random terms/system: LLM alone ICD 58.7% vs R2 65.7%; SNOMED 41.3% vs R2 51.7%; gold in top-10 87.3% / 83.0%.
→ A 4B local model is clearly weaker than the trained reranker as a decision-maker.
Speed: 0.34 s/term in isolation (batch 4, 3.3 GB peak VRAM, ~340-token prompt); the prototype's 4.6 s/term was caused by
RAM swapping from a concurrent job. Complementarity check (does it fix R2's errors?) running.
Complementarity (400 terms/system, run alone): LLM fixes 35% of R2's errors whose gold is in the top-10
(ICD: fixes 7.2% of terms, breaks 15.5%; SNOMED: fixes 11.2%, breaks 23.0%).
CV-tuned blend log p_R2 + w·log p_LLM: ICD 67.75 → 68.0%, SNOMED 50.5 → 50.5% (w≈0.1).
**Decision:** LLM not in the main pipeline — gain within noise, costs ~2 h OOF + 0.3–0.7 s/term at inference.
Documented as a tested cascade stage (the brief's "LLM adjudication on low-confidence cases").

## Exp08b — RESULT: full stack R3 = fine-tuned retrieval + cross-encoder + coupling (co-occurrence + name similarity)
| 5-fold CV | ICD | SNOMED |
|---|---|---|
| R2 (+cross-encoder) | 67.5% | 51.6% |
| **R3 (+coupling)** | **67.7%** | **51.8%** |
| seen / unseen codes | 74.6 / 45.7% | 61.7 / 39.9% |

### Progress summary so far (exact match, 5-fold CV)
| Stage | ICD | SNOMED |
|---|---|---|
| char TF-IDF + kNN baseline | 41.2 | 32.8 |
| + normalization + LightGBM reranker over lexical retrievers | 55.9 | 43.9 |
| + SapBERT dense retrieval / kNN | 63.1 | — |
| + per-fold fine-tuned SapBERT + tag-stratified SNOMED retrieval | 66.7 | 51.1 |
| + specificity features | 66.9 | 50.7 |
| + cross-encoder (term + context) | 67.5 | 51.6 |
| + ICD↔SNOMED coupling | **67.7** | **51.8** |

## Exp12 — bi-encoder fine-tuning variants (proxy: fold 0, encoder alone re-ranking R1 top-50; kNN@1)
| variant | ICD dense top-1 | ICD kNN@1 | SNOMED dense top-1 | SNOMED kNN@1 |
|---|---|---|---|---|
| base2: 2 epochs, 1 lexical hard negative (current) | 41.1 | 53.0 | 41.1 | 30.2 |
| ep4: 4 epochs | 40.9 | 53.4 | 41.1 | 29.8 |
| **iter2: iterative hard negatives (top-3 wrong codes by round-1 model's OOF score), 2 epochs** | **46.8** | 51.6 | **46.1** | 30.1 |
| iter3: iterative, 3 epochs | 46.8 | 51.4 | 46.2 | 30.2 |
→ Harder negatives, not longer training, improve the encoder (+5–6 points top-1). kNN unaffected.
Note: round-1 OOF scores used only to *select negatives* (wrong codes) for round-2; label leakage negligible.
→ Rebuilding all 5 folds with the round-2 recipe (`ftsap2`, `--iter-from ftsap`); cross-encoder scores reused
(they depend only on term/candidate text).

## Exp05b — round-2 fine-tuning (iterative hard negatives), partial read on folds 0–2 (ICD, 3-fold CV)
| ICD folds 0–2 | R1 acc | candidate recall |
|---|---|---|
| round-1 encoder (lexical negatives) | 66.9% | 95.9% |
| round-2 encoder (iterative negatives) | 65.7% | 96.1% |
| **round-1 table + round-2 encoder features** | **68.3%** | — |
| round-2 table + round-1 encoder features | 68.35% | — |
Round-2 is better at dictionary matching but slightly worse at term↔term kNN (proxy kNN@1 53.0 → 51.6), the ranker's
strongest signal → alone it loses; **together the two encoders are complementary (+1.4)** — an ensemble, as the
n2c2 2019 winner found. Final engine carries both fine-tuned encoders (+ base SapBERT).
Ops: fold 3 crashed (3.8 GB dense score matrix in lexical top-k at batch 512) → batch 64.
Ops: disk down to 6.7 GB → removed benchmark-only models from the HF cache (BioLORD, gte, BGE, e5, MedEmbed, MedCPT,
Qwen3-4B; 5.1 GB, re-downloadable) and superseded experiment tables → 12 GB free.
Pipeline updated: final engine = round-2 fine-tuned SapBERT (`models/biencoder`) + round-1 fine-tuned SapBERT
(`models/biencoder_r1`, features *_ft1_*) + base SapBERT, cross-encoder, R1/R2/R3 LightGBM.

## Exp13 — FINAL STACK RESULT (`results/exp13_ftsap2+ftsap.json`)
Round-2 fine-tuned tables (+ tag-stratified retrieval, specificity) + round-1 encoder features (*_ft1_*, ensemble) →
R1 → + cross-encoder (scores joined by term/code; covers 77% of SNOMED / ~90% ICD R1 top-30) R2 → + coupling R3.
| 5-fold CV | ICD R1 | ICD R2 | **ICD R3** | SNOMED R1 | SNOMED R2 | **SNOMED R3** |
|---|---|---|---|---|---|---|
| exact match | 67.6 | 67.6 | **68.3** | 51.6 | 52.4 | **52.2** |
| unseen codes | 44.3 | 46.8 | 46.3 | 39.6 | 40.2 | 40.3 |
Round-2 alone (no ensemble): ICD 66.9, SNOMED 51.3. Ensemble of both fine-tuned encoders: +0.7 / +0.3 at R1.
Final model = R3 for both systems (SNOMED R2 vs R3 difference = 12 terms, noise). Headline (min) ≈ 52.2%.

## Error analysis — FINAL model (`results/error_analysis_ftsap2+ftsap_final.md`)
**ICD 68.3%** — category (3-char) 78.5%, chapter 87.7%. Errors (732): retrieval miss 96 (13%), specificity 225 (31%),
same chapter wrong category 183 (25%), wrong chapter 228 (31%). Seen 75.1% / unseen 46.3%. Abbreviations 78.1%,
OCR/noise 73.0%, laterality 70.8%, ≥6 tokens 63.0%, history 35.7% (n=14).
**SNOMED 52.2%** — semantic-tag accuracy 73.2%. Errors (3,509): retrieval miss 531 (15%), wrong tag 1,591 (45%),
same tag wrong concept 1,387 (40%). Seen 62.2% / unseen 40.3%. OCR/noise 44.9%, negation 45.1%, ≥6 tokens 43.8%.
**By subconcept (SNOMED): Diagnosis 62.7% | Encounter 53.1% | Lab 47.7% | Device 45.8% | Procedure 44.3%.**
→ On the brief's in-scope entity type (diagnoses/conditions) the two systems score ICD 68.2% / SNOMED 62.7%.
Calibration: conf>0.9 → ICD 91.9% (48% of terms), SNOMED 96.2% (12% of terms).
Coverage→accuracy: ICD 100%→68.3, 70%→84.2, 50%→91.3, 30%→95.7 | SNOMED 100%→52.2, 70%→66.3, 50%→77.2, 30%→87.7.

## Final production training — `python run.py train` (resumed from cached OOF artifacts): 124 min
Fitted R1/R2/R3 × {sub, nosub} × 2 systems (12 LightGBM models), final round-1 and round-2 bi-encoders and the
cross-encoder on all 9,646 rows → `models/`. End-to-end predict test running.

## Mentor feedback (2026-10-17) → plan
1. Quality-weighted retriever fusion. 2. Stage-wise funnel (top-1/5/10/30/50 at each stage) to locate where accuracy
drops. 3. Focus on SNOMED.

## Exp14a — per-retriever quality (final round-2 tables, OOF; `results/exp14_retrievers_*.csv`)
| SNOMED retriever | R@1 | R@5 | R@10 | R@50 | R@100 | MRR |
|---|---|---|---|---|---|---|
| fine-tuned dense, **rank within gold's own semantic tag** | 54.5 | 73.1 | 77.9 | 80.5 | 80.5 | 0.627 |
| base dense, within own tag | 48.6 | 66.2 | 71.8 | 74.5 | 74.5 | 0.563 |
| fine-tuned dense (global) | 45.7 | 67.1 | 73.5 | 84.0 | 87.2 | 0.553 |
| base dense (global) | 40.1 | 58.7 | 65.3 | 77.1 | 81.0 | 0.487 |
| char TF-IDF | 32.7 | 49.8 | 56.9 | 69.8 | 71.1 | 0.408 |
| BM25 | 29.0 | 43.2 | 48.8 | 61.9 | 63.2 | 0.358 |
| kNN fine-tuned / base / char | 28.7 / 26.7 / 22.9 | | | 48.2 / 46.3 / 43.3 | | 0.34 / 0.32 / 0.29 |
Union pool recall 92.8% (470 cands/term).
| ICD retriever | R@1 | R@5 | R@10 | R@50 | R@100 | MRR |
|---|---|---|---|---|---|---|
| fine-tuned dense | 46.5 | 75.4 | 80.7 | 89.4 | 91.5 | 0.593 |
| kNN fine-tuned | 51.1 | 64.7 | 68.6 | 71.4 | 71.4 | 0.571 |
| kNN base | 48.6 | 62.4 | 66.5 | 69.9 | 69.9 | 0.546 |
| kNN char | 40.0 | 55.0 | 59.4 | 64.1 | 64.1 | 0.468 |
| base dense | 34.1 | 55.9 | 62.5 | 74.7 | 77.9 | 0.443 |
| char TF-IDF / BM25 | 25.1 / 19.8 | | | 62.2 / 54.0 | | 0.34 / 0.27 |
Union pool recall 95.8% (246 cands/term).
**Key SNOMED finding:** if the correct semantic tag were known, the fine-tuned encoder alone would rise from 45.7% to
54.5% top-1 (+8.8) → concept-type selection is the main SNOMED bottleneck. kNN is far weaker for SNOMED (R@1 29%
vs 51% ICD) — SNOMED terms are less repetitive, so local-convention memory helps less.
→ Exp14 (full funnel R1/CE/R2 + weighted RRF) and Exp15 (two-level type ranker) queued.
Ops: inference re-encoded base SapBERT under a new cache tag → fixed to reuse `emb_sapbert_*`; one cp1252 byte in
`src/engine.py` (from a Windows text-mode patch) removed; all sources normalized to LF.

## Exp14 — FUNNEL (recall@k at each stage, OOF) — `results/exp14_funnel_*.json`
| SNOMED stage | @1 | @3 | @5 | @10 | @30 | @50 |
|---|---|---|---|---|---|---|
| best retriever (fine-tuned dense) | 45.7 | 61.0 | 67.1 | 73.5 | 81.1 | 84.0 |
| quality-weighted RRF (weights = fold-aware MRR) | 38.0 | 58.6 | 68.7 | 77.4 | 84.8 | 87.2 |
| union pool | 92.8 | | | | | |
| R1 LightGBM | 51.5 | 68.1 | 73.5 | 79.4 | 85.8 | 88.5 |
| cross-encoder (within R1 top-30) | 40.1 | 57.3 | 65.2 | 75.8 | 85.7 | 88.4 |
| R2 (+CE) | 52.5 | 69.6 | 75.4 | 81.0 | 86.3 | 88.0 |
| ICD stage | @1 | @3 | @5 | @10 | @30 | @50 |
| best retriever (kNN fine-tuned / dense fine-tuned) | 51.1 / 46.5 | 63.4 / 69.1 | 67.3 / 75.4 | 70.3 / 80.7 | 71.8 / 87.5 | 72.8 / 89.4 |
| weighted RRF | 52.3 | 68.0 | 75.2 | 85.1 | 91.8 | 92.8 |
| union pool | 95.8 | | | | | |
| R1 | 67.8 | 81.8 | 85.5 | 88.9 | 92.5 | 94.2 |
| R2 (+CE) | 67.6 | 82.4 | 85.5 | 89.4 | 92.7 | 94.1 |
**Where accuracy drops:** not recall (pool 92.8 / 95.8%), but **top-10 → top-1: −28.5 (SNOMED), −21.8 (ICD)**.
Weighted RRF (mentor idea) as a ranker is far below LightGBM (38.0 vs 51.5 SNOMED top-1); as an extra R1 feature
ICD 67.6 → 67.8, SNOMED 51.6 → 51.5 (noise) → LightGBM already learns (context-dependent) retriever weights.

## End-to-end test — `run.py predict` on brief-format file (12 hand-written hard cases): OK (825 s incl. one-off
dictionary encoding with final encoders). Correct: HTN→I10, chronickidney disease stag 3→N18.30/CKD stage 3,
cardiomyopahty→I42.9, diabetes→E11.9 (unspecified), t2dm with diabetic neuropathy→E11.40, left knee pain→M25.562,
h/o herpes zoster→Z86.19, pulmonary hypertension→I27.20, copd exacerbation (hcc)→J44.1, MS (echo)→SNOMED Mitral
valve stenosis. Failure: ICD for MS in echo context → G35.D (multiple sclerosis) — ICD side follows the training
prior for "ms" and ignores context. Bugs fixed: long-format output dtype (pandas 3 strict string columns); ICD display
showed index synonyms → output display now = coders' display for known codes / dictionary title otherwise.

## Exp15 — two-level type ranker (negative result)
Type ranker (LambdaRank over the semantic tags / chapters in each term's pool; per-type aggregates of OOF R1 score,
dense/char similarity, kNN type votes, P(type|subconcept)).
| | SNOMED | ICD |
|---|---|---|
| type accuracy of R1's top candidate | 72.5% | 87.4% |
| type accuracy of dedicated type ranker | 72.5% | 87.0% |
| gold type in pool | 99.8% | 99.9% |
| R1 → + type-ranker features (→ stacking-only ablation) | 51.7 → 51.3 (51.1) | 67.9 → 67.4 (67.7) |
→ R1 already uses all available type evidence; residual type errors follow labelling convention (substance vs
measurement procedure for labs, finding vs procedure for "ProcedureCode" rows) not predictable from these signals.
Next lever: the only joint deep model — retrain the cross-encoder on the final candidate tables, 2 epochs.

## End-to-end test 2 — training-format input (300 sampled training rows; in-memory terms → plumbing check only)
`run.py predict` 300 rows in 127 s (~0.4 s/term incl. model load); ICD 95.3%, SNOMED 83.6% on these memorized rows;
74/300 flagged needs_review. Long-format output bug fixed.

## Near-synonym share of errors (final model)
Token-set similarity between predicted and gold display names among errors: SNOMED ≥90: 17.5%, ≥80: 23.3%, ≥70: 32.1%;
ICD ≥90: 10.1%, ≥80: 18.0% (32.2% same 3-char category). Examples: *Able to walk up stairs* vs *Ability to walk up
stairs*; *Bathing* vs *Bathing - activity of daily living*; *Eosinophil count* (procedure) vs *Eosinophil count*
(observable entity). Crediting ≥90-similar predictions would give SNOMED ≈60.6%, ICD ≈71.5% → a large part of the
gap is gold-standard granularity/ambiguity (brief's open question on "genuinely ambiguous gold answers").

## Exp16 — cross-encoder v2 (running): retrained on final R1 candidates (100% top-30 coverage), 2 epochs; then re-stack.

## Label ambiguity — actions (2026-10-17 evening)
1. **Output:** pipeline now emits `alt_code`, `alt_display`, `alt_confidence` (runner-up) and `ambiguous`
   (runner-up prob ≥ 0.5× winner, or near-synonym names ≥ 90) per system — for review / set-based scoring.
2. **Tolerant metrics** added to `experiments/error_analysis.py`:
| | exact | + same-name twin | + near-synonym (≥90) | right ICD category |
|---|---|---|---|---|
| ICD | 68.3 | 68.3 | **71.5** | 78.5 |
| SNOMED | 52.2 | 52.4 | **60.6** | — |
3. **Exp17 — ambiguity features** (twin_n, prior/tag/dense-sim vs best near-synonym twin in R1 top-10; majority share
   among identically-worded training terms, fold-aware). ICD: R1 67.6 → 67.8 (noise). Majority convention covers only
   137 ICD terms; twin_n importance mostly reflects "in R1 top-10". SNOMED run deferred (RAM; CE v2 training).

## Write-up PDF — `docs/HiLabs_TermMapping_Writeup.pdf` (built by `python docs/build_pdf.py`)
3 pages + appendix: headline metrics (all / diagnosis-only / long tail / partial credit / near-synonym), problem &
data findings, architecture diagram, accuracy build-up table, design decisions, error analysis (funnel, failure modes,
type bottleneck, gold ambiguity, hard slices, calibration/abstention), abandoned ideas, innovations, next steps,
external resources; appendices: funnel table, encoder benchmark, SNOMED accuracy by entity type.

## Exp16 — RESULT: cross-encoder v2 (final-R1 negatives, 2 epochs) + re-stack (`results/exp13_ftsap2+ftsap.json`)
| | CE alone (top-30) ICD | SNOMED | R2 ICD | R2 SNOMED | **R3 ICD** | **R3 SNOMED** |
|---|---|---|---|---|---|---|
| v1 (round-1 negatives, 1 epoch) | 40.6 | 40.1 | 67.6 | 52.4 | 68.3 | 52.2 |
| **v2** | **48.6** | **42.7** | 68.0 | 53.1 | **68.4** | **53.1** |
Coverage of final R1 top-30 by v2 scores: 84% (v2 negatives/top-30 came from the exp14 R1 run, which included the
wRRF feature → slightly different top-30). SNOMED +0.9.

## FINAL MODEL — error analysis (`results/error_analysis_ftsap2+ftsap_final.md`)
ICD **68.4%** (seen 75.1 / unseen 46.8), category 79.0%, near-synonym 71.8%; errors: miss 96, specificity 233,
same-chapter 178, wrong chapter 222. SNOMED **53.1%** (seen 63.2 / unseen 41.0), same-name 53.3%, near-synonym 61.2%;
errors: miss 531, wrong tag 1,518, same-tag 1,394. **SNOMED by entity type: Diagnosis 63.2% | Encounter 53.1% |
Lab 51.0% | Device 46.5% | Procedure 44.9%.** Calibration conf>0.9: ICD 91.4%, SNOMED 96.1%.
Coverage→accuracy: ICD 70%→84.0, 50%→90.8, 30%→95.5 | SNOMED 70%→67.4, 50%→76.9, 30%→88.8.
Pipeline switched to CE v2 (`CE_RUN=cectx2`, 2 epochs); final cross-encoder to be retrained on all data.
Exp17 SNOMED: 354 terms have identically-worded training terms; 37,654 top-10 candidates have a near-synonym twin.
R1 51.6 → 51.5% with ambiguity features (noise) → not adopted. Coder tie-breaking between twins is not learnable from
these signals; ambiguity handled on the output side (alt_code / ambiguous flag) and in reporting (tolerant metrics).

## Full improvement review (2026-10-17 23:50) — what else can raise SNOMED?
**External data (researched):** COMETA (Reddit → SNOMED, 20k), CADEC (AskaPatient, 6.7k), PsyTAR (6.5k) cover lay
disorder/ADR terms only — not our weak areas (labs/procedures/devices) nor our coders' conventions; licences need
requests. Official SNOMED→ICD-10-CM map, SNOMED US Edition (synonyms + IS-A hierarchy) require a UMLS licence
(approval takes days) → not usable before the deadline; our crosswalk test bounds the map's value (37%).
**Architecture (researched):** BeLink (2026) and BioNLP-2025 report BERT cross-encoders competitive with GPT-4o
rerankers; no evidence that small LoRA-tuned LLMs beat them on SNOMED; our Qwen3-4B test agrees. QLoRA on a 4 GB GPU
would need ~3–4 h per fold → infeasible for honest 5-fold evaluation.
**Gaps found in our own system:**
1. The LightGBM rankers see **no chart context** (only the cross-encoder does, and only for the top-30) although the
   #1 SNOMED error is concept type, which the chunk often reveals (lab tables with units, PT exercise logs, social
   history). → Exp18: OOF context-type classifier (entity + subconcept + chunk words) + chunk-style flags as features.
2. The cross-encoder never sees the **entity type** (subconcept) → it cannot learn "lab → measurement procedure".
   → CE v3: `[subconcept] term | context` queries with 50% subconcept dropout (one model works with/without it),
   scored both ways; negatives/top-30 from the final R1 (100% coverage, fixes v2's 84%).
3. CE v2 scores were selected by a slightly different R1 (84% coverage) → v3 uses `stages_final`.
**Plan:** context test → Exp18 features → re-stack; CE v3 CV (GPU, ~5 h) → re-stack; adopt what wins; final retrain.

## Context signal test (OOF logistic regression predicting SNOMED semantic tag)
entity 59.6% | entity + subconcept 64.4% | entity + chunk 61.1% | entity + subconcept + chunk 65.1%
(final ranker's own top-1 tag accuracy 73.2%) → chunk adds ~+0.7–1.5 points of type information; expected gain of
chunk features in the ranker is small. Exp18 features still evaluated end-to-end (exp13 … ctxfeat).

## Incident (2026-10-18 01:10): final cross-encoder save failed — "not enough space on the disk"
90-min final CE v2 training completed but `save_pretrained` failed (disk full at that moment; 12 GB free afterwards —
most likely Windows page-file growth on C: while several RAM-heavy jobs ran concurrently). models/crossencoder was left
with only config.json → `run.py predict` temporarily broken. Fix: removed the partial model; re-sequenced strictly one
heavy job at a time: exp13 + context features → `run.py train` (final CE retrain) → CE v3 CV.
Lesson for deployment: write model artifacts atomically (temp dir + rename) and check free space before saving.

## Exp18 — RESULT: chunk-context features in the full stack (`results/exp13_ftsap2+ftsap_cectx2+ctxfeat.json`)
| 5-fold CV | R1 | R2 | R3 |
|---|---|---|---|
| ICD (before → with context) | 67.6 → 67.8 | 68.0 → 68.7 | 68.4 → **68.9** |
| SNOMED (before → with context) | 51.6 → 51.9 | 53.1 → **53.8** | 53.1 → 53.6 |
→ Adopted. SNOMED R3 (coupling) has been ≤ R2 in 3 of the last 4 stacks → **final stage: ICD R3, SNOMED R2**.
Pipeline: `src/context.py` (type classifiers with/without subconcept, chunk flags), OOF features in training,
same features at inference; FINAL_STAGE per system; candidate tables freed before final GPU training.

## Incident 2 (02:50): disk at 0 GB during final CE training — page file had grown to 27.7 GB (RAM pressure from the
training process holding candidate tables + a concurrent experiment). Deleted regenerable caches (final-encoder
dictionary embeddings, superseded tables) → save succeeded; disk back to 19 GB once the process exited.
Final cross-encoder v2 saved (87 min). Fix in pipeline: free tables before GPU stage.

## Exp19 — diagnosis-only evaluation scenario (SNOMED, DiagnosisCode terms, 5-fold CV, final features + CE v2, top-50)
A all rows, subconcept known 63.2% | B all rows, no subconcept 62.6% | C diagnosis rows only 62.9% |
D all rows, diagnosis ×3 weight 63.4% (noise).
→ No specialized retrain: the all-rows ranker is best. If the test set is diagnosis-only, run with
`--assume-subconcept DiagnosisCode` (+0.6 vs. no subconcept).

## Long-tail generalization (judging criterion, 20%) — accuracy by gold-code frequency in the training folds
(final model incl. context features; honest: every learned component re-trained without the evaluated fold)
| frequency | ICD share | ICD acc | SNOMED share | SNOMED acc |
|---|---|---|---|---|
| absent (0) | 23.8% | 46.8% | 45.5% | 41.8% |
| rare (1–2) | 24.4% | 69.8% | 31.9% | 57.7% |
| 3–5 | 21.9% | 78.6% | 15.1% | 69.5% |
| 6–20 | 29.2% | 78.6% | 7.2% | 76.4% |
| >20 | 0.7% | 81.2% | 0.3% | 60.0% (n=20) |
| **rare-or-absent (≤2)** | 48% | **58.4%** | 77% | **48.3%** |

## Decision: fine-tuned LLM reranker — not pursued before the deadline
Evidence: zero-shot Qwen3-4B fixes 35% of the ranker's recoverable errors (complementary) but literature (BeLink 2026)
finds BERT cross-encoders competitive with GPT-4o rerankers in biomedicine. On a 4 GB GPU honest 5-fold CV of a
QLoRA 4B model ≈ 25–40 h (1.7B ≈ 10 h) and inference 10–20× slower. A fold-0 pilot (~2 h) was offered and declined to
keep GPU time for the final model. → Listed as next step (A100-class GPU; apply only to low-confidence terms).

## Exp20 — countermeasures for label noise and close-call decisions (final features, top-50 pool, original gold)
| | SNOMED | ICD |
|---|---|---|
| current ranker | 53.41 | 68.82 |
| training on cleaned labels (majority-resolved near-duplicates; 212 / 31 labels changed) | 53.45 (noise) | 68.65 |
| **5-seed bagged rankers (70% rows / 70% features each)** | **53.94 (+0.53)** | **68.99 (+0.17)** |
→ Bagging adopted (`ranker.fit_bag`, pipeline saves `<stage>_<system>_<variant>__bag{0..4}.txt`, inference averages);
label cleaning dropped (evaluation gold is still inconsistent; too few labels change).

## Fine-tuned LLM reranker — cloud GPU job prepared (cloud GPUs approved by organizers)
`experiments/exp21_make_llm_prompts.py` → `kaggle/data/llm_prompts.jsonl` (9,646 terms; candidates = bagged stack's
top-10 [ICD top-1 69.0%, gold in top-10 89.8%; SNOMED 53.9% / 81.6%]; 8,061 terms usable for training).
`kaggle/llm_rerank_kaggle.py`: QLoRA Qwen3-4B-Instruct-2507 (4-bit), listwise cross-entropy over letter logits, candidate
order shuffled during training (no position bias), 5-fold OOF scores + final adapter, both T4s in parallel, per-fold
checkpointing. Prompt length median 257 / p95 365 tokens (left truncation for the 0.1% > 640).
Integration plan: OOF LLM scores → features (llm_lp, llm_rank, llm_gap) in a new stage R2+LLM; adopt only if 5-fold CV
improves; inference falls back to the current engine if LLM files are absent.

## LLM reranker on Kaggle — smoke tests (fold 0, 200 held-out terms) and launch
zero-shot (lr 0) 47.0% | after 200 training examples 21.0% (shuffled candidate order removes the position prior before
content is learned) | **after 1,500 examples 58.5%** (loss 3.53 → 1.27) → learning confirmed.
Speed on T4: ~17 s per optimizer step (16 examples) → ~2 h per fold. Full run (5 folds + final adapter, 2× T4 in
parallel, ~6.3 h) launched 05:28 as Kaggle Version #1 (commit run).
LLM integration prepared: `src/llm_rerank.py` (local scorer with the identical prompt/letter scoring; OOF loader);
`exp13_stack.py` with `LLM_DIR=kaggle/llm_out` adds the R2+LLM stage and reports LLM-alone / R2+LLM accuracy;
pipeline: R2+LLM rankers (`r2l_*`, bagged) fitted when the 5 OOF score files exist; inference loads
`models/llm_adapter` (unzipped adapter_final.zip) and scores the R2 top-10; clear error if trained-with-LLM rankers
are used without the adapter.

## Exp16b — cross-encoder v3 (subconcept-aware, 50% tag dropout, final-R1 negatives, 100% top-30 coverage)
CE alone: ICD 48.4% (with tag) / 47.8% (without); SNOMED 42.5% / 42.1% (v2: 48.6 / 42.7).
Full stack with context features (`results/exp13_ftsap2+ftsap_cectx3*+ctxfeat.json`):
| | ICD R3 | SNOMED R2 |
|---|---|---|
| CE v2 (current) | 68.9 | **53.8** |
| CE v3, query without subconcept | **69.5** | 53.1 |
| CE v3, query with subconcept | 68.8 | 53.1 |
→ v3 helps ICD (+0.6) but hurts SNOMED (−0.7); headline = min(ICD, SNOMED) → **keep v2** for both (no second CE at
inference). The entity-type tag did not help the cross-encoder resolve SNOMED type confusions.
Note: exp13 prediction files were named without the cross-encoder run, so the v3 evaluations overwrote the v2+context
predictions → fixed (names now include the CE run; R2 predictions saved too). Final-model predictions (ICD R3, SNOMED
R2 = shipped configuration) are regenerated after the final retrain → `cache/pred_FINAL_*`, `results/error_analysis_FINAL.*`.
Final production retrain started 09:25 (context features + 5-seed bagged R1/R2/R3, CE v2, encoders reused).

## FINAL SHIPPED MODEL — 5-fold CV (`results/error_analysis_FINAL.md`; ICD = R3, SNOMED = R2)
| | ICD | SNOMED |
|---|---|---|
| exact match | **68.9%** | **53.8%** |
| diagnosis terms | 68.9% | **63.6%** |
| never-seen / seen codes | 46.8 / 75.8% | 41.8 / 63.7% |
| rare-or-absent (≤2) | 58.4% | 48.8% |
| near-synonym credit / ICD category | 71.9 / 78.9% | 61.5% |
| most-confident 30% / 50% / 70% | 96.0 / 91.7 / 84.3% | 88.0 / 77.2 / 67.8% |
SNOMED by entity type: Diagnosis 63.6 | Encounter 55.0 | Lab 51.6 | Device 45.8 | Procedure 45.9.
Shipped rankers are 5-seed bagged (+0.2 / +0.5 measured separately in Exp20, not included above).
Final production retrain (context + bagging): 62 min → 60 bagged rankers + 3 context models in `models/`.

## Final end-to-end tests (final models, 11:00)
- Brief schema (`chunk_id, clinical_entity, chunk_text`), 12 hand-written hard cases: 776 s first run (rebuilds final-
  encoder dictionary caches), 68 s afterwards. Output keeps the input's own columns + icd10cm_/snomed_ code, display,
  confidence, alt_code, alt_display, ambiguous, needs_review. 10/12 clearly correct (HTN→I10, chronickidney disease
  stag 3→N18.30, cardiomyopahty→I42.9, diabetes→E11.9, t2dm with diabetic neuropathy→E11.40, left knee pain→M25.562,
  h/o herpes zoster→Z86.19, pulmonary hypertension→I27.20, copd exacerbation (hcc)→J44.1, sob/doe→R06.02 / Dyspnea on
  exertion). MS (echo) → multiple sclerosis with low confidence, flagged ambiguous/needs_review, SNOMED runner-up =
  Mitral valve stenosis. ICD MS → G35.D (2025 code; G35 is the runner-up).
- Training schema (300 rows): 129 s; `--assume-subconcept DiagnosisCode` mode: 68 s. demo.py --cases: 10 cases OK.
- Fix: output now preserves the input's column names (was renaming clinical_entity → entity).

## Exp22 — RESULT: fine-tuned LLM reranker (Kaggle 2×T4, QLoRA Qwen3-4B-Instruct-2507) — `results/exp13_llm.log`
Cloud run: 5 folds × ~108–121 min + final adapter 140 min (6 h 23 min total). LLM alone, held-out folds (mixed
ICD+SNOMED): 56.1 / 56.6 / 52.8 / 54.7 / 56.6% (mean 55.4%; zero-shot was 47%).
In the full stack (OOF LLM scores → llm_lp / llm_rank / llm_gap features, new stage R2+LLM; LLM covers 83–84% of
R2's top-10 because its candidates came from a slightly different ranker):
| 5-fold CV | R2 | LLM alone | **R2+LLM** | R3 (+coupling) |
|---|---|---|---|---|
| ICD | 68.65 | 60.2 | 70.38 | **71.08** |
| SNOMED | 53.76 | 53.9 | **58.24** | 57.87 |
→ **Adopted. Final: ICD R3 71.1%, SNOMED R2+LLM 58.2%** (+2.2 / +4.5). Unseen codes: ICD 46.8 → 50.1, SNOMED 41.8 → 47.0.
Error analysis (`results/error_analysis_FINAL.md`): SNOMED Diagnosis 67.4 | Encounter 59.6 | Lab 55.9 | Device 53.5 |
Procedure 50.5; near-synonym credit ICD 74.4 / SNOMED 65.6; ICD category 81.6; long tail: never-seen 50.1 / 47.0,
rare-or-absent 61.5 / 53.7; most-confident 30%: 96.4 / 90.8.
Local inference: adapter in models/llm_adapter, 4-bit base, 2.8 GB VRAM, 0.41 s/term on the RTX 3050.
Production rankers refitting with the LLM stage (r2l_*).

## Evaluation scope confirmed: diagnosis/condition terms only
→ Relevant CV expectation: ICD 71.1%, SNOMED 67.4% (DiagnosisCode terms, final model with LLM); headline ≈ 67%.
→ Always run with `--assume-subconcept DiagnosisCode` (rankers use P(type|subconcept); the LLM prompt includes the
entity type and was always trained with one). No diagnosis-specialized retrain (Exp19: all-rows ranker is better).
README / PDF updated accordingly.

## Production refit with LLM stage — R3-SNOMED OOM → fixed
`run.py train` fitted R1/R2/R2+LLM (both systems) and R3-ICD (70 bagged files) then ran out of RAM fitting R3 for
SNOMED (3.2M × 124 float64) — a stage SNOMED does not use (final = R2+LLM). Rebuilt `models/ranker_features.json`
from the saved boosters' feature names (14 keys); training now skips R3 for systems whose FINAL_STAGE is not r3.

## Exp23 — LLM-2: SNOMED diagnosis-specialist LLM (in progress, ~6 h before deadline)
Prompts `kaggle/data2/llm2_prompts.jsonl`: 2,700 DiagnosisCode SNOMED terms, top-20 candidates (gold in top-20 90.7%
vs top-10 88.4%), 2,449 trainable; prompt median 354 / max 575 tokens; 20 letters A–T verified single tokens.
Script saves per-fold adapters now. Kaggle run (new dataset hilabs-llm2) ~2.5–3 h. Integration prepared: second adapter
on the same 4-bit base (PEFT multi-adapter), features llm2_lp/rank/gap in SNOMED R2+LLM; exp13 `LLM2_DIR=...` reports
DiagnosisCode accuracy. Expected gain +0.5–1.5 on SNOMED diagnosis; adopted only if 5-fold CV improves.

## Exp24 — can the LLM's candidate list have higher recall? (SNOMED DiagnosisCode, 5-fold OOF)
current list ranker (top-1-tuned, bagged) R@10 88.4 / R@20 90.7 | recall-oriented LambdaRank (NDCG@20, truncation 50)
88.4 / 90.5 | binary objective 88.5 / 90.6 | diverse list (ranker + CE/dense/kNN picks) 87.3 / 90.6;
ceiling = gold in final-R1 top-50 93.1% (pool 95.9%). → No better list; LLM-2 prompts unchanged. Further recall needs
better retrieval (encoder/table rebuild, many GPU hours).

## Inference speed fix (two-phase prediction)
In-pipeline LLM scoring was 5.7 s/term (300 terms in 1,724 s) because encoders, cross-encoder, paged dictionary index,
coupling name embeddings (0.74 GB) and the 4-bit LLM shared the 4 GB GPU → spill to system RAM. Now: phase A (retrieval,
CE, R2) for all systems with encoders on GPU; then encoders/CE moved to CPU, name embeddings kept on CPU, LLM loaded
once, scores every system's top-k (phase B), then unloaded. Measured: 100 terms in 145 s incl. ~60 s model load →
**~0.85 s/term with the LLM, ~0.2 s/term without** (~45 min per 3,000 terms). demo.py batches all cases in one run.

## Submission audit (judging criteria + requirements)
Accuracy & long tail: honest 5-fold numbers in PDF/README (+ appendix C long tail). Engineering: single command,
deterministic, README with setup/env/run, reproduction paths A (released models) and B (from scratch), latency/footprint
table; LLM OOF score CSVs (no chart text) now committed for reproducibility. Error analysis: PDF §5 + results/*FINAL*.
PDF: all required sections incl. new "7. Challenges & learnings"; tightened to 3 pages + 1-page appendix.
Open (user): demo video link (README + PDF), models.zip download link, repo access for judges, push.

## Exp25 — RESULT: LLM-2 (SNOMED diagnosis specialist, top-20, Kaggle 2×T4, 5 folds × ~45 min + final 54 min)
LLM-2 alone on held-out SNOMED DiagnosisCode terms: 61.5 / 64.1 / 62.7 / 59.4 / 58.5% (mean 61.2%).
Full stack (`results/exp13_llm2.log`): SNOMED DiagnosisCode 67.44% with LLM-1 + LLM-2 vs **67.44% with LLM-1 only**;
all SNOMED 58.07 vs 58.24; ICD (via coupling) 71.42 vs 71.08 (noise). → Not adopted (`USE_LLM2 = False`): redundant with
LLM-1 + stack; top-20 depth adds only 2.3 pts recall and those are the hardest cases; would cost +0.4 s/term.
Final model unchanged: ICD 71.1%, SNOMED 67.4% (diagnosis terms) / 58.2% (all).

## Version-drift handling (core requirement #7)
`src/dicts.check_dictionary_version()`: fingerprints the dictionary files (size + mtime) in `cache/DICT_VERSION.json`;
a changed file invalidates all dictionary-derived caches (parsed tables, *_system.pkl, lexical indexes, vocabulary,
description embeddings) which rebuild on demand. Previously a new release would have silently reused the old cache.

## Memory fix (prediction on a 16 GB machine)
Prediction failed with "paging file is too small" when the page file could not grow (disk low): all dictionary
embedding matrices (~5 GB) were loaded into RAM. DenseIndex and the coupling name embeddings now memory-map the .npy
caches (read on demand, not committed) → loads fine. Freed disk: removed unused LLM-2 adapters and duplicate LLM-1
adapter copies (kept score CSVs/logs), old ranker folders, pip cache.
Extra live demo cases tested: PID→N73.9, acute on chronic systolic chf→I50.23, htn uncontroled→I10, malignant
hypertension→I10 / SNOMED Malignant hypertension; known failure: RA in "echo: dilated RA" → rheumatoid arthritis.

## Brief's open questions — our evidence-based answers (added to PDF)
Partial credit: report hierarchical (ICD category 81.6%, SNOMED tag 76.9%) next to exact match. Ambiguous gold: ~20% of
SNOMED diagnosis errors are near-synonyms → accept sets of valid codes (engine outputs alt_code + ambiguous flag).
IAA ceiling: 93.7% / 84.6% measured, matching ~0.85 from the SNOMED CT Entity Linking Challenge. Per-category floors
(final model): ICD chapters 67–82% except Z 47.5% and T 51.4%; SNOMED diagnosis terms disorders 76.1%, findings 68.5%,
rare gold types (situation 18.5%, procedure 20.6%, qualifier 21.4%, observable 32.7%).
PDF: 3 pages + appendix page (external resources moved to the appendix page).

## 2026-10-18 ~20:00 — Write-up revised per reviewer comments
- All SNOMED numbers now diagnosis-only (67.4 exact, 57.4 unseen, 84.4 tag, 73.9 near-synonym, 95.9 @30% coverage); accuracy table completed with diagnosis-only column; bagged-ranker row removed.
- Added: how models are trained (fine-tuned SapBERT rounds 1/2, LambdaRank retriever weighting, cross-encoder, QLoRA LLM), design decisions tagged by stage, flagging mechanism, 7 failure-mode core-requirement table, open-question answers, label ambiguity, future improvements.
