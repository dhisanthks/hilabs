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
