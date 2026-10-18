# HiLabs Clinical Term Mapping Engine

Maps a clinical entity extracted from a medical chart (plus its chunk context) to an **ICD-10-CM** code and a
**SNOMED CT US** concept. One command, deterministic, runs locally on a 4 GB laptop GPU — no chart data leaves the machine.

- **Write-up (PDF):** [`docs/HiLabs_TermMapping_Writeup.pdf`](docs/HiLabs_TermMapping_Writeup.pdf)
- **Demo video:** _link to be added_
- **Trained models (Hugging Face, 1.7 GB):** https://huggingface.co/Dhisanth/hilabs-term-mapping — downloaded automatically on the first run
- **Full experiment log:** [`docs/WORKLOG.md`](docs/WORKLOG.md)

## Results (5-fold cross-validation on diagnosis / condition terms — the evaluation set's scope)

Every learned component (encoders, cross-encoder, LLM, kNN memory, priors, rankers) is re-trained inside each fold;
folds are grouped by source chunk, so no evaluated term is ever seen in training.

| | ICD-10-CM | SNOMED CT |
|---|---|---|
| **Exact match** | **71.1%** | **67.4%** |
| Codes never seen in training (long tail) | 50.1% | 57.4% |
| Rare or never-seen codes (≤ 2 training examples) | 61.5% | 63.3% |
| Partial credit: ICD 3-char category / SNOMED semantic tag | 81.6% | 84.4% |
| Exact or near-synonym (name similarity ≥ 90) | 74.4% | 73.9% |

**Headline (the lower of the two systems): 67.4%, set by SNOMED CT.** Label self-consistency on near-duplicate
training terms is 93.7% (ICD) / 84.6% (SNOMED) — a practical ceiling for exact match.

**Coverage / abstention.** Every row always gets a code (blanking a code can only lower exact match), plus a calibrated
`confidence`. Abstaining on the least-confident terms gives:

| Terms attempted (most confident first) | 100% | 90% | 70% | 50% | 30% |
|---|---|---|---|---|---|
| SNOMED CT (diagnosis) | 67.4% | 72.4% | 82.4% | 90.1% | 95.9% |
| ICD-10-CM | 71.1% | 76.3% | 85.4% | 92.3% | 96.4% |

---

## Run it on your test set

### 1. Requirements
- Python **3.11+** (developed on 3.13), Windows or Linux
- NVIDIA GPU with **≥ 4 GB VRAM** and CUDA (needed for the 4-bit LLM stage; without a GPU the engine falls back
  automatically to the no-LLM mode — see step 7)
- 16 GB RAM, **~13 GB free disk** (our models 1.7 GB, base models 3.2 GB — both from Hugging Face — and dictionary caches ~7 GB)
- Internet on the first run (downloads SapBERT and Qwen3-4B-Instruct-2507 4-bit from Hugging Face)

### 2. Install
```bash
git clone <this repo> && cd hilabs
python -m venv .venv
.venv\Scripts\activate                     # Linux/macOS: source .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
Optional: `copy .env.example .env` (Linux: `cp`) and set `HF_TOKEN` if Hugging Face rate-limits downloads.
No API keys are needed — nothing calls an external service.

### 3. Data files (not in the repo — proprietary)
Put these three files in the **repository root**, with exactly these names:

| File | Why it is needed at prediction time |
|---|---|
| `icd10cm_codes.xlsx` | ICD-10-CM answer space |
| `SNOMEDCT_US_codes_and_descriptions.xlsx` | SNOMED CT answer space |
| `hackathon_train.csv` | the engine's memory of coded examples (nearest-training-term retrieval and display names) |

### 4. Trained models — pulled automatically from Hugging Face
The weights are on the Hugging Face Hub: **https://huggingface.co/Dhisanth/hilabs-term-mapping**. Nothing to do:
if `models/` is missing, the first `run.py predict` downloads them into `models/` (~1.7 GB, once). To fetch them
manually instead: `hf download Dhisanth/hilabs-term-mapping --local-dir models`. (Another repo id can be set with
the environment variable `HILABS_MODELS_REPO`.)

### 5. Predict
```bash
python run.py predict --input test.csv --output outputs/predictions.csv --assume-subconcept DiagnosisCode
```
`--assume-subconcept DiagnosisCode` fills a missing or empty `subconcept` (the evaluation set is diagnoses/conditions
only; rows that already have a subconcept keep it).

**Input** — either layout works; column names are case-insensitive:

| Layout | Columns | Output |
|---|---|---|
| **Training schema** (one row per term × coding system) | `entity, chunk_text, coding_system[, code, display, subconcept]` | same file, row order kept, with `code` and `display` filled for each row's `coding_system`, plus `confidence, alt_code, alt_display, alt_confidence, ambiguous, needs_review` |
| Brief schema (one row per term) | `chunk_id, clinical_entity[, chunk_text]` | input columns + `icd10cm_code, snomed_code` with display / confidence / runner-up per system, and `needs_review` |

- In the training schema `code` / `display` may be blank or absent; any existing values are overwritten with predictions
  (keep your gold file separately). `coding_system` spelling variants (`ICD-10-CM`, `icd10`, `SNOMED CT`, …) are accepted.
- `chunk_text` is optional but improves accuracy (abbreviations, context).
- Rows of the same (entity, chunk) are mapped once and both systems inform each other.

**Run time** (RTX 3050 Laptop, 4 GB): ~1 min model load; the **first run** also builds the dictionary embedding and
lexical caches (~20–30 min, once, under `cache/`). Then ~0.85 s per term in batch (~45 min per 3,000 terms).

### 6. Check accuracy
Score the predictions against your gold file (the same test file with the true codes filled in):
```bash
python scripts/score.py --pred outputs/predictions.csv --gold gold.csv
```
Rows are matched by position (`run.py` keeps the input's row order); both layouts are supported (training schema:
`code` column per `coding_system`; brief schema: `icd10cm_code` / `snomed_code`). Example output (format only — numbers
shown are our cross-validation figures):
```
ICD-10-CM  n= 1200  exact match  71.1%   3-char category  81.6%   unflagged 70.0% of rows →  85.4% exact
SNOMED CT  n= 1200  exact match  67.4%   unflagged 70.0% of rows →  82.4% exact
HEADLINE (lower of the two): 67.4%
```
- **exact match** — the brief's primary metric, per coding system; **HEADLINE** — the lower of the two.
- **3-char category** — ICD partial credit (right category, wrong specificity).
- **unflagged** — accuracy if the rows flagged `needs_review` are abstained on (coverage / abstention).

### 7. Options and troubleshooting
| Situation | What to do |
|---|---|
| No CUDA GPU, or the LLM does not fit / fails | `--no-llm` (automatic when no CUDA GPU is found): skips the LLM stage, final decision = cross-encoder ranker; ~0.2 s/term, about 3–4 points lower accuracy (ICD ≈ 68%, SNOMED dx ≈ 64%) |
| CUDA out of memory | close other GPU programs; or `--no-llm` |
| "rankers were trained with the LLM stage but models/llm_adapter is missing" | incomplete download — delete `models/` and run again |
| Hugging Face download errors / rate limits | set `HF_TOKEN` in `.env`; `HF_HOME` moves the model cache |
| Review threshold | `--review-threshold 0.5` (default) sets `needs_review = confidence < threshold` |
| Dictionaries replaced by a new release | handled automatically (see *Version drift*) |

Hard-case walkthrough (abbreviation collision, OCR noise, specificity): `python demo.py --cases`.

---

## How it works

```
entity + chunk ─► 1 normalize ─► 2 candidate generation ─► 3 R1 LightGBM ─► 4 cross-encoder ─► 5 fine-tuned LLM ─► code
                  (query variants)  (~250–475 codes/term)    LambdaRank        (R2, top-30)       (R2+LLM, top-10)   + confidence
                                                             ~110 features                         ICD: + coupling (R3)
```
1. **Normalize** — unidecode, markup / `(hcc)` stripping, ~350 multi-sense clinical abbreviations (MS → 3 senses),
   spelling / OCR repair (glued words, stutter, edit distance) — all as *variants*; the original is always kept.
2. **Retrieve** — char n-gram TF-IDF, BM25, two SapBERT encoders fine-tuned on the coders' own mappings (round 2 with
   iterative hard negatives), base SapBERT, kNN over training terms; for SNOMED also top-k inside every semantic tag.
   Gold code is in the candidate pool for ~96% of terms.
3. **R1** — LightGBM LambdaRank over ~110 features per (term, candidate): retriever scores and ranks, fuzzy similarity,
   specificity (qualifiers a candidate adds or misses), laterality, ICD structure, SNOMED tag, priors, chunk context.
4. **R2** — SapBERT cross-encoder reads term + ±120 chars of chunk context with each of the top-30 candidates.
5. **R2+LLM** — Qwen3-4B-Instruct-2507, QLoRA-fine-tuned on Kaggle GPUs, scores R2's top-10 listwise via letter
   log-probabilities (one forward pass, deterministic); its scores are features for the final ranker.
   **R3** (ICD only) — the SNOMED prediction's top-5 informs the ICD decision.

All rankers are 5-seed bagged. Details, design decisions and error analysis: the PDF and `docs/WORKLOG.md`.

## Latency and footprint (RTX 3050 Laptop, 4 GB VRAM, 16 GB RAM; both systems per term)
| | one term at a time: median / p95 | batch (per term) | 3,000-term file |
|---|---|---|---|
| without LLM stage (`--no-llm`) | 2.7 s / 3.0 s | ~0.2 s | ~10 min |
| **final engine (with fine-tuned LLM)** | **3.7 s / 4.4 s** | **~0.85 s** | **~45 min** |

Batch mode shares dictionary search and GPU work across terms, so chart-volume files are much cheaper per term than
single lookups. Prediction runs in two GPU phases (retrieval / cross-encoder / rankers, then the 4-bit LLM alone, 2.8 GB
VRAM) so everything fits a 4 GB card; files can be split across GPUs. Deterministic: fixed seeds, LightGBM
deterministic mode, no sampling anywhere.

## Reproducing from scratch
**A. Released models (minutes):** steps 1–5 above (weights pulled from Hugging Face).
**B. Full retrain (deterministic, resumable):** `python run.py train` (~8 h on an RTX 3050): lexical out-of-fold
candidates → two rounds of per-fold encoder fine-tuning → cross-encoder → bagged rankers. The LLM stage uses the
committed out-of-fold LLM scores (`kaggle/llm_out/llm_scores_f*.csv` — row id, code, score; no chart text) and the
adapter from the cloud job (`kaggle/README_KAGGLE.md`: 5 folds + final, ~6.5 h on Kaggle 2× T4); without them the
engine trains and runs without the LLM stage. `python run.py all --input test.csv ...` trains if needed, then predicts.

## Version drift (new ICD-10-CM / SNOMED releases)
The engine fingerprints the dictionary files (`cache/DICT_VERSION.json`). Replace the two `.xlsx` files with a new
release and the next run rebuilds every dictionary-derived cache (parsed tables, lexical indexes, embeddings) — new
codes are retrievable at once. Run `python run.py train` afterwards to refresh learned conventions. Cache files are
written atomically, so an interrupted run never leaves a corrupt cache.

## Repository layout
| Path | Contents |
|---|---|
| `run.py` | CLI entry point (`train` / `predict` / `all`) |
| `src/pipeline.py` | training and inference orchestration (`Mapper`) |
| `src/dicts.py` | dictionary loading, code metadata, version fingerprint |
| `src/normalize.py`, `src/abbrev.py`, `src/spell.py`, `src/query.py` | normalization and query variants |
| `src/lexical.py`, `src/encoders.py`, `src/candidates.py`, `src/engine.py`, `src/features.py` | retrieval and features |
| `src/finetune.py` | contrastive fine-tuning of the bi-encoders (iterative hard negatives) |
| `src/crossenc.py`, `src/context.py`, `src/coupling.py` | cross-encoder, chunk-context features, ICD↔SNOMED coupling |
| `src/llm_rerank.py` | fine-tuned LLM reranker (local inference) |
| `src/ranker.py`, `src/cv.py`, `src/evaluate.py` | bagged LambdaRank, grouped folds, evaluation |
| `src/llm.py` | zero-shot LLM judge (experiment only, not in the final engine) |
| `kaggle/` | cloud training job for the LLM adapter + out-of-fold LLM scores |
| `scripts/score.py` | score predictions against a gold file |
| `scripts/profile_*.py`, `scripts/coverage.py` | data and dictionary profiling |
| `experiments/`, `results/` | every experiment (numbered as in the work log) and its metrics |
| `docs/` | write-up PDF (+ `build_pdf.py`), work log, related research, demo script |
| `demo.py` | interactive / hard-case demo |

## External resources
| Resource | Use | Licence |
|---|---|---|
| SapBERT `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` | bi-encoders (base + fine-tuned), cross-encoder | Apache-2.0 |
| Qwen3-4B-Instruct-2507 (`unsloth/Qwen3-4B-Instruct-2507-bnb-4bit`) + QLoRA (PEFT, bitsandbytes) | fine-tuned LLM reranker | Apache-2.0 |
| BioLORD-2023, gte-base, BGE-base, e5-base, MedEmbed, MedCPT | benchmarked, not in the final engine | see model cards |
| ICD-10-CM and SNOMED CT US dictionaries | supplied by the challenge (answer space) | challenge terms |
| Clinical abbreviation list (`src/abbrev.py`) | hand-curated from general clinical documentation conventions | — |
| PyTorch, sentence-transformers, transformers, LightGBM, scikit-learn, RapidFuzz | libraries | permissive |
