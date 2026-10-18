# HiLabs Clinical Term Mapping Engine

Maps a clinical entity extracted from a medical chart (plus its chunk context) to an **ICD-10-CM** code
and a **SNOMED CT US** concept. The engine runs end to end from one command, is deterministic (fixed seeds),
and runs on a laptop GPU (developed on an RTX 3050, 4 GB).

> Results, design rationale and error analysis: see [`docs/WORKLOG.md`](docs/WORKLOG.md) (full experiment log)
> and the write-up PDF. Demo video: _link to be added_.

## Results (5-fold cross-validation, grouped by chunk; every learned component re-trained per fold)

| Exact match | ICD-10-CM | SNOMED CT |
|---|---|---|
| All terms | **71.1%** | **58.2%** |
| Diagnosis / condition terms (brief's scope) | 71.1% | **67.4%** |
| Codes never seen in training (long tail) | 50.1% | 47.0% |
| Rare or never-seen codes (≤ 2 training examples) | 61.5% | 53.7% |
| Partial credit: ICD 3-char category / SNOMED semantic tag | 81.6% | 76.9% |
| Exact or near-synonym (name similarity ≥ 90) | 74.4% | 65.6% |
| Most-confident 30% of terms (abstain on the rest) | 96.4% | 90.8% |

The shipped rankers are additionally 5-seed bagged (+0.2 ICD / +0.5 SNOMED, measured separately).
**The evaluation set contains only diagnosis/condition terms (confirmed by the organisers)** → the relevant expectation
is **ICD ≈ 71%, SNOMED ≈ 67%**, and the engine must be run with `--assume-subconcept DiagnosisCode` (the rankers and the
LLM prompt use the entity type; the flag fills it when the input has no `subconcept` column).
Label self-consistency on near-duplicate training terms is 93.7% (ICD) / 84.6% (SNOMED) — a practical ceiling for
exact match. Full write-up: [`docs/HiLabs_TermMapping_Writeup.pdf`](docs/HiLabs_TermMapping_Writeup.pdf).

## Quick start for evaluators
1. **Environment** — Python 3.10+, NVIDIA GPU ≥ 4 GB (CUDA), 16 GB RAM, ~15 GB free disk, internet on first run:
   ```bash
   python -m venv .venv && .venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   pip install -r requirements.txt
   ```
2. **Data** (not in the repo — proprietary; place in the repository root): `hackathon_train.csv` (also used at prediction
   time as the engine's memory of coded examples), `icd10cm_codes.xlsx`, `SNOMEDCT_US_codes_and_descriptions.xlsx`.
3. **Trained models** — download `models.zip` (_link to be added_) and unzip it in the repository root → `models/`.
4. **Predict** (the evaluation set is diagnoses/conditions only):
   ```bash
   python run.py predict --input eval.csv --output outputs/predictions.csv --assume-subconcept DiagnosisCode
   ```
   First run: downloads SapBERT + Qwen3-4B-4bit from Hugging Face (~3.2 GB) and builds dictionary embedding caches
   (~20–30 min, once). Then ~0.85 s per term. Output keeps the input columns and adds `icd10cm_code`, `snomed_code`
   (+ display, confidence, runner-up, review flags).
5. Optional: `python demo.py --cases` (hard-case walkthrough) · `python run.py train` (rebuild all models, ~8 h).

## How it works

```
entity + chunk ──► normalize ──► candidate generation (~250–475 codes/term) ──► LightGBM LambdaRank ──► code + confidence
                   │               ├─ char n-gram TF-IDF over dictionary descriptions
                   │               ├─ BM25 over dictionary descriptions
                   │               ├─ fine-tuned SapBERT (dense) over descriptions   ┐ SNOMED: also top-k inside
                   │               ├─ base SapBERT (dense) over descriptions         ┘ each semantic tag
                   │               ├─ fine-tuned + base SapBERT kNN over training terms (coder conventions)
                   │               └─ char kNN over training terms
                   └─ abbreviation expansion (multi-sense), spelling / OCR repair, glued-word splitting
```

~100 features per (term, candidate): retriever scores and ranks, fuzzy string similarity, specificity
(extra / missing qualifier words), laterality agreement, ICD code structure (chapter, leaf, "unspecified"),
SNOMED semantic tag, code/category priors, P(tag | subconcept), and chunk-context features (concept-type classifier
over term + chunk, chunk-style flags). Then a cross-encoder reads term + chunk context with the top-30 candidates
(R2), a **fine-tuned LLM reranker** (QLoRA Qwen3-4B-Instruct, trained on cloud GPUs — see `kaggle/`) scores the R2
top-10 (R2+LLM), and ICD↔SNOMED coupling refines ICD (R3); all rankers are 5-seed bagged LightGBM LambdaRank models.

**LLM component:** the adapter (`models/llm_adapter`, from `kaggle/llm_out/adapter_final.zip`) runs locally in 4-bit
(2.8 GB VRAM, ~0.4 s/term). Training it reproducibly: `kaggle/README_KAGGLE.md` (5-fold + final, ~6.5 h on 2× T4).
`run.py train` uses the out-of-fold LLM scores in `kaggle/llm_out/` when present; without them the engine trains and
runs without the LLM stage (ICD 68.9% / SNOMED 53.8%).

Every number reported is **5-fold cross-validated, grouped by source chunk**, with the encoder fine-tuned
separately inside each fold, so no evaluation term is ever seen during training.

## Setup

Requirements: Python 3.10+ (developed on 3.13), an NVIDIA GPU with ≥4 GB VRAM recommended (CPU works, slower),
~8 GB free disk for model weights and embedding caches.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or the CPU build
pip install -r requirements.txt
```

Place the data files (not committed — proprietary) in the repository root:

| File | Source |
|---|---|
| `hackathon_train.csv` | challenge dataset link |
| `icd10cm_codes.xlsx` | challenge ICD-10-CM dictionary |
| `SNOMEDCT_US_codes_and_descriptions.xlsx` | challenge SNOMED CT dictionary |

### Environment variables
Copy `.env.example` to `.env`. Nothing is required; optional: `HF_TOKEN` (Hugging Face rate limits),
`HF_HOME` (model cache location). No external API is called: all models run locally, so no chart data
leaves the machine.

## Run

```bash
# one command: trains everything if needed (resumable), then maps the input file
# (evaluation set = diagnoses/conditions only → always pass --assume-subconcept DiagnosisCode)
python run.py all --input eval.csv --output outputs/predictions.csv --assume-subconcept DiagnosisCode

# or separately
python run.py train
python run.py predict --input eval.csv --output outputs/predictions.csv --assume-subconcept DiagnosisCode
```

Accepted input schemas:
- `chunk_id, clinical_entity[, chunk_text]` → output adds `icd10cm_code, snomed_code` (+ display names and confidence) for every row
- `entity, chunk_text, coding_system[, subconcept]` (training format) → output adds `code, display, confidence` for the requested system

Each prediction carries:
- `confidence` — calibrated probability of the chosen code (conf > 0.9 is correct 91% ICD / 96% SNOMED);
- `alt_code`, `alt_display`, `alt_confidence` — the runner-up candidate;
- `ambiguous` — the top two are a close call or near-synonyms (for review or set-based scoring);
- `needs_review` — confidence below `--review-threshold` (default 0.5).

Live demo of hard cases (abbreviation collision, OCR noise, specificity): `python demo.py --cases`.

## Reproducing the submitted predictions
**A. With the released models (minutes):** download `models.zip` (_link to be added_ — 1.7 GB: bi-encoders,
cross-encoder, LLM adapter, bagged rankers, context models; too large for GitHub) and unzip it **in the repository
root** (the archive contains the `models/` folder), then
```bash
python run.py predict --input eval.csv --output outputs/predictions.csv --assume-subconcept DiagnosisCode
```
**B. From scratch (deterministic, resumable):** `python run.py train` rebuilds every model locally (~8 h on an RTX 3050:
lexical OOF → two rounds of per-fold encoder fine-tuning → cross-encoder → rankers). The LLM stage uses the committed
out-of-fold scores in `kaggle/llm_out/llm_scores_f*.csv` and the adapter from the cloud job (`kaggle/README_KAGGLE.md`,
~6.5 h on Kaggle 2× T4); without them the engine trains and runs without the LLM stage.

## Latency and footprint (RTX 3050 Laptop, 4 GB VRAM, 16 GB RAM)
| | per term (after ~1 min one-off model load) | 3,000-term file |
|---|---|---|
| without LLM stage | ~0.2 s | ~10 min |
| **final engine (with fine-tuned LLM)** | **~0.85 s** | **~45 min** |
Prediction runs in two GPU phases (retrieval / cross-encoder / rankers, then the 4-bit LLM alone) so everything fits a
4 GB card; on a datacenter GPU the LLM step batches far larger. Disk: ~2 GB models + ~6 GB dictionary embedding caches
(rebuilt automatically on first run). Deterministic: fixed seeds, LightGBM deterministic mode, no sampling anywhere
(the LLM is read via next-token letter log-probabilities).

## Version drift (new ICD-10-CM / SNOMED releases)
The engine fingerprints the dictionary files (`cache/DICT_VERSION.json`). Replace `icd10cm_codes.xlsx` /
`SNOMEDCT_US_codes_and_descriptions.xlsx` with a new release and the next run automatically rebuilds every
dictionary-derived cache (parsed tables, lexical indexes, description embeddings) — new codes are retrievable at once.
Run `python run.py train` afterwards to refresh the learned conventions; codes retired from the new release are
dropped from the training memory automatically.

## Repository layout

| Path | Contents |
|---|---|
| `run.py` | CLI entry point |
| `src/dicts.py` | dictionary loading, code metadata (ICD structure, SNOMED semantic tags) |
| `src/normalize.py`, `src/abbrev.py`, `src/spell.py`, `src/query.py` | normalization and query variants |
| `src/lexical.py`, `src/encoders.py`, `src/candidates.py` | retrievers |
| `src/features.py`, `src/engine.py` | candidate table + features |
| `src/ranker.py`, `src/evaluate.py` | LightGBM LambdaRank reranker, CV evaluation |
| `src/finetune.py` | contrastive fine-tuning of the bi-encoder |
| `src/crossenc.py`, `src/coupling.py` | cross-encoder reranker, ICD↔SNOMED coupling |
| `src/llm.py` | local LLM adjudicator (tested, not in the final pipeline) |
| `src/pipeline.py` | training and inference orchestration |
| `experiments/` | every experiment, numbered as in the work log |
| `results/` | experiment outputs (metrics only) |
| `docs/WORKLOG.md` | chronological log of all experiments, numbers and decisions |
| `docs/RESEARCH.md` | related work and what we took from it |
| `docs/HiLabs_TermMapping_Writeup.pdf` (`docs/build_pdf.py`) | technical write-up |
| `demo.py` | interactive / hard-case demo |

## External resources

| Resource | Use | Licence |
|---|---|---|
| SapBERT `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` | bi-encoder (base + fine-tuned) | Apache-2.0 |
| BioLORD-2023, gte-base, BGE-base, e5-base, MedEmbed, MedCPT | benchmarked, not in final engine | see model cards |
| Qwen3-4B-Instruct-2507 (`unsloth/Qwen3-4B-Instruct-2507-bnb-4bit`) + QLoRA (PEFT, bitsandbytes) | fine-tuned LLM reranker | Apache-2.0 |
| ICD-10-CM and SNOMED CT US dictionaries | supplied by the challenge (answer space) | challenge terms |
| Clinical abbreviation list (`src/abbrev.py`) | hand-curated from general clinical documentation conventions | — |
| Python libraries: PyTorch, sentence-transformers, transformers, LightGBM, scikit-learn, RapidFuzz | — | permissive |

## Reproducibility
Fixed seeds everywhere (`seed=7`, LightGBM `deterministic=True`, fold assignment `seed=13`). Expensive
artifacts (dictionary embeddings, per-fold candidate tables, models) are cached under `cache/` and `models/`;
deleting them forces a full rebuild.
