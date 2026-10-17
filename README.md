# HiLabs Clinical Term Mapping Engine

Maps a clinical entity extracted from a medical chart (plus its chunk context) to an **ICD-10-CM** code
and a **SNOMED CT US** concept. The engine runs end to end from one command, is deterministic (fixed seeds),
and runs on a laptop GPU (developed on an RTX 3050, 4 GB).

> Results, design rationale and error analysis: see [`docs/WORKLOG.md`](docs/WORKLOG.md) (full experiment log)
> and the write-up PDF. Demo video: _link to be added_.

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

~90 features per (term, candidate): retriever scores and ranks, fuzzy string similarity, specificity
(extra / missing qualifier words), laterality agreement, ICD code structure (chapter, leaf, "unspecified"),
SNOMED semantic tag, code/category priors and P(tag | subconcept) learned from training data.

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
python run.py all --input eval.csv --output outputs/predictions.csv

# or separately
python run.py train
python run.py predict --input eval.csv --output outputs/predictions.csv
```

Accepted input schemas:
- `chunk_id, clinical_entity[, chunk_text]` → output adds `icd10cm_code, snomed_code` (+ display names and confidence) for every row
- `entity, chunk_text, coding_system[, subconcept]` (training format) → output adds `code, display, confidence` for the requested system

Each prediction carries a `confidence` (softmax probability of the winning candidate) and a
`needs_review` flag (confidence below `--review-threshold`, default 0.5) for abstention / human review.

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
| `src/crossenc.py`, `src/coupling.py` | cross-encoder reranker, ICD↔SNOMED coupling (experimental stages) |
| `src/pipeline.py` | training and inference orchestration |
| `experiments/` | every experiment, numbered as in the work log |
| `results/` | experiment outputs (metrics only) |
| `docs/WORKLOG.md` | chronological log of all experiments, numbers and decisions |

## External resources

| Resource | Use | Licence |
|---|---|---|
| SapBERT `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` | bi-encoder (base + fine-tuned) | Apache-2.0 |
| BioLORD-2023, gte-base, BGE-base, e5-base, MedEmbed, MedCPT | benchmarked, not in final engine | see model cards |
| ICD-10-CM and SNOMED CT US dictionaries | supplied by the challenge (answer space) | challenge terms |
| Clinical abbreviation list (`src/abbrev.py`) | hand-curated from general clinical documentation conventions | — |
| Python libraries: PyTorch, sentence-transformers, transformers, LightGBM, scikit-learn, RapidFuzz | — | permissive |

## Reproducibility
Fixed seeds everywhere (`seed=7`, LightGBM `deterministic=True`, fold assignment `seed=13`). Expensive
artifacts (dictionary embeddings, per-fold candidate tables, models) are cached under `cache/` and `models/`;
deleting them forces a full rebuild.
