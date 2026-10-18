# Demo video script (2–3 min screen recording)

Brief requirements: architecture in 30–45 s; live mapping end to end incl. hard cases (abbreviation collision,
OCR-mangled term, specificity judgement); show error analysis / confidence output, not just the happy path.

**Before recording:** open `docs/HiLabs_TermMapping_Writeup.pdf` (page 1–2), a terminal in the project folder with the
venv active, and `results/error_analysis_*_final.md`. Run `python demo.py --cases` once beforehand so models and
dictionary caches are warm (first load takes ~1–2 min).

---

## 0:00 – 0:40 · Problem and architecture (show the PDF architecture diagram)
> "CodEx extracts clinical terms from charts. Our engine maps each term, with its chunk context, to one ICD-10-CM code
> and one SNOMED CT concept.
> First we normalize: expand clinical abbreviations — keeping every meaning of ambiguous ones — and repair OCR and typos.
> Then six retrievers propose about 250 to 475 candidate codes: character n-grams, BM25, two SapBERT encoders we
> fine-tuned on the coders' own mappings, base SapBERT, and nearest training terms, which capture local coding
> conventions. A LightGBM ranker scores every candidate on about 100 features; a cross-encoder then reads the term and
> its chart context with each of the top 30; a fine-tuned LLM (trained on cloud GPUs, running locally) judges the
> top 10; and ICD and SNOMED predictions inform each other. Everything is 5-fold
> cross-validated and runs locally on a laptop GPU."

## 0:40 – 2:00 · Live hard cases (terminal: `python demo.py --cases`)
Scroll through the output; pause on each and say one line:
1. **`HTN`** → I10 Essential hypertension, high confidence — *"abbreviation expanded."*
2. **`MS` in a neurology note** → SNOMED *Multiple sclerosis* (0.91), ICD G35. **`MS` in an echo report** → SNOMED
   **Mitral valve stenosis** (0.77) — *"abbreviation collision resolved from the chart context: every sense is retrieved,
   and the fine-tuned LLM plus the cross-encoder read the context ('mean gradient', 'LA enlarged')."* Be honest: *"ICD
   still says multiple sclerosis here — it follows the training prior; that's in our error analysis."*
3. **`chronickidney disease stag 3`** → N18.30 / CKD stage 3 — *"OCR glued words and a truncated word repaired."*
4. **`cardiomyopahty`** → I42.9 — *"typo repaired by spelling correction against the dictionary vocabulary."*
5. **`diabetes`** → E11.9 (unspecified) vs **`t2dm with diabetic neuropathy`** → E11.40 — *"specificity calibration:
   unspecified when the text says nothing more, specific when it does."*
6. **`pulmonary hypertension`** → I27.20, not I10 — *"the nearest string is the wrong concept; the ranker avoids it."*
7. **`left knee pain`** → M25.562 — *"laterality."*  **`h/o herpes zoster`** → Z86.19 — *"history mention."*

Point at the per-row output: code, confidence, the runner-up candidates with scores and which retrievers found them.

**Live typing (interactive mode, `.venv\Scripts\python demo.py`) — tested inputs:**
| type as term | type as context | expected result |
|---|---|---|
| `acute on chronic systolic chf` | `admitted with acute on chronic systolic chf exacerbation` | ICD **I50.23**, SNOMED *Acute on chronic systolic heart failure* (specificity) |
| `PID` | `gyn: lower abdominal pain, cervical motion tenderness, treated for PID` | ICD **N73.9**, SNOMED *Female pelvic inflammatory disease* (abbreviation) |
| `htn uncontroled` | `a/p: htn uncontroled, increase lisinopril` | ICD **I10**; SNOMED flagged ⚠ NEEDS REVIEW (typo + review flag) |
| `malignant hypertension` | `bp 230/130 with papilledema, malignant hypertension` | ICD **I10** (ICD-10-CM codes it to I10), SNOMED *Malignant hypertension* (systems differ correctly) |
Each live term takes ~30–60 s (models + LLM load per run) — cut the wait in editing.
Avoid `RA` with "echo: dilated RA" (known failure: still maps to rheumatoid arthritis) unless showing a failure on purpose.

## 2:00 – 2:50 · Error analysis and confidence (show the PDF error-analysis section / results md)
> "Exact-match accuracy in 5-fold cross-validation is about 71% for ICD and 67% for SNOMED on diagnosis terms — the
> evaluation scope. Never-seen codes still get 47 to 50%, which shows the system generalizes rather than memorizes.
> The funnel shows the right answer is in our top 10 for 81 to 90% of terms; the loss is the final pick among close
> candidates. Much of that is label ambiguity: near-identical training terms share the same gold code only 85% of the
> time for SNOMED, and 18% of our SNOMED errors are near-synonyms of the gold concept.
> Confidence is calibrated: on its most confident 30% the engine is about 96% accurate for ICD and 91% for SNOMED, so low-confidence and
> ambiguous rows are flagged for human review, with a runner-up code."

## 2:50 – 3:00 · Close
> "One command reproduces everything: `python run.py all --input eval.csv --output predictions.csv`. Thank you."

---
**Upload:** save as MP4, upload to an internal drive, put the link in `README.md` (top) and in `docs/build_pdf.py`
(`RESULTS["demo_link"]`), then rebuild the PDF.
