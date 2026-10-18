"""Interactive demo: map a clinical term (+ optional chunk context) to ICD-10-CM and SNOMED CT, and show
how the engine got there (normalized query variants, top candidates with scores, confidence, review flag).

  python demo.py                       # interactive prompt
  python demo.py --cases               # run the built-in hard cases (abbreviation collision, OCR noise, specificity)
"""
import argparse
import os
import sys
import warnings

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.dicts import SYSTEMS
from src.query import variants

HARD_CASES = [
    ("HTN", "pmh: htn, hld, t2dm on metformin"),                                   # abbreviation
    ("MS", "neuro: relapsing remitting MS on ocrelizumab, MRI with new lesions"),  # abbreviation collision
    ("MS", "echo: moderate MS with mean gradient 8 mmHg, LA enlarged"),            # ... other sense
    ("chronickidney disease stag 3", "assessment: chronickidney disease stag 3, creatinine 1.6"),  # OCR noise
    ("cardiomyopahty", "hx of ischemic cardiomyopahty, ef 30%"),                   # typo
    ("diabetes", "a/p: diabetes - continue current regimen"),                      # specificity: unspecified
    ("t2dm with diabetic neuropathy", "feet: decreased monofilament, t2dm with diabetic neuropathy"),  # specific
    ("left knee pain", "c/o left knee pain x 2 weeks after fall"),                 # laterality
    ("h/o herpes zoster", "past history: h/o herpes zoster 2019"),                 # history
    ("pulmonary hypertension", "echo: rvsp 55, pulmonary hypertension"),           # near-string trap vs I10
]


def show_batch(m, cases, k=5, subconcept="DiagnosisCode"):
    """Map all (entity, chunk) cases in ONE engine run (models and the LLM load once), then print each."""
    # the evaluation set is diagnoses/conditions only → same entity type as `run.py --assume-subconcept DiagnosisCode`
    q = pd.DataFrame({"entity": [e for e, _ in cases], "chunk_text": [c for _, c in cases],
                      "subconcept": [subconcept] * len(cases)})
    rows = np.arange(len(q))
    res = m.run(q, {s: rows for s in SYSTEMS})
    for i, (entity, chunk) in enumerate(cases):
        print("\n" + "=" * 100)
        print(f"TERM: {entity!r}")
        if chunk:
            print(f"CONTEXT: {chunk[:120]!r}")
        print(f"normalized query variants: {variants(entity)}")
        for s in SYSTEMS:
            r = res[s].loc[i]
            flag = "  ⚠ NEEDS REVIEW" if r.confidence < 0.5 else ""
            print(f"\n  {s:12s} → {r.code:18s} {r.display}   (confidence {r.confidence:.2f}){flag}")
            c = m.last_cands[s]
            c = c[c.qid == i].sort_values("prob", ascending=False).head(k)
            cs = m.engines[s].cs
            for _, row in c.iterrows():
                src = [n for n, col in (("char", "char_rank"), ("bm25", "bm25_rank"), ("dense-ft", "dense_ft_rank"),
                                       ("dense-base", "dense_base_rank"), ("kNN-train", "knn_ft_rank"))
                       if col in row and row[col] < 50]
                ce_txt = f"ce={row.ce:6.2f}" if row.ce == row.ce else "ce=  —  "
                llm_txt = (f"  llm={np.exp(row.llm_lp):4.2f}" if "llm_lp" in row and row.llm_lp == row.llm_lp else "")
                print(f"      {row.prob:5.2f}  {cs.codes[int(row.code_idx)]:18s} {str(cs.name_of[int(row.code_idx)])[:58]:58s}"
                      f"  {ce_txt}{llm_txt}  via {','.join(src)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", action="store_true")
    a = ap.parse_args()
    from src.pipeline import Mapper
    print("loading models ...")
    m = Mapper()
    if a.cases:
        show_batch(m, HARD_CASES)
        return
    while True:
        try:
            e = input("\nterm (empty to quit)> ").strip()
            if not e:
                break
            c = input("chunk context (optional)> ").strip()
            show_batch(m, [(e, c)])
        except (EOFError, KeyboardInterrupt):
            break


if __name__ == "__main__":
    sys.exit(main())
