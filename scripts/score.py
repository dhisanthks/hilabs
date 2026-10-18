"""Score predictions against a gold file (exact match per coding system; headline = the lower of the two).

  python scripts/score.py --pred outputs/predictions.csv --gold test_with_gold.csv

Rows are aligned by position (run.py keeps the input's row order). Supported layouts:
  training schema   — one row per (term, coding_system); gold and predicted codes in the `code` column
  brief schema      — one row per term; `icd10cm_code` / `snomed_code` columns in both files
Also reports ICD 3-character-category accuracy and accuracy when abstaining on rows flagged `needs_review`.
"""
import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    a = ap.parse_args()
    p = pd.read_csv(a.pred, dtype=str, keep_default_na=False)
    g = pd.read_csv(a.gold, dtype=str, keep_default_na=False)
    if len(p) != len(g):
        raise SystemExit(f"row count differs: pred {len(p)} vs gold {len(g)} — rows are matched by position")
    res = {}
    if "coding_system" in g.columns:
        key = g["coding_system"].str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
        systems = {"ICD-10-CM": key.str.startswith("ICD"), "SNOMED CT": key.str.startswith("SNOMED")}
        for name, m in systems.items():
            if m.any():
                res[name] = (p.loc[m, "code"].str.strip(), g.loc[m, "code"].str.strip(), m)
    else:
        for name, col in (("ICD-10-CM", "icd10cm_code"), ("SNOMED CT", "snomed_code")):
            if col in g.columns:
                res[name] = (p[col].str.strip(), g[col].str.strip(), pd.Series(True, index=g.index))
    acc = {}
    for name, (pc, gc, m) in res.items():
        ok = pc == gc
        acc[name] = ok.mean()
        line = f"{name:10s} n={len(gc):5d}  exact match {100 * ok.mean():5.1f}%"
        if name == "ICD-10-CM":
            cat = pc.str.replace(".", "", regex=False).str[:3] == gc.str.replace(".", "", regex=False).str[:3]
            line += f"   3-char category {100 * cat.mean():5.1f}%"
        if "needs_review" in p.columns:
            keep = p.loc[m, "needs_review"].str.lower() != "true"
            if keep.any():
                line += f"   unflagged {100 * keep.mean():4.1f}% of rows → {100 * ok[keep].mean():5.1f}% exact"
        print(line)
    if len(acc) == 2:
        print(f"HEADLINE (lower of the two): {100 * min(acc.values()):.1f}%")


if __name__ == "__main__":
    main()
