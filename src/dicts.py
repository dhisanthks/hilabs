"""Load and index the ICD-10-CM and SNOMED CT US dictionaries.

The dictionaries define the legal answer space: every predicted code must come from here.
Raw xlsx files are parsed once and cached as parquet under cache/.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
ICD_XLSX = os.path.join(ROOT, "icd10cm_codes.xlsx")
SNOMED_XLSX = os.path.join(ROOT, "SNOMEDCT_US_codes_and_descriptions.xlsx")

SYSTEMS = ("ICD10CM", "SNOMEDCT_US")
_TAG_RE = re.compile(r"\(([^()]+)\)\s*$")


_DICT_DERIVED = ("icd.parquet", "snomed.parquet", "ICD10CM_system.pkl", "SNOMEDCT_US_system.pkl", "vocab.pkl",
                 "lex_char_ICD10CM.pkl", "lex_char_SNOMEDCT_US.pkl", "lex_bm25_ICD10CM.pkl", "lex_bm25_SNOMEDCT_US.pkl")
_version_checked = False


def check_dictionary_version():
    """Version drift: fingerprint the supplied dictionary files (size + mtime). If they changed since the caches were
    built (e.g. a new ICD-10-CM October release or SNOMED biannual release is dropped in), every dictionary-derived
    cache (parsed tables, lexical indexes, vocabulary, description embeddings) is invalidated and rebuilt on demand."""
    global _version_checked
    if _version_checked:
        return
    _version_checked = True
    import glob
    import json
    os.makedirs(CACHE, exist_ok=True)
    sig = {os.path.basename(f): [os.path.getsize(f), int(os.path.getmtime(f))]
           for f in (ICD_XLSX, SNOMED_XLSX) if os.path.exists(f)}
    stamp = os.path.join(CACHE, "DICT_VERSION.json")
    old = json.load(open(stamp)) if os.path.exists(stamp) else None
    if old is not None and old != sig:
        stale = [os.path.join(CACHE, f) for f in _DICT_DERIVED] + glob.glob(os.path.join(CACHE, "emb_*.npy"))
        for f in stale:
            if os.path.exists(f):
                os.remove(f)
        print(f"[dicts] dictionary files changed {old} -> {sig}: rebuilt dictionary-derived caches", flush=True)
    json.dump(sig, open(stamp, "w"))


def _load_raw(system: str) -> pd.DataFrame:
    os.makedirs(CACHE, exist_ok=True)
    pq = os.path.join(CACHE, "icd.parquet" if system == "ICD10CM" else "snomed.parquet")
    if os.path.exists(pq):
        df = pd.read_parquet(pq)
    elif system == "ICD10CM":
        df = pd.read_excel(ICD_XLSX, dtype=str)
        df.columns = ["code", "description"]
        df.to_parquet(pq)
    else:
        df = pd.read_excel(SNOMED_XLSX, sheet_name="SNOMED codes", dtype=str)
        df.columns = ["code", "description", "order"]
        df.to_parquet(pq)
    df = df.dropna(subset=["code", "description"]).copy()
    df["code"] = df["code"].str.strip()
    df["description"] = df["description"].str.strip()
    return df


@dataclass
class CodeSystem:
    """One coding system: description rows + per-code metadata."""

    name: str
    desc: pd.DataFrame  # one row per (code, description): code, description, code_idx
    codes: np.ndarray  # distinct codes, index = code_idx
    code_to_idx: dict = field(default_factory=dict)
    name_of: np.ndarray = None  # preferred display name per code
    tag_of: np.ndarray = None  # SNOMED semantic tag per code ('' for ICD)
    meta: pd.DataFrame = None  # per-code engineered metadata

    def __len__(self):
        return len(self.codes)


def _icd_meta(codes: np.ndarray, names: np.ndarray) -> pd.DataFrame:
    c = pd.Series(codes)
    flat = c.str.replace(".", "", regex=False)
    is_range = c.str.contains("-")
    codeset = set(codes)
    # a code is a leaf if no other code extends it (billable-ish)
    prefixes = set()
    for x in codes:
        if "-" in x:
            continue
        for k in range(3, len(x)):
            prefixes.add(x[:k])
    has_child = c.isin(prefixes) | c.map(lambda x: (x + ".") in prefixes if "." not in x else False)
    lname = pd.Series(names).str.lower()
    return pd.DataFrame(
        {
            "code": codes,
            "chapter": flat.str[0],
            "category": flat.str[:3],
            "length": flat.str.len(),
            "is_range": is_range,
            "is_leaf": ~has_child & ~is_range,
            "unspecified": lname.str.contains("unspecified").values,
            "other": lname.str.match(r"^other\b").values,
            "left": lname.str.contains(r"\bleft\b").values,
            "right": lname.str.contains(r"\bright\b").values,
            "bilateral": lname.str.contains(r"\bbilateral\b").values,
            "initial": c.str.match(r"^[ST].*A$").values,
        }
    )


def load_system(system: str) -> CodeSystem:
    check_dictionary_version()
    pkl = os.path.join(CACHE, f"{system}_system.pkl")
    if os.path.exists(pkl):
        return pd.read_pickle(pkl)
    raw = _load_raw(system)
    if system == "SNOMEDCT_US":
        raw["order"] = pd.to_numeric(raw["order"], errors="coerce").fillna(99)
        raw = raw.sort_values(["code", "order"], kind="stable")
    raw = raw.drop_duplicates(["code", "description"]).reset_index(drop=True)
    codes = raw["code"].drop_duplicates().to_numpy()
    code_to_idx = {c: i for i, c in enumerate(codes)}
    raw["code_idx"] = raw["code"].map(code_to_idx).astype(np.int64)

    if system == "SNOMEDCT_US":
        # preferred name: first description without a trailing semantic tag; tag from the FSN
        tags = raw["description"].str.extract(_TAG_RE)[0]
        tag_of = tags.groupby(raw["code_idx"]).first().reindex(range(len(codes))).fillna("").to_numpy()
        no_tag = raw[tags.isna()]
        name = no_tag.groupby("code_idx")["description"].first()
        fallback = raw.groupby("code_idx")["description"].first().str.replace(_TAG_RE, "", regex=True)
        name_of = name.reindex(range(len(codes))).fillna(fallback).to_numpy()
        meta = pd.DataFrame({"code": codes, "tag": tag_of})
    else:
        tag_of = np.array([""] * len(codes), dtype=object)
        # prefer the longest description (usually the full official title) as display
        longest = raw.assign(L=raw["description"].str.len()).sort_values("L", ascending=False)
        name_of = longest.groupby("code_idx")["description"].first().reindex(range(len(codes))).to_numpy()
        meta = _icd_meta(codes, name_of)

    cs = CodeSystem(system, raw, codes, code_to_idx, name_of, tag_of, meta)
    pd.to_pickle(cs, pkl)
    return cs


def load_train(path: str | None = None) -> pd.DataFrame:
    path = path or os.path.join(ROOT, "hackathon_train.csv")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["chunk_text"] = df["chunk_text"].fillna("")
    # group id: the (entity, chunk) mention — both coding-system rows of a mention share it
    df["mention_id"] = df.groupby(["entity", "chunk_text"], sort=False).ngroup()
    df["doc_id"] = df.groupby("chunk_text", sort=False).ngroup()
    return df
