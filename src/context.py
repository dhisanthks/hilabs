"""Chunk-context features for the rankers (Exp18): concept-type probability from term + chunk (+ subconcept) and
chunk-style flags. The rankers otherwise see no chart context.

  p_type_ctx / type_ctx_top – P(candidate's type | term char-ngrams, chunk words[, subconcept]); type = SNOMED semantic
                              tag or ICD chapter (logistic regression, out-of-fold for training data)
  ch_*                      – chunk style: table markup, digit density, length, lab / exercise / history / assessment /
                              device / social / procedure / vitals vocabulary counts
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from .normalize import clean

VOCAB = {
    "ch_lab": r"\b(?:serum|plasma|blood|urine|result|reference|range|flag|abnormal|high|low|mg/dl|mmol|g/dl|u/l|meq|cbc|bmp|cmp|panel|specimen|lab)\b",
    "ch_exercise": r"\b(?:reps?|sets?|x\s?\d+|sec|seconds|minutes|theraband|exercise|stretch|strength|rom|gait|ambulat\w*|transfer\w*|pt|ot)\b",
    "ch_history": r"\b(?:history|hx|pmh|psh|past|prior|surgical|family|fhx)\b",
    "ch_assess": r"\b(?:assessment|plan|diagnos\w*|impression|problem|a/p|icd|chronic|acute)\b",
    "ch_device": r"\b(?:device|equipment|dme|walker|cane|wheelchair|pump|monitor|meter|lancet|pen|needle|cpap|oxygen)\b",
    "ch_social": r"\b(?:lives|living|spouse|alone|home|stairs|tobacco|alcohol|smok\w*|drinks|employ\w*|education)\b",
    "ch_procedure": r"\b(?:procedure|performed|surgery|operation|ultrasound|ct|mri|x-?ray|echo|ekg|injection|biopsy|scan)\b",
    "ch_vitals": r"\b(?:bp|pulse|temp|spo2|resp|weight|height|bmi|hr)\b",
}


def _chunk(c):
    return re.sub(r"header_\d+:|::|<br>", " ", str(c)).lower()


def chunk_flags(chunks: pd.Series, raw: pd.Series) -> pd.DataFrame:
    ch = chunks.map(_chunk)
    out = pd.DataFrame({k: ch.str.count(v).clip(upper=20).astype(np.float32) for k, v in VOCAB.items()})
    out["ch_table"] = raw.astype(str).str.contains("header_").astype(np.float32).to_numpy()
    out["ch_digits"] = (ch.str.count(r"\d") / (ch.str.len() + 1)).astype(np.float32).to_numpy()
    out["ch_len"] = np.log1p(ch.str.len()).astype(np.float32).to_numpy()
    return out


class HashedAnalyzer:
    """Wraps a fitted vectorizer's analyzer so its vocabulary can be stored as hashes instead of plain chart words
    (released model files must not contain text from the proprietary training charts). Features are unchanged."""

    def __init__(self, analyzer):
        self.analyzer = analyzer

    def __call__(self, doc):
        return [_hash_token(t) for t in self.analyzer(doc)]


def _hash_token(t):
    import hashlib
    return hashlib.sha256(("hilabs-ctx|" + t).encode("utf-8")).hexdigest()[:16]


def hash_vocabulary(vec: TfidfVectorizer) -> TfidfVectorizer:
    """In place: plain-text vocabulary → hashed vocabulary + hashing analyzer (identical transform output)."""
    if isinstance(vec.analyzer, HashedAnalyzer):
        return vec
    orig = vec.build_analyzer()
    vec.vocabulary_ = {_hash_token(t): i for t, i in vec.vocabulary_.items()}
    vec.analyzer = HashedAnalyzer(orig)
    if hasattr(vec, "stop_words_"):
        vec.stop_words_ = None
    return vec


class ContextTypeModel:
    """Logistic regression predicting the answer's type from term + chunk (+ subconcept)."""

    def __init__(self, use_sub: bool):
        self.use_sub = use_sub

    def _X(self, d, fit=False):
        ent = d["entity"].map(clean); ch = d["chunk_text"].map(_chunk)
        if fit:
            self.v1 = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True).fit(ent)
            self.v2 = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30000, sublinear_tf=True).fit(ch)
            if self.use_sub:
                self.oh = OneHotEncoder(handle_unknown="ignore").fit(d[["subconcept"]])
        parts = [self.v1.transform(ent), self.v2.transform(ch)]
        if self.use_sub:
            parts.append(self.oh.transform(d[["subconcept"]]) * 3)
        return sp.hstack(parts).tocsr()

    def fit(self, d, y):
        self.clf = LogisticRegression(C=4, max_iter=400, random_state=7).fit(self._X(d, fit=True), y)
        return self

    def proba(self, d, classes):
        pr = self.clf.predict_proba(self._X(d))
        col = {c: i for i, c in enumerate(classes)}
        P = np.zeros((len(d), len(classes)), np.float32)
        for j, c in enumerate(self.clf.classes_):
            if c in col:
                P[:, col[c]] = pr[:, j]
        return P


def type_of(cs):
    return cs.tag_of if cs.name == "SNOMEDCT_US" else cs.meta["chapter"].to_numpy()


def features_for(cand: pd.DataFrame, P: np.ndarray, flags: pd.DataFrame, cs, row_col="qid") -> pd.DataFrame:
    """Per-candidate context features. P / flags indexed by the term id in cand[row_col]."""
    grp = type_of(cs)
    classes = np.array(sorted(set(grp)))
    col = {c: i for i, c in enumerate(classes)}
    ctype = np.array([col[g] for g in grp])[cand["code_idx"].to_numpy()]
    q = cand[row_col].to_numpy()
    out = pd.DataFrame(index=cand.index)
    out["p_type_ctx"] = P[q, ctype]
    out["type_ctx_top"] = (P.argmax(1)[q] == ctype).astype(np.int8)
    for c in flags.columns:
        out[c] = flags[c].to_numpy()[q]
    return out


def classes_of(cs):
    return np.array(sorted(set(type_of(cs))))


def oof_probs(d: pd.DataFrame, cs, use_sub: bool, n_folds: int) -> np.ndarray:
    grp = type_of(cs)
    gi = d["code"].map(cs.code_to_idx)
    y = np.where(gi.notna(), grp[gi.fillna(0).astype(int)], "NA")
    classes = classes_of(cs)
    P = np.zeros((len(d), len(classes)), np.float32)
    for f in range(n_folds):
        tr, te = (d.fold != f).to_numpy(), (d.fold == f).to_numpy()
        m = ContextTypeModel(use_sub).fit(d[tr], y[tr])
        P[te] = m.proba(d[te], classes)
    return P


def fit_full(d: pd.DataFrame, cs, use_sub: bool) -> ContextTypeModel:
    grp = type_of(cs)
    gi = d["code"].map(cs.code_to_idx)
    y = np.where(gi.notna(), grp[gi.fillna(0).astype(int)], "NA")
    m = ContextTypeModel(use_sub).fit(d, y)
    hash_vocabulary(m.v2)   # the saved model must not contain plain chart words
    return m
