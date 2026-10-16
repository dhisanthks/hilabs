"""Candidate generation: merge several retrievers into one (query row, candidate code) table.

Retrievers
  char   – char n-gram TF-IDF over dictionary descriptions (typo/OCR robust)
  bm25   – word BM25 over dictionary descriptions (exact clinical words)
  dense  – bi-encoder over dictionary descriptions (semantic paraphrase)
  knn    – bi-encoder kNN over *training* terms → their gold codes (local coder convention)
  kchar  – char TF-IDF kNN over training terms
Every retriever runs on all query variants (original, abbreviation-expanded, spelling-repaired)
and keeps the max score per code. Output: long DataFrame [row, code_idx, <retriever features>].
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .lexical import _topk_sparse


def _aggregate(I, S, owner, desc_code, n_rows, top):
    """Per original row: best score per code over all variant rows; keep `top` codes. → dict row→(codes, scores)."""
    out = [dict() for _ in range(n_rows)]
    for r in range(len(owner)):
        dct = out[owner[r]]
        codes = desc_code[I[r]] if desc_code is not None else I[r]
        for c, s in zip(codes, S[r]):
            if s > dct.get(c, -1e9):
                dct[c] = s
    res = []
    for dct in out:
        items = sorted(dct.items(), key=lambda x: -x[1])[:top]
        res.append(items)
    return res


def to_frame(items_per_row, name):
    rows, codes, scores, ranks = [], [], [], []
    for r, items in enumerate(items_per_row):
        for k, (c, s) in enumerate(items):
            rows.append(r); codes.append(c); scores.append(s); ranks.append(k)
    return pd.DataFrame({"row": np.array(rows, np.int32), "code_idx": np.array(codes, np.int64),
                         f"{name}_sim": np.array(scores, np.float32), f"{name}_rank": np.array(ranks, np.int16)})


def lexical_candidates(index, var_lists, top=100, k=200):
    qs, owner = [], []
    for r, vs in enumerate(var_lists):
        qs += vs; owner += [r] * len(vs)
    I, S = index.search(qs, k=k)
    return to_frame(_aggregate(I, S, owner, index.cs.desc.code_idx.to_numpy(), len(var_lists), top), index.kind)


def dense_candidates(Qv, owner, D, desc_code, n_rows, top=100, k=300, name="dense", groups=None, kg=15):
    """Global dense top-k; if groups given (SNOMED semantic tags), also top-kg codes within each group,
    with a within-group rank feature."""
    from .encoders import topk_dense, topk_dense_grouped
    if not groups:
        I, S = topk_dense(Qv, D, k=k)
        return to_frame(_aggregate(I, S, owner, desc_code, n_rows, top), name)
    I, S, G = topk_dense_grouped(Qv, D, groups, k=k, kg=kg * 3)
    main = to_frame(_aggregate(I, S, owner, desc_code, n_rows, top), name)
    gframes = []
    for g, (gi, gs) in G.items():
        f = to_frame(_aggregate(gi, gs, owner, desc_code, n_rows, kg), "g")
        gframes.append(f[["row", "code_idx", "g_rank"]])
    gf = pd.concat(gframes, ignore_index=True).drop_duplicates(["row", "code_idx"])
    gf = gf.rename(columns={"g_rank": f"{name}_grank"})
    return main.merge(gf, on=["row", "code_idx"], how="outer")


def knn_features(Qv_rows, M, mem_codes, k=25, name="knn"):
    """kNN over memory (training) term embeddings. Qv_rows: one vector per query row (original variant).
    Features per (row, code): max sim, sum of sims, count among top-k, best neighbour rank."""
    from .encoders import topk_dense
    I, S = topk_dense(Qv_rows, M, k=min(k, len(M)))
    return _knn_frame(I, S, mem_codes, name)


def _knn_frame(I, S, mem_codes, name):
    recs = {}
    for r in range(I.shape[0]):
        for j in range(I.shape[1]):
            c = mem_codes[I[r, j]]; s = float(S[r, j])
            key = (r, c)
            if key not in recs:
                recs[key] = [s, s, 1, j]
            else:
                v = recs[key]; v[1] += s; v[2] += 1
    if not recs:
        return pd.DataFrame(columns=["row", "code_idx"])
    keys = np.array(list(recs.keys()), dtype=np.int64)
    vals = np.array(list(recs.values()), dtype=np.float32)
    return pd.DataFrame({"row": keys[:, 0].astype(np.int32), "code_idx": keys[:, 1],
                         f"{name}_max": vals[:, 0], f"{name}_sum": vals[:, 1],
                         f"{name}_cnt": vals[:, 2], f"{name}_rank": vals[:, 3]})


def char_knn_features(queries, mem_texts, mem_codes, k=25, name="kchar"):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, dtype=np.float32)
    A = vec.fit_transform(mem_texts)
    B = vec.transform(queries)
    I, S = [], []
    AT = A.T.tocsr()
    for s in range(0, B.shape[0], 1024):
        i, v = _topk_sparse(B[s:s + 1024], AT, min(k, A.shape[0] - 1))
        I.append(i); S.append(v)
    return _knn_frame(np.vstack(I), np.vstack(S), mem_codes, name)


def merge(frames):
    out = None
    for f in frames:
        if f is None or len(f) == 0:
            continue
        out = f if out is None else out.merge(f, on=["row", "code_idx"], how="outer")
    return out
