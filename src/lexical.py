"""Lexical retrieval over dictionary descriptions: character n-gram TF-IDF and word BM25.

Both index one row per description and aggregate to codes by max score. Sparse matrices are
cached under cache/ so the index is built once per dictionary version.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .dicts import CACHE, atomic_save
from .normalize import clean_desc


def _topk_sparse(Q, DT, k):
    S = (Q @ DT).toarray() if sp.issparse(Q) else Q @ DT
    k = min(k, S.shape[1] - 1)
    idx = np.argpartition(-S, kth=k, axis=1)[:, :k]
    sc = np.take_along_axis(S, idx, 1)
    o = np.argsort(-sc, 1)
    return np.take_along_axis(idx, o, 1), np.take_along_axis(sc, o, 1)


class LexicalIndex:
    """kind = 'char' (char_wb 2-4 TF-IDF, cosine) or 'bm25' (word uni+bigram BM25)."""

    def __init__(self, cs, kind: str = "char", k1: float = 1.2, b: float = 0.75):
        self.cs, self.kind = cs, kind
        path = os.path.join(CACHE, f"lex_{kind}_{cs.name}.pkl")
        if os.path.exists(path):
            self.vec, self.DT, self.idf = pickle.load(open(path, "rb"))
            return
        texts = cs.desc.description.map(clean_desc).tolist()
        if kind == "char":
            self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True,
                                       dtype=np.float32)
            D = self.vec.fit_transform(texts)
            self.idf = None
        else:
            self.vec = CountVectorizer(ngram_range=(1, 2), min_df=1, token_pattern=r"[a-z0-9]+", dtype=np.float32)
            tf = self.vec.fit_transform(texts).tocsr()
            N = tf.shape[0]
            dfreq = np.bincount(tf.indices, minlength=tf.shape[1])
            self.idf = np.log(1 + (N - dfreq + 0.5) / (dfreq + 0.5)).astype(np.float32)
            dl = np.asarray(tf.sum(1)).ravel()
            norm = k1 * (1 - b + b * dl / dl.mean())
            tf = tf.tocoo()
            w = tf.data * (k1 + 1) / (tf.data + norm[tf.row]) * self.idf[tf.col]
            D = sp.csr_matrix((w.astype(np.float32), (tf.row, tf.col)), shape=tf.shape)
        self.DT = D.T.tocsr()
        atomic_save(path, lambda f: pickle.dump((self.vec, self.DT, self.idf), f, protocol=4))

    def query_matrix(self, queries):
        Q = self.vec.transform(queries)
        if self.kind == "bm25":
            Q.data[:] = 1.0
        return Q

    def search(self, queries, k=100, bs=64):
        """Returns (desc_idx [n,k], score [n,k])."""
        I, S = [], []
        for s in range(0, len(queries), bs):
            i, v = _topk_sparse(self.query_matrix(queries[s:s + bs]), self.DT, k)
            I.append(i); S.append(v)
        return np.vstack(I), np.vstack(S)

    def score_pairs(self, query: str, desc_rows: np.ndarray) -> np.ndarray:
        """Score one query against specific description rows."""
        q = self.query_matrix([query])
        return np.asarray((q @ self.DT[:, desc_rows]).todense()).ravel()
