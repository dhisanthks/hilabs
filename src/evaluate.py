"""CV reranking + metrics shared by all experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import ranker
from .cv import N_FOLDS


def cv_rerank(cand: pd.DataFrame, d: pd.DataFrame, params=None, rounds=600, feats=None):
    """cand has qid (index into d), fold, label. Returns (metrics, per-row prediction frame, last model, feats)."""
    cand = cand.copy()
    cand["score"] = 0.0
    model = None
    for f in range(N_FOLDS):
        tr = (cand.fold != f).to_numpy()
        model, fts = ranker.fit(cand[tr], feats=feats, rounds=rounds, params=params)
        cand.loc[~tr, "score"] = ranker.predict(model, cand[~tr], fts)
    best = ranker.top1(cand)
    lab = cand.set_index(["qid", "code_idx"])["label"]
    ok = lab.reindex(list(zip(best.index, best.code_idx))).fillna(0).to_numpy()
    pred = pd.DataFrame({"code_idx": best.code_idx.to_numpy(), "score": best.score.to_numpy(),
                         "margin": best.margin.to_numpy(), "prob": best.prob.to_numpy(), "ok": ok}, index=best.index)
    pred = pred.reindex(range(len(d)))
    pred["ok"] = pred["ok"].fillna(0)
    recall = cand.groupby("qid").label.max().reindex(range(len(d))).fillna(0)
    seen = np.zeros(len(d), bool)
    for f in range(N_FOLDS):
        m = (d.fold == f).to_numpy()
        seen[m] = d.code[m].isin(set(d.code[~m])).to_numpy()
    pred["seen"] = seen
    pred["in_pool"] = recall.to_numpy()
    acc = pred["ok"]
    m = dict(acc=round(float(acc.mean()), 4), recall=round(float(recall.mean()), 4),
             acc_seen=round(float(acc[seen].mean()), 4), acc_unseen=round(float(acc[~seen].mean()), 4),
             acc_given_in_pool=round(float(acc[recall > 0].mean()), 4))
    return m, pred, model, fts, cand
