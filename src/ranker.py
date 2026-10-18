"""LightGBM LambdaRank reranker over the candidate table."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

NON_FEATURES = {"row", "code_idx", "label", "fold", "qid", "score"}

PARAMS = dict(objective="lambdarank", metric="ndcg", eval_at=[1], learning_rate=0.05, num_leaves=31,
              min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              lambdarank_truncation_level=20, verbose=-1, seed=7, deterministic=True, num_threads=8)


def feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURES]


def fit(cand: pd.DataFrame, feats=None, rounds=600, params=None):
    """cand must have qid (query id), label. Queries without a positive are dropped."""
    feats = feats or feature_cols(cand)
    has_pos = cand.groupby("qid")["label"].transform("max") > 0
    c = cand[has_pos].sort_values("qid", kind="stable")
    grp = c.groupby("qid", sort=False).size().to_numpy()
    ds = lgb.Dataset(c[feats], c["label"], group=grp, free_raw_data=False)
    p = dict(PARAMS, **(params or {}))
    model = lgb.train(p, ds, num_boost_round=rounds)
    return model, feats


def predict(model, cand, feats):
    return model.predict(cand[feats], num_threads=8)


def top1(cand: pd.DataFrame, score_col="score"):
    """Best candidate per qid → DataFrame[qid, code_idx, score, margin, prob]."""
    c = cand[["qid", "code_idx", score_col]].sort_values(["qid", score_col], ascending=[True, False], kind="stable")
    first = c.groupby("qid", sort=False).head(1).set_index("qid")
    second = c.groupby("qid", sort=False).nth(1).set_index("qid")[score_col]
    first["margin"] = first[score_col] - second.reindex(first.index).fillna(first[score_col] - 10)
    # softmax prob of the winner within its query
    ex = np.exp(c[score_col] - c.groupby("qid")[score_col].transform("max"))
    first["prob"] = (1.0 / ex.groupby(c["qid"]).sum()).reindex(first.index)
    return first


BAG_SEEDS = (7, 11, 23, 41, 97)


def fit_bag(cand: pd.DataFrame, feats, seeds=BAG_SEEDS, rounds=600):
    """Seed-bagged LambdaRank (Exp20: +0.5 SNOMED, +0.2 ICD): each model sees 70% rows / 70% features."""
    models = []
    for sd in seeds:
        m, feats = fit(cand, feats=feats, rounds=rounds,
                       params=dict(seed=sd, bagging_seed=sd, feature_fraction_seed=sd,
                                   bagging_fraction=0.7, feature_fraction=0.7))
        models.append(m)
    return models, feats


def predict_any(models, X):
    """Average score of a booster or a list of boosters."""
    if not isinstance(models, (list, tuple)):
        return models.predict(X, num_threads=8)
    return np.mean([m.predict(X, num_threads=8) for m in models], axis=0)
