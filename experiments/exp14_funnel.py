"""Exp14: stage-by-stage funnel analysis (mentor suggestion) + quality-weighted retriever fusion.

For each system, recall@k (k = 1, 3, 5, 10, 30, 50) of the gold code at every stage:
  each retriever → union pool → weighted-RRF fusion → R1 → cross-encoder (within R1 top-30) → R2 → R3.
Shows where the gold code is lost. Also tests quality-weighted reciprocal-rank fusion (weights = each retriever's MRR
on the *training folds*, fold-aware) as (a) a standalone ranker and (b) an extra R1 feature.

Usage: python exp14_funnel.py <run=ftsap2> <add_from=ftsap> [systems]
Saves cache/stages_<system>.parquet (qid, code_idx, label, fold, s1, s2, s3, wrrf) for follow-up experiments.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch
from src.dicts import load_train, load_system, SYSTEMS, CACHE
from src.cv import assign_folds, N_FOLDS
from src.engine import log
from src.evaluate import cv_rerank
from src.coupling import pair_table, coupling_features

run = sys.argv[1] if len(sys.argv) > 1 else 'ftsap2'
add_from = sys.argv[2] if len(sys.argv) > 2 else 'ftsap'
systems = sys.argv[3].split(',') if len(sys.argv) > 3 else list(SYSTEMS)
KS = [1, 3, 5, 10, 30, 50]
df = load_train(); df['fold'] = assign_folds(df)


def recall_at(c, col, n, ascending=False):
    """recall@k of gold when ranking candidates by col within each qid."""
    r = c.groupby('qid')[col].rank(ascending=ascending, method='first')
    g = r[c.label == 1]
    return {f'@{k}': round(float((g <= k).sum() / n * 100), 1) for k in KS}


def load_table(s):
    cand = pd.concat([pd.read_parquet(os.path.join(CACHE, f'oofpart_{run}_{s}_f{f}.parquet')) for f in range(N_FOLDS)],
                     ignore_index=True)
    if add_from:
        old = pd.concat([pd.read_parquet(os.path.join(CACHE, f'oofpart_{add_from}_{s}_f{f}.parquet'),
                                         columns=None) for f in range(N_FOLDS)], ignore_index=True)
        cols = [c for c in old.columns if c.startswith(('dense_ft_', 'knn_ft_'))]
        old = old[['qid', 'code_idx'] + cols].rename(columns={c: c.replace('_ft_', '_ft1_') for c in cols})
        cand = cand.merge(old, on=['qid', 'code_idx'], how='left'); del old
    return cand


def weighted_rrf(cand, k0=60):
    """Fold-aware quality-weighted RRF: weight of a retriever = its MRR on the other folds."""
    rcols = [c for c in cand.columns if c.endswith('_rank') and not c.startswith('q_') and c not in ('ce_rank',)]
    out = np.zeros(len(cand), np.float32)
    weights = {}
    for f in range(N_FOLDS):
        tr = (cand.fold != f).to_numpy(); te = ~tr
        g = cand[tr & (cand.label == 1).to_numpy()]
        nq = cand.loc[tr, 'qid'].nunique()
        w = {c: float(np.where(g[c].fillna(999) < 999, 1.0 / (g[c].fillna(999) + 1), 0).sum() / nq) for c in rcols}
        weights[f] = w
        acc = np.zeros(te.sum(), np.float32)
        for c, wc in w.items():
            r = cand.loc[te, c].fillna(999).to_numpy()
            acc += np.where(r < 999, wc / (k0 + r), 0)
        out[te] = acc
    return out, weights[0]


summary = {}
for s in systems:
    d = df[df.coding_system == s].reset_index(drop=True)
    n = len(d)
    cand = load_table(s)
    res = {}
    rcols = [c for c in cand.columns if c.endswith('_rank') and not c.startswith('q_')]
    for c in rcols:
        res[f'retriever {c}'] = recall_at(cand.assign(_r=-cand[c].fillna(999)), '_r', n)
    res['union pool'] = {f'@{k}': round(float(cand.groupby("qid").label.max().sum() / n * 100), 1) for k in KS}
    cand['wrrf'], w0 = weighted_rrf(cand)
    res['weighted RRF (MRR weights)'] = recall_at(cand, 'wrrf', n)
    log(s, 'retriever weights (fold 0):', {k: round(v, 3) for k, v in sorted(w0.items(), key=lambda x: -x[1])})
    # R1 without / with wrrf feature
    m1a, *_ = cv_rerank(cand.drop(columns=['wrrf']), d)
    m1, _, _, _, cand = cv_rerank(cand, d)
    res['R1 (LightGBM)'] = recall_at(cand, 'score', n)
    log(s, 'R1 without wRRF', m1a['acc'], '| with wRRF feature', m1['acc'])
    cand['s1'] = cand['score']
    cand = cand.sort_values(['qid', 'score'], ascending=[True, False])
    cand['r1'] = cand.groupby('qid').cumcount().astype(np.int16)
    ce = pd.read_parquet(os.path.join(CACHE, f'ce_scores_cectx_{s}.parquet'))[['qid', 'code_idx', 'ce']]
    cand = cand.drop(columns=['score']).merge(ce, on=['qid', 'code_idx'], how='left')
    res['cross-encoder (within R1 top-30)'] = recall_at(cand.assign(_c=cand.ce.fillna(-99)), '_c', n)
    cand['ce_rank'] = cand.groupby('qid').ce.rank(ascending=False, method='first')
    cand['ce_gap'] = cand.ce - cand.groupby('qid').ce.transform('max')
    m2, _, _, _, cand = cv_rerank(cand, d)
    res['R2 (+CE)'] = recall_at(cand, 'score', n)
    cand['s2'] = cand['score']
    summary[s] = dict(table=res, R1_without_wrrf=m1a, R1=m1, R2=m2)
    cand[['qid', 'code_idx', 'label', 'fold', 's1', 's2', 'wrrf', 'r1']].to_parquet(os.path.join(CACHE, f'stages_{s}.parquet'))
    print(f'\n===== {s} — recall@k (% of terms whose gold code is within the top k) =====')
    print(pd.DataFrame(res).T.to_string())
    del cand
json.dump(summary, open(f'results/exp14_funnel_{"_".join(systems)}.json', 'w'), indent=1)
