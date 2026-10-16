"""Exp04: full candidate generation + LambdaRank reranker under 5-fold CV.

Usage: python exp04_rerank_cv.py <run_name> [encoder_tag ...]
Saves OOF candidate tables to cache/oof_<run>_<system>.parquet and prints per-system metrics.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src.dicts import load_train, SYSTEMS, CACHE
from src.cv import assign_folds, N_FOLDS
from src.engine import Engine, log
from src import ranker

run = sys.argv[1]
enc_tags = sys.argv[2:]
encoders = {}
if enc_tags:
    from src.encoders import Encoder
    for t in enc_tags:
        encoders[t] = Encoder(t if not os.path.isdir(t) else t)
df = load_train(); df['fold'] = assign_folds(df)
only = os.environ.get('ONLY_SYSTEM')
summary = {}
for sysn in SYSTEMS:
    if only and sysn != only:
        continue
    d = df[df.coding_system == sysn].reset_index(drop=True)
    path = os.path.join(CACHE, f'oof_{run}_{sysn}.parquet')
    if os.path.exists(path):
        cand = pd.read_parquet(path)
    else:
        eng = Engine(sysn, encoders)
        parts = []
        for f in range(N_FOLDS):
            tr, te = d[d.fold != f], d[d.fold == f].reset_index()
            eng.set_memory(tr)
            c = eng.candidates(te)
            c['qid'] = te['index'].to_numpy()[c['row'].to_numpy()]
            c['fold'] = f
            parts.append(c)
        cand = pd.concat(parts, ignore_index=True)
        gold = d.code.map(eng.cs.code_to_idx).fillna(-1).astype(np.int64).to_numpy()
        cand['label'] = (cand.code_idx.to_numpy() == gold[cand.qid.to_numpy()]).astype(np.int8)
        cand.to_parquet(path)
    recall = cand.groupby('qid').label.max().reindex(range(len(d))).fillna(0)
    log(sysn, 'candidate recall', round(recall.mean(), 4), 'cands/row', round(len(cand) / len(d), 1))
    # CV reranker
    cand['score'] = 0.0
    for f in range(N_FOLDS):
        tr = cand.fold != f
        model, feats = ranker.fit(cand[tr])
        cand.loc[~tr, 'score'] = ranker.predict(model, cand[~tr], feats)
    best = ranker.top1(cand)
    hit = cand.set_index(['qid', 'code_idx']).label
    pred_ok = np.array([hit.get((q, c), 0) for q, c in zip(best.index, best.code_idx)])
    acc = pd.Series(pred_ok, index=best.index).reindex(range(len(d))).fillna(0)
    # long-tail: gold code unseen in train folds
    seen = np.zeros(len(d), bool)
    for f in range(N_FOLDS):
        m = (d.fold == f).to_numpy()
        seen[m] = d.code[m].isin(set(d.code[~m])).to_numpy()
    res = dict(acc=round(acc.mean(), 4), recall=round(recall.mean(), 4),
               acc_seen=round(acc[seen].mean(), 4), acc_unseen=round(acc[~seen].mean(), 4), n_unseen=int((~seen).sum()))
    imp = pd.Series(model.feature_importance('gain'), index=feats).sort_values(ascending=False)
    log(sysn, res)
    print(imp.head(25).round(0).to_string())
    summary[sysn] = res
    best.assign(gold_ok=pred_ok).to_parquet(os.path.join(CACHE, f'pred_{run}_{sysn}.parquet'))
json.dump(summary, open(f'results/exp04_{run}.json', 'w'), indent=1)
print(summary)
