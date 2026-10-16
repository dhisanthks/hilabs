"""Exp07: cross-encoder reranking under CV, stacked into the LightGBM reranker.

Usage: python exp07_crossenc_cv.py <cand_run> <ce_run> [--query plain|sub|ctx|subctx] [--epochs N] [--topk K]
Needs cache/oof_<cand_run>_<system>.parquet for both systems (candidate tables with labels).
"""
import sys, os, gc, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch
from src.dicts import load_train, load_system, SYSTEMS, CACHE
from src.cv import assign_folds, N_FOLDS
from src.engine import log
from src.evaluate import cv_rerank
from src.normalize import clean
from src.crossenc import CrossEncoder, cand_text, make_query, context_window

ap = argparse.ArgumentParser()
ap.add_argument('cand_run'); ap.add_argument('ce_run')
ap.add_argument('--query', default='plain'); ap.add_argument('--epochs', type=int, default=1)
ap.add_argument('--topk', type=int, default=30); ap.add_argument('--neg', type=int, default=15)
ap.add_argument('--base', default='sapbert'); ap.add_argument('--systems', default=','.join(SYSTEMS))
a = ap.parse_args()
systems = a.systems.split(',')

df = load_train(); df['fold'] = assign_folds(df)
S = {}
for s in systems:
    d = df[df.coding_system == s].reset_index(drop=True)
    cand = pd.read_parquet(os.path.join(CACHE, f'oof_{a.cand_run}_{s}.parquet'))
    m, pred, _, _, cand = cv_rerank(cand, d)          # OOF first-stage scores → hard negatives / top-k
    log(s, 'first stage', m)
    cs = load_system(s)
    sub = d['subconcept'] if 'sub' in a.query else None
    ctx = [context_window(e, c) for e, c in zip(d.entity, d.chunk_text)] if 'ctx' in a.query else None
    q = [make_query(clean(d.entity[i]), sub[i] if sub is not None else None, ctx[i] if ctx else None)
         for i in range(len(d))]
    gold = d.code.map(cs.code_to_idx).fillna(-1).astype(int).to_numpy()
    cand = cand.sort_values(['qid', 'score'], ascending=[True, False])
    cand['r1'] = cand.groupby('qid').cumcount()
    S[s] = dict(d=d, cand=cand, cs=cs, q=q, gold=gold)

ce_col = np.full(0, 0)
for f in range(N_FOLDS):
    path = {s: os.path.join(CACHE, f'ce_{a.ce_run}_{s}_f{f}.npy') for s in systems}
    if all(os.path.exists(p) for p in path.values()):
        continue
    groups = []
    for s in systems:
        X = S[s]; c = X['cand']; d = X['d']
        tr = c[(c.fold != f) & (c.label == 0) & (c.r1 < a.neg + 5)]
        negs = tr.groupby('qid').code_idx.apply(list)
        for qi in np.where((d.fold != f).to_numpy() & (X['gold'] >= 0))[0]:
            ng = negs.get(qi, [])[:a.neg]
            if not ng:
                continue
            groups.append((X['q'][qi], [cand_text(X['cs'], X['gold'][qi])] + [cand_text(X['cs'], k) for k in ng]))
    log(f'fold {f}: {len(groups)} training groups')
    ce = CrossEncoder(a.base).fit(groups, epochs=a.epochs, group_size=a.neg + 1)
    for s in systems:
        X = S[s]; c = X['cand']
        te = c[(c.fold == f) & (c.r1 < a.topk)]
        sc = ce.score([X['q'][i] for i in te.qid], [cand_text(X['cs'], k) for k in te.code_idx])
        np.save(path[s], np.stack([te.index.to_numpy(), sc]).astype(np.float64))
    del ce; gc.collect(); torch.cuda.empty_cache()

summary = {}
for s in systems:
    X = S[s]; c = X['cand'].drop(columns=['score'])
    c['ce'] = np.nan
    for f in range(N_FOLDS):
        arr = np.load(os.path.join(CACHE, f'ce_{a.ce_run}_{s}_f{f}.npy'))
        c.loc[arr[0].astype(np.int64), 'ce'] = arr[1]
    c['ce_rank'] = c.groupby('qid')['ce'].rank(ascending=False, method='first').fillna(999)
    c['ce_gap'] = c['ce'] - c.groupby('qid')['ce'].transform('max')
    c['ce'] = c['ce'].fillna(c['ce'].min() - 1); c['ce_gap'] = c['ce_gap'].fillna(-99)
    c['r1'] = c['r1'].astype(np.int16)
    m, pred, model, feats, _ = cv_rerank(c, X['d'])
    log(s, 'with cross-encoder', m)
    # cross-encoder alone (argmax ce among top-k)
    top = c.sort_values(['qid', 'ce'], ascending=[True, False]).groupby('qid').head(1)
    log(s, 'cross-encoder alone acc', round(top.label.sum() / len(X['d']), 4))
    imp = pd.Series(model.feature_importance('gain'), index=feats).sort_values(ascending=False)
    print(imp.head(12).round(0).to_string())
    c.to_parquet(os.path.join(CACHE, f'oof_{a.cand_run}+{a.ce_run}_{s}.parquet'))
    pred.to_parquet(os.path.join(CACHE, f'pred_{a.cand_run}+{a.ce_run}_{s}.parquet'))
    summary[s] = m
json.dump(summary, open(f'results/exp07_{a.cand_run}_{a.ce_run}.json', 'w'), indent=1)
print(summary)
