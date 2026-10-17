"""Exp08: joint ICD↔SNOMED coupling as a second reranking stage (CV, fold-aware pair statistics).

Usage: python exp08_coupling_cv.py <run>      (uses cache/oof_<run>_<system>.parquet for both systems)
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch
from src.dicts import load_train, load_system, SYSTEMS, CACHE
from src.cv import assign_folds, N_FOLDS
from src.engine import log
from src.evaluate import cv_rerank
from src.coupling import pair_table, coupling_features


def name_embeddings(cs, tag='sapbert'):
    path = os.path.join(CACHE, f'emb_names_{tag}_{cs.name}.npy')
    if not os.path.exists(path):
        from src.encoders import Encoder
        from src.normalize import clean_desc
        enc = Encoder(tag)
        np.save(path, enc.encode([clean_desc(str(x)) for x in cs.name_of], query=False, batch_size=1024))
        del enc; torch.cuda.empty_cache()
    return torch.from_numpy(np.load(path)).to('cuda' if torch.cuda.is_available() else 'cpu')


run = sys.argv[1]
use_sim = '--no-sim' not in sys.argv
df = load_train(); df['fold'] = assign_folds(df)
X = {}
for s in SYSTEMS:
    d = df[df.coding_system == s].reset_index(drop=True)
    cand = pd.read_parquet(os.path.join(CACHE, f'oof_{run}_{s}.parquet'))
    m, pred, _, _, cand = cv_rerank(cand.drop(columns=[c for c in cand.columns if c.startswith('cp_')]), d)
    log(s, 'stage-1', m)
    cand['mention_id'] = d.mention_id.to_numpy()[cand.qid.to_numpy()]
    ex = np.exp(cand.score - cand.groupby('qid').score.transform('max'))
    cand['prob'] = ex / ex.groupby(cand.qid).transform('sum')
    top = cand.sort_values(['qid', 'prob'], ascending=[True, False]).groupby('qid').head(5)
    X[s] = dict(d=d, cand=cand, top=top[['mention_id', 'code_idx', 'prob', 'fold']], cs=load_system(s), m1=m)

emb = {s: (name_embeddings(X[s]['cs']) if use_sim else None) for s in SYSTEMS}
pairs_all = df.pivot_table(index=['mention_id'], columns='coding_system', values='code', aggfunc='first').dropna()
pairs_all['fold'] = df.groupby('mention_id').fold.first().reindex(pairs_all.index)
pairs_all = pairs_all.reset_index()
summary = {}
for s in SYSTEMS:
    o = [t for t in SYSTEMS if t != s][0]
    c = X[s]['cand'].drop(columns=['score', 'prob'])
    feats = []
    for f in range(N_FOLDS):
        part = c[c.fold == f]
        pt = pair_table(pairs_all[pairs_all.fold != f])
        topb = X[o]['top'][X[o]['top'].fold == f]
        feats.append(coupling_features(part, s, topb, X[s]['cs'], X[o]['cs'], emb[s], emb[o], pt))
    c = c.join(pd.concat(feats))
    m, pred, model, fts, _ = cv_rerank(c.drop(columns=['mention_id']), X[s]['d'])
    log(s, 'stage-2 coupled', m, 'mentions with other-system rows:', round(float(c.groupby('qid').cp_has.max().mean()), 3))
    imp = pd.Series(model.feature_importance('gain'), index=fts).sort_values(ascending=False)
    print(imp.head(15).round(0).to_string())
    summary[s] = dict(stage1=X[s]['m1'], stage2=m)
    pred.to_parquet(os.path.join(CACHE, f'pred_{run}+cp_{s}.parquet'))
    c.to_parquet(os.path.join(CACHE, f'oof_{run}+cp_{s}.parquet'))
json.dump(summary, open(f'results/exp08_{run}{"" if use_sim else "_nosim"}.json', 'w'), indent=1)
print(summary)
