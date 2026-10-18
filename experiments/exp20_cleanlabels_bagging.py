"""Exp20: two cheap countermeasures, evaluated on the ORIGINAL gold labels (5-fold CV, final features, top-50 pool).

  base     – current ranker
  clean    – training labels cleaned: within groups of near-identical training terms (same normalized key) whose gold
             codes conflict, relabel to the group's majority code (training folds only; evaluation untouched)
  bag5     – average of 5 rankers with different seeds / bagging subsamples
Usage: python exp20_cleanlabels_bagging.py [system]
"""
import sys, os, re, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, lightgbm as lgb
from src.dicts import load_train, load_system, CACHE
from src.cv import assign_folds, N_FOLDS
from src.engine import log
from src.normalize import clean
from src import ranker

s = sys.argv[1] if len(sys.argv) > 1 else 'SNOMEDCT_US'
cs = load_system(s)
df = load_train(); df['fold'] = assign_folds(df)
d = df[df.coding_system == s].reset_index(drop=True)
STOP = r'\b(?:the|a|an|of|and|with|in|on|to|is|are|has|have|patient|pt|mild|moderate|severe|likely|possible|noted)\b'
d['k'] = d.entity.map(lambda e: ' '.join(sorted(re.sub(r'[^a-z0-9 ]', ' ', re.sub(STOP, ' ', clean(e))).split())))
gold = d.code.map(cs.code_to_idx).fillna(-1).astype(int).to_numpy()

st = pd.read_parquet(os.path.join(CACHE, f'stages_final_{s}.parquet'))
keep = st[st.r1 < 50][['qid', 'code_idx']]
cand = pd.concat([pd.read_parquet(os.path.join(CACHE, f'oofpart_ftsap2_{s}_f{f}.parquet')) for f in range(N_FOLDS)],
                 ignore_index=True).merge(keep, on=['qid', 'code_idx'])
old = pd.concat([pd.read_parquet(os.path.join(CACHE, f'oofpart_ftsap_{s}_f{f}.parquet')) for f in range(N_FOLDS)],
                ignore_index=True)
cols = [c for c in old.columns if c.startswith(('dense_ft_', 'knn_ft_'))]
cand = cand.merge(old[['qid', 'code_idx'] + cols].rename(columns={c: c.replace('_ft_', '_ft1_') for c in cols}),
                  on=['qid', 'code_idx'], how='left'); del old
ce = pd.read_parquet(os.path.join(CACHE, f'ce_scores_cectx2_{s}.parquet'))[['qid', 'code_idx', 'ce']]
cand = cand.merge(ce, on=['qid', 'code_idx'], how='left')
cand['ce_rank'] = cand.groupby('qid').ce.rank(ascending=False, method='first')
cand['ce_gap'] = cand.ce - cand.groupby('qid').ce.transform('max')
ctx = pd.read_parquet(os.path.join(CACHE, f'ctxfeat_{s}.parquet'))
cand = cand.merge(ctx, on=['qid', 'code_idx'], how='left')
feats = ranker.feature_cols(cand)
log(s, 'table', cand.shape)


def evaluate(scores):
    c = cand.assign(score=scores)
    top = c.sort_values(['qid', 'score'], ascending=[True, False]).groupby('qid').head(1)
    return round(float(top.label.sum() / len(d)), 4)


def fit_predict(labels, seeds=(7,), bag=False):
    sc = np.zeros(len(cand))
    for f in range(N_FOLDS):
        tr = (cand.fold != f).to_numpy(); te = ~tr
        c = cand[tr].assign(lab=labels[tr])
        c = c[c.groupby('qid').lab.transform('max') > 0].sort_values('qid', kind='stable')
        grp = c.groupby('qid', sort=False).size().to_numpy()
        for sd in seeds:
            p = dict(ranker.PARAMS, seed=sd, bagging_seed=sd, feature_fraction_seed=sd)
            if bag:
                p.update(bagging_fraction=0.7, feature_fraction=0.7)
            m = lgb.train(p, lgb.Dataset(c[feats], c.lab, group=grp), num_boost_round=600)
            sc[te] += m.predict(cand.loc[te, feats]) / len(seeds)
    return sc


res = {}
orig = cand.label.to_numpy()
res['base'] = evaluate(fit_predict(orig))
# cleaned labels: majority code within conflicting near-duplicate groups, computed per training split
clean_lab = np.zeros((N_FOLDS, len(cand)), np.int8)
q = cand.qid.to_numpy(); ci = cand.code_idx.to_numpy()
changed = 0
for f in range(N_FOLDS):
    tr_rows = d[d.fold != f]
    maj = tr_rows.groupby('k').code.agg(lambda x: x.value_counts().index[0])
    newgold = gold.copy()
    for i in tr_rows.index:
        mc = cs.code_to_idx.get(maj.get(d.k[i], d.code[i]), gold[i])
        if mc != gold[i]:
            newgold[i] = mc; changed += 1
    clean_lab[f] = (ci == newgold[q]).astype(np.int8)
log(s, 'training labels changed by cleaning (summed over folds):', changed)
sc = np.zeros(len(cand))
for f in range(N_FOLDS):
    tr = (cand.fold != f).to_numpy(); te = ~tr
    c = cand[tr].assign(lab=clean_lab[f][tr])
    c = c[c.groupby('qid').lab.transform('max') > 0].sort_values('qid', kind='stable')
    grp = c.groupby('qid', sort=False).size().to_numpy()
    m = lgb.train(ranker.PARAMS, lgb.Dataset(c[feats], c.lab, group=grp), num_boost_round=600)
    sc[te] = m.predict(cand.loc[te, feats])
res['clean labels'] = evaluate(sc)
res['bag5 (seeds)'] = evaluate(fit_predict(orig, seeds=(7, 11, 23, 41, 97), bag=True))
for k, v in res.items():
    log(s, f'{k:15s} accuracy (original gold): {v}')
json.dump(res, open(f'results/exp20_{s}.json', 'w'), indent=1)
