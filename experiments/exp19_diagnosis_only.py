"""Exp19: if the evaluation set is diagnosis/condition terms only (the brief's scope), what is the best ranker?

SNOMED, evaluated on DiagnosisCode terms only (5-fold CV, same folds), reranking each term's final-R1 top-50
(light on memory; 88% of gold codes are inside) with the final features + cross-encoder v2 scores:
  A  ranker trained on ALL rows, subconcept known          (= eval file has subconcept, or --assume-subconcept)
  B  ranker trained on ALL rows, no subconcept features    (= eval file without subconcept column)
  C  ranker trained on DIAGNOSIS rows only                 (specialized ranker)
  D  ranker trained on ALL rows but diagnosis rows up-weighted x3
Usage: python exp19_diagnosis_only.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, lightgbm as lgb
from src.dicts import load_train, CACHE
from src.cv import assign_folds, N_FOLDS
from src.engine import log
from src import ranker

s = 'SNOMEDCT_US'
df = load_train(); df['fold'] = assign_folds(df)
d = df[df.coding_system == s].reset_index(drop=True)
is_dx = (d.subconcept == 'DiagnosisCode').to_numpy()
st = pd.read_parquet(os.path.join(CACHE, f'stages_final_{s}.parquet'))
keep = st[st.r1 < 50][['qid', 'code_idx', 'r1']]
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
cand['is_dx'] = is_dx[cand.qid.to_numpy()]
feats_all = [c for c in ranker.feature_cols(cand) if c != 'is_dx']
feats_nosub = [c for c in feats_all if c not in ('p_grp_sub', 'subconcept')]
log('table', cand.shape, '| dx terms', int(is_dx.sum()), '| gold in top-50 (dx)',
    round(float(cand[cand.is_dx].groupby('qid').label.max().sum() / is_dx.sum()), 4))


def run(train_mask_fn, feats, weight_dx=1.0):
    sc = np.zeros(len(cand))
    for f in range(N_FOLDS):
        tr = (cand.fold != f).to_numpy() & train_mask_fn(cand)
        te = (cand.fold == f).to_numpy()
        c = cand[tr]
        has_pos = c.groupby('qid').label.transform('max') > 0
        c = c[has_pos].sort_values('qid', kind='stable')
        grp = c.groupby('qid', sort=False).size().to_numpy()
        w = np.where(c.groupby('qid', sort=False).is_dx.first().to_numpy(), weight_dx, 1.0)
        ds = lgb.Dataset(c[feats], c.label, group=grp)
        ds.set_weight(np.repeat(w, grp)) if weight_dx != 1.0 else None
        m = lgb.train(ranker.PARAMS, ds, num_boost_round=600)
        sc[te] = m.predict(cand.loc[te, feats])
    c = cand.assign(score=sc)
    top = c[c.is_dx].sort_values(['qid', 'score'], ascending=[True, False]).groupby('qid').head(1)
    return round(float(top.label.sum() / is_dx.sum()), 4)


res = {
    'A all rows, subconcept known': run(lambda c: np.ones(len(c), bool), feats_all),
    'B all rows, no subconcept': run(lambda c: np.ones(len(c), bool), feats_nosub),
    'C diagnosis rows only': run(lambda c: c.is_dx.to_numpy(), feats_nosub),
    'D all rows, diagnosis x3 weight': run(lambda c: np.ones(len(c), bool), feats_all, weight_dx=3.0),
}
for k, v in res.items():
    log(f'{k:38s} SNOMED accuracy on diagnosis terms: {v}')
json.dump(res, open('results/exp19_diagnosis_only.json', 'w'), indent=1)
