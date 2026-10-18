"""Exp24: raise recall@10/20 of the candidate list given to the LLM (SNOMED DiagnosisCode terms, 5-fold OOF).

  A  current list ranker (LambdaRank tuned for top-1)                        — as used for llm/llm2 prompts
  B  recall-oriented LambdaRank (NDCG@20 objective, truncation 50, graded label gain)
  C  diverse list: top-(K-6) of A + best 2 by cross-encoder + 2 by fine-tuned dense + 2 by kNN not already in the list
Reports recall@10 / @20 on diagnosis terms (gold in the final-R1 top-50 bounds all variants).
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src.dicts import load_train, CACHE
from src.cv import assign_folds, N_FOLDS
from src.engine import log
from src import ranker

s = 'SNOMEDCT_US'
df = load_train(); df['fold'] = assign_folds(df)
d = df[df.coding_system == s].reset_index(drop=True)
dx = np.where((d.subconcept == 'DiagnosisCode').to_numpy())[0]
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
cand = cand.merge(pd.read_parquet(os.path.join(CACHE, f'ctxfeat_{s}.parquet')), on=['qid', 'code_idx'], how='left')
feats = ranker.feature_cols(cand)
nd = len(dx); dxset = set(dx)
log('gold in final-R1 top-50 (dx):', round(float(cand[cand.qid.isin(dxset)].groupby('qid').label.max().sum() / nd), 4))


def oof_scores(params, rounds=600):
    sc = np.zeros(len(cand))
    for f in range(N_FOLDS):
        tr = (cand.fold != f).to_numpy(); te = ~tr
        ms, _ = ranker.fit_bag(cand[tr], feats=feats, rounds=rounds) if params is None else \
            ([ranker.fit(cand[tr], feats=feats, rounds=rounds, params=params)[0]], None)
        sc[te] = ranker.predict_any(ms, cand.loc[te, feats])
    return sc


def recall(lists):
    """lists: {qid: [code_idx,...]}"""
    lab = cand[cand.label == 1].set_index('qid').code_idx
    out = {}
    for k in (10, 20):
        hit = sum(1 for q in dx if q in lab.index and lab[q] in lists.get(q, [])[:k])
        out[f'@{k}'] = round(hit / nd, 4)
    return out


def ranked_lists(score_col):
    c = cand[cand.qid.isin(dxset)].sort_values(['qid', score_col], ascending=[True, False])
    return c.groupby('qid').code_idx.apply(list).to_dict()


res = {}
cand['sA'] = oof_scores(None)
LA = ranked_lists('sA'); res['A current (top-1 tuned, bagged)'] = recall(LA)
cand['sB'] = oof_scores(dict(eval_at=[20], lambdarank_truncation_level=50, label_gain=[0, 1]), rounds=600)
LB = ranked_lists('sB'); res['B recall-oriented LambdaRank'] = recall(LB)
cand['sBin'] = oof_scores(dict(objective='binary', metric='auc'), rounds=600)
res['B2 binary objective'] = recall(ranked_lists('sBin'))
# C: diversity fill
side = {}
for col in ('ce', 'dense_ft_sim', 'knn_ft_max'):
    side[col] = ranked_lists(col)
for K in (10, 20):
    LC = {}
    for q, base in LA.items():
        lst = base[:K - 6]
        for col in ('ce', 'dense_ft_sim', 'knn_ft_max'):
            added = 0
            for ci in side[col].get(q, []):
                if ci not in lst:
                    lst.append(ci); added += 1
                if added == 2:
                    break
        for ci in base:                      # top up to K if side lists ran short
            if len(lst) >= K:
                break
            if ci not in lst:
                lst.append(ci)
        LC[q] = lst[:K]
    r = recall(LC)
    res[f'C diverse list (K={K})'] = r[f'@{K}']
for k, v in res.items():
    log(f'{k:38s} {v}')
json.dump(res, open('results/exp24_recall_lists.json', 'w'), indent=1)
