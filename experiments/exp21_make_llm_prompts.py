"""Exp21: build the LLM-reranker dataset (for the cloud GPU job in kaggle/).

For every training term: the current best stack's top-10 candidates (out-of-fold R2-level ranker over the final R1
top-50, with cross-encoder v2 + context features), plus term, chunk context, subconcept, coding system, fold and the
gold position (-1 if gold is outside the top-10). Written to kaggle/data/llm_prompts.jsonl.
Usage: python exp21_make_llm_prompts.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, lightgbm as lgb
from src.dicts import load_train, load_system, SYSTEMS, CACHE, ROOT
from src.cv import assign_folds, N_FOLDS
from src.engine import log
from src.crossenc import context_window
from src.normalize import clean
from src import ranker

OUT = os.path.join(ROOT, 'kaggle', 'data'); os.makedirs(OUT, exist_ok=True)
K = 10
df = load_train(); df['fold'] = assign_folds(df)
records = []
for s in SYSTEMS:
    cs = load_system(s)
    d = df[df.coding_system == s].reset_index(drop=True)
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
    sc = np.zeros(len(cand))
    for f in range(N_FOLDS):
        tr = (cand.fold != f).to_numpy(); te = ~tr
        ms, _ = ranker.fit_bag(cand[tr], feats=feats)
        sc[te] = ranker.predict_any(ms, cand.loc[te, feats])
    cand['score'] = sc
    cand = cand.sort_values(['qid', 'score'], ascending=[True, False])
    cand['rk'] = cand.groupby('qid').cumcount()
    top = cand[cand.rk < K]
    acc1 = top[top.rk == 0].label.sum() / len(d)
    inK = top.groupby('qid').label.max().sum() / len(d)
    log(s, f'stack top-1 {acc1:.4f} | gold in top-{K} {inK:.4f}')
    for q, g in top.groupby('qid'):
        g = g.sort_values('rk')
        ci = g.code_idx.to_numpy()
        texts = [f"{cs.codes[c]} {cs.name_of[c]}" if s == 'ICD10CM' else
                 (f"{cs.name_of[c]} ({cs.tag_of[c]})" if cs.tag_of[c] else str(cs.name_of[c])) for c in ci]
        lab = g.label.to_numpy()
        records.append(dict(qid=int(q), system=s, fold=int(d.fold[q]), term=clean(d.entity[q]),
                            context=context_window(d.entity[q], d.chunk_text[q], 150), subconcept=d.subconcept[q],
                            cand_codes=[str(cs.codes[c]) for c in ci], cand_text=texts,
                            gold=int(np.argmax(lab)) if lab.max() > 0 else -1,
                            stack_scores=[round(float(x), 5) for x in g.score]))
with open(os.path.join(OUT, 'llm_prompts.jsonl'), 'w', encoding='utf-8') as fh:
    for r in records:
        fh.write(json.dumps(r, ensure_ascii=False) + '\n')
log('wrote', len(records), 'records to', os.path.join(OUT, 'llm_prompts.jsonl'))
