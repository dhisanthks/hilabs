"""Exp15: two-level decision for the concept-type bottleneck (SNOMED semantic tag / ICD chapter).

Level 1 — type ranker (LightGBM LambdaRank over the *types* present in each term's candidate pool), features
aggregated per (term, type): max / top-3 mean of the OOF R1 score, max dense (fine-tuned, base) and char similarity,
#candidates, best R1 rank, kNN type vote from the 25 most similar training terms (fold-aware, base SapBERT
embeddings; generalizes to unseen codes), P(type | subconcept), type and subconcept identity.
Level 2 — code ranker R1' = R1 features + p_type2 (level-1 softmax prob of the candidate's type), type_rank2,
r1_in_type (candidate's rank by OOF R1 score within its own type).

Usage: python exp15_type_ranker.py [systems]   (needs cache/stages_<system>.parquet from exp14)
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src.dicts import load_train, load_system, SYSTEMS, CACHE
from src.cv import assign_folds, N_FOLDS
from src.engine import log
from src.evaluate import cv_rerank
from src import ranker

systems = sys.argv[1].split(',') if len(sys.argv) > 1 else ['SNOMEDCT_US', 'ICD10CM']
df = load_train(); df['fold'] = assign_folds(df)
E_all = np.load(os.path.join(CACHE, 'entity_emb_sapbert.npy')).astype(np.float32)
summary = {}
for s in systems:
    cs = load_system(s)
    m = (df.coding_system == s).to_numpy()
    d = df[m].reset_index(drop=True); E = E_all[m]
    grp_of = cs.tag_of if s == 'SNOMEDCT_US' else cs.meta['chapter'].to_numpy()
    gold_idx = d.code.map(cs.code_to_idx)
    gold_grp = np.where(gold_idx.notna(), grp_of[gold_idx.fillna(0).astype(int)], 'NA')
    # ---- full candidate table (same as final stack) + OOF R1 score
    cand = pd.concat([pd.read_parquet(os.path.join(CACHE, f'oofpart_ftsap2_{s}_f{f}.parquet')) for f in range(N_FOLDS)],
                     ignore_index=True)
    old = pd.concat([pd.read_parquet(os.path.join(CACHE, f'oofpart_ftsap_{s}_f{f}.parquet')) for f in range(N_FOLDS)],
                    ignore_index=True)
    cols = [c for c in old.columns if c.startswith(('dense_ft_', 'knn_ft_'))]
    cand = cand.merge(old[['qid', 'code_idx'] + cols].rename(columns={c: c.replace('_ft_', '_ft1_') for c in cols}),
                      on=['qid', 'code_idx'], how='left'); del old
    st = pd.read_parquet(os.path.join(CACHE, f'stages_{s}.parquet'))[['qid', 'code_idx', 's1']]
    cand = cand.merge(st, on=['qid', 'code_idx'], how='left')
    cand['grp'] = grp_of[cand.code_idx.to_numpy()]
    # ---- kNN type votes (fold-aware)
    vote = {}
    for f in range(N_FOLDS):
        te = np.where(d.fold.to_numpy() == f)[0]; mem = np.where(d.fold.to_numpy() != f)[0]
        S = E[te] @ E[mem].T
        nn = np.argpartition(-S, 25, axis=1)[:, :25]
        for i, q in enumerate(te):
            sims = S[i, nn[i]]; g = gold_grp[mem[nn[i]]]
            w = np.maximum(sims, 0)
            tot = w.sum() + 1e-6
            vote[q] = {t: float(w[g == t].sum() / tot) for t in np.unique(g)}
    # ---- level 1: per (term, type) table
    cand = cand.sort_values(['qid', 's1'], ascending=[True, False])
    cand['r1'] = cand.groupby('qid').cumcount()
    agg = cand.groupby(['qid', 'grp']).agg(
        s1_max=('s1', 'max'), s1_top3=('s1', lambda x: x.nlargest(3).mean()), n_cand=('s1', 'size'),
        best_r1=('r1', 'min'), dft_max=('dense_ft_sim', 'max'), dbase_max=('dense_base_sim', 'max'),
        char_max=('char_sim', 'max'), fold=('fold', 'first')).reset_index()
    agg['s1_gap'] = agg.s1_max - agg.groupby('qid').s1_max.transform('max')
    agg['knn_vote'] = [vote.get(q, {}).get(g, 0.0) for q, g in zip(agg.qid, agg.grp)]
    sub = d.subconcept.to_numpy()
    pt = pd.crosstab(d.subconcept, gold_grp, normalize='index')
    agg['p_grp_sub'] = [pt.at[sub[q], g] if (sub[q] in pt.index and g in pt.columns) else 0.0 for q, g in zip(agg.qid, agg.grp)]
    agg['grp_cat'] = pd.Categorical(agg.grp)
    agg['sub_cat'] = pd.Categorical(sub[agg.qid.to_numpy()])
    agg['label'] = (agg.grp.to_numpy() == gold_grp[agg.qid.to_numpy()]).astype(np.int8)
    agg['code_idx'] = agg['grp_cat'].cat.codes  # placeholder id for cv_rerank bookkeeping
    feats = ['s1_max', 's1_top3', 'n_cand', 'best_r1', 'dft_max', 'dbase_max', 'char_max', 's1_gap', 'knn_vote',
             'p_grp_sub', 'grp_cat', 'sub_cat']
    agg['score'] = 0.0
    for f in range(N_FOLDS):
        tr = (agg.fold != f).to_numpy()
        mdl, fts = ranker.fit(agg[tr], feats=feats)
        agg.loc[~tr, 'score'] = mdl.predict(agg.loc[~tr, fts])
    top = agg.sort_values(['qid', 'score'], ascending=[True, False]).groupby('qid').head(1)
    type_acc = float(top.label.sum() / len(d))
    r1_type = cand[cand.r1 == 0]
    r1_type_acc = float((r1_type.grp.to_numpy() == gold_grp[r1_type.qid.to_numpy()]).sum() / len(d))
    log(s, 'type accuracy: R1 argmax type', round(r1_type_acc, 4), '| type ranker', round(type_acc, 4),
        '| gold type in pool', round(float(agg.groupby('qid').label.max().mean()), 4))
    ex = np.exp(agg.score - agg.groupby('qid').score.transform('max'))
    agg['p_type2'] = ex / ex.groupby(agg.qid).transform('sum')
    agg['type_rank2'] = agg.groupby('qid').score.rank(ascending=False, method='first')
    cand = cand.merge(agg[['qid', 'grp', 'p_type2', 'type_rank2', 'knn_vote']], on=['qid', 'grp'], how='left')
    cand['r1_in_type'] = cand.groupby(['qid', 'grp']).cumcount()
    cand = cand.drop(columns=['grp', 's1'])
    newf = ['p_type2', 'type_rank2', 'knn_vote', 'r1_in_type']
    m0, *_ = cv_rerank(cand.drop(columns=newf + ['r1']), d)
    m0b, *_ = cv_rerank(cand.drop(columns=newf), d)          # ablation: stacked R1 rank only
    m1, pred, model, fts, _ = cv_rerank(cand, d)
    log(s, 'R1 baseline', m0['acc'], '| + R1 rank (stacking only)', m0b['acc'], '| + type-ranker features', m1['acc'], m1)
    imp = pd.Series(model.feature_importance('gain'), index=fts).sort_values(ascending=False)
    print(imp.head(12).round(0).to_string(), flush=True)
    cand[['qid', 'code_idx', 'p_type2', 'type_rank2', 'knn_vote', 'r1_in_type', 'r1']].to_parquet(
        os.path.join(CACHE, f'typefeat_{s}.parquet'))
    summary[s] = dict(r1_type_acc=r1_type_acc, type_ranker_acc=type_acc, R1=m0, R1_rankonly=m0b, R1_type=m1)
    del cand, agg
json.dump(summary, open(f'results/exp15_type_ranker.json', 'w'), indent=1)
print(summary)
