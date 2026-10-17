"""Exp17: label-ambiguity features — learn the coders' preferred variant among near-synonym candidates.

Over each term's top-10 OOF R1 candidates (cache/stages_<system>.parquet):
  twin_n          – # other top-10 candidates whose display name is near-identical (token-set ratio >= 90)
  prior_vs_twins  – this code's training frequency (log, fold-aware) minus the best twin's   (coder preference)
  tag_vs_twins    – P(this type | subconcept) minus the best twin's                         (type convention)
  ftsim_vs_twins  – fine-tuned dense similarity minus the best twin's
Majority convention (fold-aware): among training terms whose normalized wording is identical (sorted content tokens),
  maj_share – share that used this code; maj_n – how many such terms.
Evaluated as R1 + features vs R1 (same full candidate table, 5-fold CV).
Usage: python exp17_ambiguity_features.py [systems]
"""
import sys, os, re, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from rapidfuzz import fuzz
from src.dicts import load_train, load_system, SYSTEMS, CACHE
from src.cv import assign_folds, N_FOLDS
from src.engine import log
from src.evaluate import cv_rerank
from src.normalize import clean, clean_desc

systems = sys.argv[1].split(',') if len(sys.argv) > 1 else ['SNOMEDCT_US', 'ICD10CM']
STOP = r'\b(?:the|a|an|of|and|with|in|on|to|is|are|has|have|patient|pt|mild|moderate|severe|likely|possible|noted)\b'
df = load_train(); df['fold'] = assign_folds(df)


def key(e):
    t = re.sub(STOP, ' ', clean(e)); t = re.sub(r'[^a-z0-9 ]', ' ', t)
    return ' '.join(sorted(t.split()))


summary = {}
for s in systems:
    cs = load_system(s)
    d = df[df.coding_system == s].reset_index(drop=True)
    cand = pd.concat([pd.read_parquet(os.path.join(CACHE, f'oofpart_ftsap2_{s}_f{f}.parquet')) for f in range(N_FOLDS)],
                     ignore_index=True)
    old = pd.concat([pd.read_parquet(os.path.join(CACHE, f'oofpart_ftsap_{s}_f{f}.parquet')) for f in range(N_FOLDS)],
                    ignore_index=True)
    cols = [c for c in old.columns if c.startswith(('dense_ft_', 'knn_ft_'))]
    cand = cand.merge(old[['qid', 'code_idx'] + cols].rename(columns={c: c.replace('_ft_', '_ft1_') for c in cols}),
                      on=['qid', 'code_idx'], how='left'); del old
    st = pd.read_parquet(os.path.join(CACHE, f'stages_{s}.parquet'), columns=['qid', 'code_idx', 'r1'])
    cand = cand.merge(st, on=['qid', 'code_idx'], how='left')
    top = cand[cand.r1 < 10][['qid', 'code_idx', 'code_prior', 'dense_ft_sim'] +
                             (['p_grp_sub'] if 'p_grp_sub' in cand else [])].copy()
    names = {ci: clean_desc(str(cs.name_of[ci])) for ci in top.code_idx.unique()}
    feats = {k: np.full(len(top), np.nan, np.float32) for k in ('twin_n', 'prior_vs_twins', 'tag_vs_twins', 'ftsim_vs_twins')}
    pos = 0
    rows_index = []
    for q, g in top.groupby('qid', sort=False):
        ci = g.code_idx.to_numpy(); nm = [names[c] for c in ci]
        pr = g.code_prior.to_numpy(); ds = g.dense_ft_sim.to_numpy()
        pg = g.p_grp_sub.to_numpy() if 'p_grp_sub' in g else np.zeros(len(g))
        for i in range(len(g)):
            tw = [j for j in range(len(g)) if j != i and fuzz.token_set_ratio(nm[i], nm[j]) >= 90]
            feats['twin_n'][pos + i] = len(tw)
            if tw:
                feats['prior_vs_twins'][pos + i] = pr[i] - pr[tw].max()
                feats['tag_vs_twins'][pos + i] = pg[i] - pg[tw].max()
                feats['ftsim_vs_twins'][pos + i] = ds[i] - ds[tw].max()
        rows_index.append(g.index.to_numpy()); pos += len(g)
    idx = np.concatenate(rows_index)
    for k, v in feats.items():
        cand[k] = np.nan
        cand.loc[idx, k] = v
    # majority convention among identically-worded training terms (fold-aware)
    d['k'] = d.entity.map(key)
    gold = d.code.map(cs.code_to_idx).fillna(-1).astype(int).to_numpy()
    share = {}
    for f in range(N_FOLDS):
        mem = d[(d.fold != f)]
        tab = mem.groupby('k').apply(lambda x: pd.Series(gold[x.index]).value_counts(normalize=True).to_dict())
        cnt = mem.groupby('k').size()
        for q in np.where(d.fold.to_numpy() == f)[0]:
            kk = d.k[q]
            if kk in tab.index:
                share[q] = (tab[kk], int(cnt[kk]))
    qa = cand.qid.to_numpy(); ca = cand.code_idx.to_numpy()
    cand['maj_share'] = np.array([share[q][0].get(c, 0.0) if q in share else np.nan for q, c in zip(qa, ca)], np.float32)
    cand['maj_n'] = np.array([share[q][1] if q in share else 0 for q in qa], np.float32)
    log(s, 'terms with identically-worded training terms:', len(share), '| top-10 candidates with a twin:',
        int((feats['twin_n'] > 0).sum()))
    cand = cand.drop(columns=['r1'])
    newf = ['twin_n', 'prior_vs_twins', 'tag_vs_twins', 'ftsim_vs_twins', 'maj_share', 'maj_n']
    m0, *_ = cv_rerank(cand.drop(columns=newf), d)
    m1, pred, model, fts, _ = cv_rerank(cand, d)
    log(s, 'R1', m0['acc'], '| R1 + ambiguity features', m1['acc'], m1)
    imp = pd.Series(model.feature_importance('gain'), index=fts)
    print(imp[newf].round(0).to_string(), flush=True)
    cand[['qid', 'code_idx'] + newf].to_parquet(os.path.join(CACHE, f'ambfeat_{s}.parquet'))
    summary[s] = dict(R1=m0, R1_amb=m1)
    del cand
json.dump(summary, open('results/exp17_ambiguity.json', 'w'), indent=1)
print(summary)
