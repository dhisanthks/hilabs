"""Exp13: full stack on a candidate-table run: R1 → (+cross-encoder scores joined by (qid, code)) R2 → (+coupling) R3.

Usage: python exp13_stack.py <cand_run> [ce_run=cectx]
Writes results/exp13_<run>.json, cache/pred_<run>_final_<system>.parquet (for error analysis).
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch
from src.dicts import load_train, load_system, SYSTEMS, CACHE
from src.cv import assign_folds, N_FOLDS
from src.engine import log
from src.evaluate import cv_rerank
from src.coupling import pair_table, coupling_features
from src.normalize import clean_desc

run = sys.argv[1]
ce_run = sys.argv[2] if len(sys.argv) > 2 else 'cectx'
add_from = sys.argv[3] if len(sys.argv) > 3 else None   # e.g. ftsap: add round-1 encoder features as *_ft1_*
extra = [x for x in (sys.argv[4].split(',') if len(sys.argv) > 4 else []) if x]   # extra keyed feature tables, e.g. ctxfeat
df = load_train(); df['fold'] = assign_folds(df)


def name_emb(cs):
    path = os.path.join(CACHE, f'emb_names_sapbert_{cs.name}.npy')
    return torch.from_numpy(np.load(path)).to('cuda' if torch.cuda.is_available() else 'cpu')


def probs(c):
    ex = np.exp(c.score - c.groupby('qid').score.transform('max'))
    return ex / ex.groupby(c.qid).transform('sum')


X, summary = {}, {}
for s in SYSTEMS:
    d = df[df.coding_system == s].reset_index(drop=True)
    cand = pd.concat([pd.read_parquet(os.path.join(CACHE, f'oofpart_{run}_{s}_f{f}.parquet')) for f in range(N_FOLDS)],
                     ignore_index=True)
    if add_from:
        old = pd.concat([pd.read_parquet(os.path.join(CACHE, f'oofpart_{add_from}_{s}_f{f}.parquet')) for f in range(N_FOLDS)],
                        ignore_index=True)
        cols = [c for c in old.columns if c.startswith(('dense_ft_', 'knn_ft_'))]
        old = old[['qid', 'code_idx'] + cols].rename(columns={c: c.replace('_ft_', '_ft1_') for c in cols})
        cand = cand.merge(old, on=['qid', 'code_idx'], how='left')
        log(s, 'added', len(cols), 'round-1 encoder features; coverage', round(float(cand[cols[0].replace('_ft_', '_ft1_')].notna().mean()), 3))
        del old
    for name in extra:
        ef = pd.read_parquet(os.path.join(CACHE, f'{name}_{s}.parquet'))
        cand = cand.merge(ef, on=['qid', 'code_idx'], how='left')
        log(s, 'added extra features', name, ef.shape[1] - 2)
    m1, _, _, _, cand = cv_rerank(cand, d)
    log(s, 'R1', m1)
    cand = cand.sort_values(['qid', 'score'], ascending=[True, False])
    cand['r1'] = cand.groupby('qid').cumcount().astype(np.int16)
    cand[['qid', 'code_idx', 'label', 'fold', 'r1']].to_parquet(os.path.join(CACHE, f'stages_final_{s}.parquet'))
    ce = pd.read_parquet(os.path.join(CACHE, f'ce_scores_{ce_run}_{s}.parquet'))[['qid', 'code_idx', 'ce']]
    cand = cand.drop(columns=['score']).merge(ce, on=['qid', 'code_idx'], how='left')
    cand['ce_rank'] = cand.groupby('qid').ce.rank(ascending=False, method='first')
    cand['ce_gap'] = cand.ce - cand.groupby('qid').ce.transform('max')
    log(s, 'CE coverage of R1 top-30:', round(float(cand[cand.r1 < 30].ce.notna().mean()), 3))
    m2, pred2, _, _, cand = cv_rerank(cand, d)
    log(s, 'R2 (+CE)', m2)
    llm_dir = os.environ.get('LLM_DIR')
    if llm_dir:   # fine-tuned LLM reranker (out-of-fold scores from the cloud job) → R2+LLM
        from src.llm_rerank import load_oof_scores, add_llm_feats
        ls_ = load_oof_scores(llm_dir, s, load_system(s).code_to_idx)
        lp = cand[['qid', 'code_idx']].merge(ls_, on=['qid', 'code_idx'], how='left')['llm_lp'].to_numpy()
        cand = cand.assign(_r=cand.sort_values(['qid', 'score'], ascending=[True, False]).groupby('qid').cumcount())
        cand = add_llm_feats(cand.drop(columns=['score']), lp)
        log(s, 'LLM coverage of candidates in R2 top-10:',
            round(float(cand[cand._r < 10].llm_lp.notna().mean()), 3))
        cand = cand.drop(columns=['_r'])
        llm2_dir = os.environ.get('LLM2_DIR')
        if llm2_dir and s == 'SNOMEDCT_US':   # SNOMED diagnosis-specialist LLM (top-20)
            ls2 = load_oof_scores(llm2_dir, s, load_system(s).code_to_idx)
            lp2 = cand[['qid', 'code_idx']].merge(ls2, on=['qid', 'code_idx'], how='left')['llm_lp'].to_numpy()
            cand = add_llm_feats(cand, lp2, prefix='llm2')
            log(s, 'LLM-2 scored candidates:', int(np.isfinite(lp2).sum()))
        topllm = cand.dropna(subset=['llm_lp']).sort_values(['qid', 'llm_lp'], ascending=[True, False]).groupby('qid').head(1)
        log(s, 'LLM alone (its top-1 among scored):', round(float(topllm.label.sum() / len(d)), 4))
        m2, pred2, _, _, cand = cv_rerank(cand, d)
        log(s, 'R2+LLM', m2)
        dxm = (d.subconcept == 'DiagnosisCode').to_numpy()
        log(s, 'R2+LLM accuracy on DiagnosisCode terms:', round(float(pred2.ok.to_numpy()[dxm].mean()), 4))
    cand['prob'] = probs(cand)
    cand['mention_id'] = d.mention_id.to_numpy()[cand.qid.to_numpy()]
    top = cand.sort_values(['qid', 'prob'], ascending=[True, False]).groupby('qid').head(5)
    X[s] = dict(d=d, cand=cand.drop(columns=['score']), top=top[['mention_id', 'code_idx', 'prob', 'fold']], pred2=pred2,
                cs=load_system(s), m1=m1, m2=m2)

emb = {s: name_emb(X[s]['cs']) for s in SYSTEMS}
pairs = df.pivot_table(index='mention_id', columns='coding_system', values='code', aggfunc='first').dropna()
pairs['fold'] = df.groupby('mention_id').fold.first().reindex(pairs.index)
pairs = pairs.reset_index()
for s in SYSTEMS:
    o = [t for t in SYSTEMS if t != s][0]
    c = X[s]['cand']
    feats = [coupling_features(c[c.fold == f], s, X[o]['top'][X[o]['top'].fold == f], X[s]['cs'], X[o]['cs'],
                               emb[s], emb[o], pair_table(pairs[pairs.fold != f])) for f in range(N_FOLDS)]
    c = c.join(pd.concat(feats)).drop(columns=['prob', 'mention_id'])
    m3, pred, model, fts, _ = cv_rerank(c, X[s]['d'])
    log(s, 'R3 (+coupling)', m3)
    imp = pd.Series(model.feature_importance('gain'), index=fts).sort_values(ascending=False)
    print(imp.head(15).round(0).to_string(), flush=True)
    tagname = f'{run}{"+" + add_from if add_from else ""}_{ce_run}{"+" + "+".join(extra) if extra else ""}'
    pred.to_parquet(os.path.join(CACHE, f'pred_{tagname}_R3_{s}.parquet'))
    X[s]['pred2'].to_parquet(os.path.join(CACHE, f'pred_{tagname}_R2_{s}.parquet'))
    summary[s] = dict(R1=X[s]['m1'], R2=X[s]['m2'], R3=m3)
    del c
json.dump(summary, open(f'results/exp13_{run}{"+" + add_from if add_from else ""}_{ce_run}{"+" + "+".join(extra) if extra else ""}.json', 'w'), indent=1)
print(json.dumps(summary, indent=1))
