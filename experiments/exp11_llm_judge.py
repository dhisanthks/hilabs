"""Exp11: local LLM adjudication (Qwen3-4B-Instruct-2507 4-bit) over R2's top-10, with retrieval-augmented few-shot
examples from the training folds (fold-aware: examples never come from the evaluated term's fold).

Usage:
  python exp11_llm_judge.py <r2_table_run> --limit 300     # prototype on a random sample per system
  python exp11_llm_judge.py <r2_table_run>                 # all rows → cache/llm_<system>.parquet + stacking eval
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch
from src.dicts import load_train, load_system, SYSTEMS, CACHE
from src.cv import assign_folds, N_FOLDS
from src.engine import log
from src.evaluate import cv_rerank
from src.crossenc import context_window
from src.llm import LLMJudge

ap = argparse.ArgumentParser()
ap.add_argument('run'); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--k', type=int, default=10)
ap.add_argument('--shots', type=int, default=5); ap.add_argument('--systems', default=','.join(SYSTEMS))
ap.add_argument('--use-sub', action='store_true')
a = ap.parse_args()

df = load_train(); df['fold'] = assign_folds(df)
E_all = np.load(os.path.join(CACHE, 'entity_emb_sapbert.npy')).astype(np.float32)
judge = LLMJudge()
summary = {}
for s in a.systems.split(','):
    cs = load_system(s)
    m = (df.coding_system == s).to_numpy()
    d = df[m].reset_index(drop=True); E = E_all[m]
    cand = pd.read_parquet(os.path.join(CACHE, f'oof_{a.run}_{s}.parquet'))
    m2, pred2, _, _, cand = cv_rerank(cand.drop(columns=[c for c in cand.columns if c.startswith('llm_')]), d)
    log(s, 'R2', m2)
    cand = cand.sort_values(['qid', 'score'], ascending=[True, False])
    cand['r2'] = cand.groupby('qid').cumcount()
    top = cand[cand.r2 < a.k]
    qids = np.sort(top.qid.unique())
    if a.limit:
        qids = np.sort(np.random.RandomState(7).choice(qids, min(a.limit, len(qids)), replace=False))
    top = top[top.qid.isin(qids)]
    # few-shot neighbours from other folds
    shots = {}
    for f in range(N_FOLDS):
        qf = qids[d.fold.to_numpy()[qids] == f]
        mem = np.where(d.fold.to_numpy() != f)[0]
        if len(qf) == 0:
            continue
        S = E[qf] @ E[mem].T
        nn = np.argsort(-S, 1)[:, :a.shots]
        for i, q in enumerate(qf):
            shots[q] = [(d.entity[mem[j]][:60], (d.code[mem[j]] + ' ' if s == 'ICD10CM' else '') + d.display[mem[j]])
                        for j in nn[i]]
    def ctext(ci):
        n = cs.name_of[ci]
        return f"{cs.codes[ci]} {n}" if s == 'ICD10CM' else (f"{n} ({cs.tag_of[ci]})" if cs.tag_of[ci] else n)
    groups = top.groupby('qid')
    prompts, ncand, keys = [], [], []
    for q, g in groups:
        cl = [ctext(ci) for ci in g.code_idx]
        sub = d.subconcept[q] if a.use_sub else None
        prompts.append(judge.prompt(d.entity[q], context_window(d.entity[q], d.chunk_text[q], 150), s, cl, shots.get(q, []), sub))
        ncand.append(len(cl)); keys.append(g.index.to_numpy())
    t = time.time()
    lps = judge.score(prompts, ncand)
    log(s, f'LLM scored {len(prompts)} terms in {time.time() - t:.0f}s ({(time.time() - t) / max(len(prompts), 1):.2f}s/term)')
    cand['llm_lp'] = np.nan
    for idx, lp in zip(keys, lps):
        cand.loc[idx, 'llm_lp'] = lp
    sub = cand[cand.qid.isin(qids)]
    llm_top = sub.dropna(subset=['llm_lp']).sort_values(['qid', 'llm_lp'], ascending=[True, False]).groupby('qid').head(1)
    r2_top = sub[sub.r2 == 0]
    res = dict(n=len(qids), llm_alone=float(llm_top.label.sum() / len(qids)), r2=float(r2_top.label.sum() / len(qids)),
               oracle_topk=float(sub[sub.r2 < a.k].groupby('qid').label.max().mean()))
    log(s, res)
    if a.limit:
        sub.to_parquet(os.path.join(CACHE, f'llm_proto_{s}.parquet'))
    if not a.limit:
        cand['llm_rank'] = cand.groupby('qid').llm_lp.rank(ascending=False, method='first').fillna(99)
        cand['llm_gap'] = (cand.llm_lp - cand.groupby('qid').llm_lp.transform('max')).fillna(-99)
        cand['llm_lp'] = cand.llm_lp.fillna(-30)
        cand[['qid', 'code_idx', 'llm_lp', 'llm_rank', 'llm_gap']].to_parquet(os.path.join(CACHE, f'llm_{s}.parquet'))
        m3, pred3, model, fts, _ = cv_rerank(cand.drop(columns=['score', 'r2']), d)
        log(s, 'R2 + LLM features', m3)
        res['stacked'] = m3; res['r2_full'] = m2
        pred3.to_parquet(os.path.join(CACHE, f'pred_{a.run}+llm_{s}.parquet'))
        cand.drop(columns=['score', 'r2']).to_parquet(os.path.join(CACHE, f'oof_{a.run}+llm_{s}.parquet'))
    summary[s] = res
json.dump(summary, open(f'results/exp11_{a.run}{"_proto" if a.limit else ""}.json', 'w'), indent=1)
print(summary)
