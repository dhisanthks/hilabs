"""Exp05: per-fold contrastive fine-tuning of a bi-encoder, then candidates + reranker under CV.

Usage: python exp05_finetune_cv.py <run_name> <base_encoder> [--with-base] [--epochs N]
Hard negatives come from the lexical OOF candidate tables (exp04 'lex' run).
"""
import sys, os, gc, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch
from src.dicts import load_train, SYSTEMS, CACHE
from src.cv import assign_folds, N_FOLDS
from src.engine import Engine, log
from src.encoders import Encoder
from src.finetune import build_pairs, build_pairs_multi, finetune, iterative_hard_negatives, mine_hard_negatives
from src.evaluate import cv_rerank

ap = argparse.ArgumentParser()
ap.add_argument('run'); ap.add_argument('base')
ap.add_argument('--with-base', action='store_true')
ap.add_argument('--epochs', type=int, default=2)
ap.add_argument('--folds', default='0,1,2,3,4')
ap.add_argument('--iter-from', default=None, help='round-1 OOF run name for iterative hard negatives')
a = ap.parse_args()

df = load_train(); df['fold'] = assign_folds(df)
per_sys = {s: df[df.coding_system == s].reset_index(drop=True) for s in SYSTEMS}
hard = {}
for s in SYSTEMS:
    p = os.path.join(CACHE, f'oof_lex_{s}.parquet')
    if os.path.exists(p):
        hard.update(mine_hard_negatives(pd.read_parquet(p), per_sys[s], s))
log('hard negatives', len(hard))
hard_multi = None
if a.iter_from:
    oofs = {s: pd.concat([pd.read_parquet(os.path.join(CACHE, f'oofpart_{a.iter_from}_{s}_f{f}.parquet'),
                                          columns=['qid', 'code_idx', 'label', 'dense_ft_sim']) for f in range(N_FOLDS)])
            for s in SYSTEMS}
    hard_multi = iterative_hard_negatives(oofs, per_sys, k=3)
    log('iterative hard negatives for', len(hard_multi), 'terms')

base_enc = Encoder(a.base) if a.with_base else None
for f in [int(x) for x in a.folds.split(',')]:
    outp = {s: os.path.join(CACHE, f'oofpart_{a.run}_{s}_f{f}.parquet') for s in SYSTEMS}
    if all(os.path.exists(p) for p in outp.values()):
        continue
    tr = df[df.fold != f]
    pairs = build_pairs_multi(tr, hard_multi) if hard_multi else build_pairs(tr, hard)
    log(f'fold {f}: {len(pairs)} pairs; fine-tuning {a.base}')
    st = finetune(a.base, pairs, epochs=a.epochs)
    encs = {'ft': Encoder.from_model(st)}
    if base_enc:
        encs['base'] = base_enc
    for s in SYSTEMS:
        d = per_sys[s]
        eng = Engine(s, encs, cache_dense=False)
        if base_enc:  # base encoder dictionary embeddings are fold-independent → cache them
            from src.engine import DenseIndex
            eng.dense['base'] = DenseIndex(eng.cs, base_enc, a.base, cache=True)
        eng.set_memory(d[d.fold != f])
        te = d[d.fold == f].reset_index()
        c = eng.candidates(te)
        c['qid'] = te['index'].to_numpy()[c['row'].to_numpy()]
        c['fold'] = f
        gold = d.code.map(eng.cs.code_to_idx).fillna(-1).astype(np.int64).to_numpy()
        c['label'] = (c.code_idx.to_numpy() == gold[c.qid.to_numpy()]).astype(np.int8)
        c.to_parquet(outp[s])
        del eng; gc.collect(); torch.cuda.empty_cache()
    del st, encs; gc.collect(); torch.cuda.empty_cache()

summary = {}
for s in SYSTEMS:
    parts = [os.path.join(CACHE, f'oofpart_{a.run}_{s}_f{f}.parquet') for f in range(N_FOLDS)]
    if not all(os.path.exists(p) for p in parts):
        continue
    cand = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    m, pred, model, feats, _ = cv_rerank(cand, per_sys[s])
    log(s, m)
    imp = pd.Series(model.feature_importance('gain'), index=feats).sort_values(ascending=False)
    print(imp.head(20).round(0).to_string())
    pred.to_parquet(os.path.join(CACHE, f'pred_{a.run}_{s}.parquet'))
    summary[s] = m
json.dump(summary, open(f'results/exp05_{a.run}.json', 'w'), indent=1)
print(summary)
