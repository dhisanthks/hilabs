"""Exp16: cross-encoder v2 — retrained on the FINAL candidate tables (hard negatives = OOF R1's top wrong candidates
from cache/stages_<system>.parquet), 2 epochs; scores the final R1 top-30 for every term (100% coverage).
Output: cache/ce_scores_<name>_<system>.parquet (qid, code_idx, fold, ce) — consumed by exp13_stack.py and pipeline.

Usage: python exp16_crossenc_v2.py <name=cectx2> [--epochs 2] [--neg 9] [--max-len 96] [--topk 30]
"""
import sys, os, gc, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch
from src.dicts import load_train, load_system, SYSTEMS, CACHE
from src.cv import assign_folds, N_FOLDS
from src.engine import log
from src.crossenc import CrossEncoder, cand_text, context_window, make_query
from src.normalize import clean

ap = argparse.ArgumentParser()
ap.add_argument('name', nargs='?', default='cectx2'); ap.add_argument('--epochs', type=int, default=2)
ap.add_argument('--neg', type=int, default=9); ap.add_argument('--max-len', type=int, default=96)
ap.add_argument('--topk', type=int, default=30); ap.add_argument('--gpb', type=int, default=6)
a = ap.parse_args()

df = load_train(); df['fold'] = assign_folds(df)
X = {}
for s in SYSTEMS:
    d = df[df.coding_system == s].reset_index(drop=True)
    cs = load_system(s)
    st = pd.read_parquet(os.path.join(CACHE, f'stages_{s}.parquet'), columns=['qid', 'code_idx', 'label', 'fold', 'r1'])
    X[s] = dict(d=d, cs=cs, st=st[st.r1 < max(a.topk, a.neg + 5)],
                q=[make_query(clean(e), None, context_window(e, c)) for e, c in zip(d.entity, d.chunk_text)],
                gold=d.code.map(cs.code_to_idx).fillna(-1).astype(int).to_numpy())
parts = {s: [] for s in SYSTEMS}
for f in range(N_FOLDS):
    part_path = {s: os.path.join(CACHE, f'cepart_{a.name}_{s}_f{f}.parquet') for s in SYSTEMS}
    if all(os.path.exists(p) for p in part_path.values()):
        for s in SYSTEMS:
            parts[s].append(pd.read_parquet(part_path[s]))
        continue
    groups = []
    for s, x in X.items():
        st, d = x['st'], x['d']
        negs = st[(st.fold != f) & (st.label == 0)].sort_values(['qid', 'r1']).groupby('qid').code_idx.apply(list)
        for qi in np.where((d.fold != f).to_numpy() & (x['gold'] >= 0))[0]:
            ng = negs.get(qi, [])[:a.neg]
            if ng:
                groups.append((x['q'][qi], [cand_text(x['cs'], x['gold'][qi])] + [cand_text(x['cs'], k) for k in ng]))
    log(f'fold {f}: {len(groups)} groups, {a.epochs} epochs')
    ce = CrossEncoder('sapbert', max_len=a.max_len).fit(groups, epochs=a.epochs, group_size=a.neg + 1,
                                                        groups_per_batch=a.gpb, log_every=500)
    for s, x in X.items():
        te = x['st'][(x['st'].fold == f) & (x['st'].r1 < a.topk)]
        sc = ce.score([x['q'][i] for i in te.qid], [cand_text(x['cs'], k) for k in te.code_idx])
        p = te[['qid', 'code_idx', 'fold']].assign(ce=sc)
        p.to_parquet(part_path[s]); parts[s].append(p)
    del ce; gc.collect(); torch.cuda.empty_cache()
for s in SYSTEMS:
    k = pd.concat(parts[s], ignore_index=True)
    k.to_parquet(os.path.join(CACHE, f'ce_scores_{a.name}_{s}.parquet'))
    st = X[s]['st']; m = st[st.r1 < a.topk].merge(k, on=['qid', 'code_idx'])
    top = m.sort_values(['qid', 'ce'], ascending=[True, False]).groupby('qid').head(1)
    log(s, 'cross-encoder v2 alone (top-30) acc', round(float(top.label.sum() / len(X[s]['d'])), 4))
