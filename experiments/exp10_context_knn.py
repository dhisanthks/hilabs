"""Exp10: context-kNN features (idea from MedCAT: concept context embeddings).

For each mention, embed the text *around* the entity in its chunk (entity removed). For each candidate code
seen in the training folds, compare with the contexts of training mentions coded with it:
  ctx_max  – max cosine to a training context with this code
  ctx_mean – mean cosine over this code's training contexts
  ctx_has  – candidate code has training contexts at all
Also: ent_ctx_max – similarity of (entity + context) embeddings to training (entity + context) of the code.
Fold-aware (memory = other folds). Usage: CUDA_VISIBLE_DEVICES= python exp10_context_knn.py <cand_run>
"""
import sys, os, re, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src.dicts import load_train, load_system, SYSTEMS, CACHE
from src.cv import assign_folds, N_FOLDS
from src.normalize import clean
from src.engine import log
from src.evaluate import cv_rerank


def ctx_only(entity, chunk, width=150):
    if not isinstance(chunk, str) or not chunk:
        return ""
    ch = re.sub(r"header_\d+:|<br>|::", " ", chunk)
    ch = re.sub(r"\s+", " ", ch).strip().lower()
    e = entity.lower().strip()
    i = ch.find(e[:30])
    if i < 0:
        return clean(ch[:2 * width])
    return clean(ch[max(0, i - width):i] + " " + ch[i + len(e):i + len(e) + width])


run = sys.argv[1]
df = load_train(); df['fold'] = assign_folds(df)
path = os.path.join(CACHE, 'ctx_emb_sapbert.npy')
path2 = os.path.join(CACHE, 'entctx_emb_sapbert.npy')
if os.path.exists(path) and os.path.exists(path2):
    C_all, EC_all = np.load(path), np.load(path2)
else:
    from src.encoders import Encoder
    enc = Encoder('sapbert', max_len=96)
    ctx = [ctx_only(e, c) for e, c in zip(df.entity, df.chunk_text)]
    C_all = enc.encode(ctx, batch_size=64); np.save(path, C_all)
    EC_all = enc.encode([clean(e) + " | " + c for e, c in zip(df.entity, ctx)], batch_size=64); np.save(path2, EC_all)
    log('encoded contexts')
summary = {}
for s in (sys.argv[2:] or SYSTEMS):
    cs = load_system(s)
    m = (df.coding_system == s).to_numpy()
    d = df[m].reset_index(drop=True)
    C = C_all[m].astype(np.float32); EC = EC_all[m].astype(np.float32)
    has_ctx = (d.chunk_text.str.len() > 0).to_numpy()
    codes = d.code.map(cs.code_to_idx).fillna(-1).astype(int).to_numpy()
    cand = pd.read_parquet(os.path.join(CACHE, f'oof_{run}_{s}.parquet'))
    cmax = np.zeros(len(cand), np.float32); cmean = np.zeros(len(cand), np.float32)
    chas = np.zeros(len(cand), np.int8); ecmax = np.zeros(len(cand), np.float32)
    for f in range(N_FOLDS):
        mem = np.where((d.fold != f).to_numpy() & has_ctx & (codes >= 0))[0]
        te_mask = (cand.fold == f).to_numpy()
        sub = cand.loc[te_mask, ['qid', 'code_idx']]
        q_rows = np.sort(sub.qid.unique())
        qpos = {q: i for i, q in enumerate(q_rows)}
        S = C[q_rows] @ C[mem].T
        S2 = EC[q_rows] @ EC[mem].T
        mem_codes = codes[mem]
        by_code = pd.Series(np.arange(len(mem))).groupby(mem_codes).apply(np.array).to_dict()
        idx_te = np.where(te_mask)[0]
        sub_codes = sub.code_idx.to_numpy(); sub_q = np.array([qpos[q] for q in sub.qid])
        order = np.argsort(sub_codes, kind='stable')
        sc = sub_codes[order]
        bounds = np.flatnonzero(np.diff(sc)) + 1
        for grp in np.split(order, bounds):
            c = sub_codes[grp[0]]
            cols = by_code.get(c)
            if cols is None:
                continue
            rows = sub_q[grp]
            block = S[np.ix_(rows, cols)]; block2 = S2[np.ix_(rows, cols)]
            tgt = idx_te[grp]
            cmax[tgt] = block.max(1); cmean[tgt] = block.mean(1); chas[tgt] = 1; ecmax[tgt] = block2.max(1)
        log(s, 'fold', f, 'done')
    # rows without any context get 0 + flag
    cand['ctx_max'] = cmax; cand['ctx_mean'] = cmean; cand['ctx_has'] = chas; cand['entctx_max'] = ecmax
    cand['ctx_max_gap'] = cand.ctx_max - cand.groupby('qid').ctx_max.transform('max')
    cand['entctx_gap'] = cand.entctx_max - cand.groupby('qid').entctx_max.transform('max')
    new = ['ctx_max', 'ctx_mean', 'ctx_has', 'entctx_max', 'ctx_max_gap', 'entctx_gap']
    m0, *_ = cv_rerank(cand.drop(columns=new), d)
    m1, pred1, model, fts, _ = cv_rerank(cand, d)
    log(s, 'without context-kNN', m0['acc'], '| with', m1['acc'], m1)
    imp = pd.Series(model.feature_importance('gain'), index=fts).sort_values(ascending=False)
    print(imp.head(12).round(0).to_string(), flush=True)
    cand[['qid', 'code_idx'] + new].to_parquet(os.path.join(CACHE, f'ctxfeat_{run}_{s}.parquet'))
    summary[s] = dict(without=m0, with_ctx=m1)
json.dump(summary, open(f'results/exp10_{run}_{"_".join(summary)}.json', 'w'), indent=1)
print(summary)
