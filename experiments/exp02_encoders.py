"""Exp02: benchmark off-the-shelf dense encoders for (a) dictionary retrieval recall and
(b) kNN over training terms. Same folds as exp01. Usage: python exp02_encoders.py sapbert biolord ..."""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch
from src.dicts import load_system, load_train, SYSTEMS
from src.normalize import clean, clean_desc
from src.cv import assign_folds
from src.encoders import Encoder, topk_dense


def code_ranks(desc_idx, cs_code_idx, gold):
    ranks = []
    for r in range(len(gold)):
        seen = list(dict.fromkeys(cs_code_idx[desc_idx[r]]))
        ranks.append(seen.index(gold[r]) if gold[r] in seen else 999)
    return np.array(ranks)


df = load_train(); df['fold'] = assign_folds(df); df['q'] = df.entity.map(clean)
out = {}
for name in sys.argv[1:]:
    t0 = time.time()
    enc = Encoder(name)
    res = {}
    for sysn in SYSTEMS:
        cs = load_system(sysn)
        d = df[df.coding_system == sysn].reset_index(drop=True)
        descs = cs.desc.description.map(clean_desc).tolist()
        t = time.time()
        D = enc.encode(descs, query=False, batch_size=1024)
        enc_time = time.time() - t
        Q = enc.encode(d.q.tolist())
        I, S = topk_dense(Q, D, k=300)
        del D; torch.cuda.empty_cache()
        gold = d.code.map(cs.code_to_idx).fillna(-1).astype(int).to_numpy()
        ranks = code_ranks(I, cs.desc.code_idx.to_numpy(), gold)
        r = {f'R@{k}': round(float((ranks < k).mean()), 4) for k in [1, 5, 10, 50, 100]}
        # kNN over training terms
        pred = np.empty(len(d), dtype=object)
        for f in range(5):
            tr, te = (d.fold != f).to_numpy(), (d.fold == f).to_numpy()
            i2, _ = topk_dense(Q[te], Q[tr], k=1)
            pred[te] = d.code.to_numpy()[tr][i2[:, 0]]
        r['kNN@1'] = round(float((pred == d.code).mean()), 4)
        r['enc_sec'] = round(enc_time, 1)
        res[sysn] = r
        print(name, sysn, r, flush=True)
    out[name] = res
    del enc; torch.cuda.empty_cache()
    print(name, 'total sec', round(time.time() - t0, 1), flush=True)
    with open(os.path.join('results', f'exp02_{name}.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
