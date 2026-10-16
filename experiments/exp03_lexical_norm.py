"""Exp03: does normalization (abbrev + spelling variants) raise lexical recall? char TF-IDF vs BM25."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src.dicts import load_system, load_train, SYSTEMS
from src.normalize import clean
from src.query import variants
from src.lexical import LexicalIndex

def code_scores(I, S, owner, desc_code, k=300):
    """aggregate description hits to per-row {code: max score} over variant rows."""
    out = {}
    for r in range(len(owner)):
        o = owner[r]; dct = out.setdefault(o, {})
        for di, s in zip(I[r], S[r]):
            c = desc_code[di]
            if s > dct.get(c, -1): dct[c] = s
    return out

df = load_train()
for sysn in SYSTEMS:
    cs = load_system(sysn); desc_code = cs.desc.code_idx.to_numpy()
    d = df[df.coding_system == sysn].reset_index(drop=True)
    gold = d.code.map(cs.code_to_idx).fillna(-1).astype(int).to_numpy()
    V = [variants(e) for e in d.entity]
    print(sysn, 'avg variants', np.mean([len(v) for v in V]))
    for kind in ['char', 'bm25']:
        t = time.time(); idx = LexicalIndex(cs, kind); print(kind, 'index', round(time.time() - t, 1))
        for mode in ['clean', 'variants']:
            qs, owner = [], []
            for r, v in enumerate(V):
                vs = [clean(d.entity[r])] if mode == 'clean' else v
                qs += vs; owner += [r] * len(vs)
            I, S = idx.search(qs, k=200)
            agg = code_scores(I, S, owner, desc_code)
            ranks = []
            for r in range(len(d)):
                items = sorted(agg[r].items(), key=lambda x: -x[1]); cl = [c for c, _ in items]
                ranks.append(cl.index(gold[r]) if gold[r] in cl else 999)
            ranks = np.array(ranks)
            print(sysn, kind, mode, {k: round(float((ranks < k).mean()), 4) for k in [1, 10, 50, 100]}, round(time.time() - t, 1), flush=True)
