"""Exp01: lexical baselines. char-TF-IDF over dictionary descriptions; char-TF-IDF kNN over train terms."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.dicts import load_system, load_train, SYSTEMS
from src.normalize import clean, clean_desc
from src.cv import assign_folds

def topk_sparse(Q, D, k=50, bs=512):
    out_i, out_s = [], []
    DT = D.T.tocsr()
    for s in range(0, Q.shape[0], bs):
        S = (Q[s:s+bs] @ DT).toarray()
        idx = np.argpartition(-S, kth=min(k, S.shape[1]-1), axis=1)[:, :k]
        sc = np.take_along_axis(S, idx, 1)
        o = np.argsort(-sc, 1)
        out_i.append(np.take_along_axis(idx, o, 1)); out_s.append(np.take_along_axis(sc, o, 1))
    return np.vstack(out_i), np.vstack(out_s)

df = load_train(); df['fold'] = assign_folds(df); df['q'] = df.entity.map(clean)
res = []
for sysn in SYSTEMS:
    t = time.time()
    cs = load_system(sysn)
    d = df[df.coding_system == sysn].reset_index(drop=True)
    desc = cs.desc.description.map(clean_desc)
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2, sublinear_tf=True, dtype=np.float32)
    D = vec.fit_transform(desc); Q = vec.transform(d.q)
    idx, sc = topk_sparse(Q, D, k=200)
    code_idx = cs.desc.code_idx.to_numpy()[idx]
    gold = d.code.map(cs.code_to_idx).fillna(-1).to_numpy()
    # dedupe codes per row keeping best rank
    ranks = []
    for r in range(len(d)):
        seen = list(dict.fromkeys(code_idx[r]))
        ranks.append(seen.index(gold[r]) if gold[r] in seen else 999)
    ranks = np.array(ranks)
    for k in [1, 5, 10, 50, 100]:
        print(sysn, f'dict char-tfidf R@{k}', round((ranks < k).mean(), 4))
    # kNN over training terms (CV)
    pred = np.empty(len(d), dtype=object); simv = np.zeros(len(d))
    for f in range(5):
        tr, te = d.fold != f, d.fold == f
        v2 = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), sublinear_tf=True, dtype=np.float32)
        A = v2.fit_transform(d.q[tr]); B = v2.transform(d.q[te])
        i2, s2 = topk_sparse(B, A, k=1)
        pred[te.to_numpy()] = d.code[tr].to_numpy()[i2[:, 0]]; simv[te.to_numpy()] = s2[:, 0]
    knn_acc = (pred == d.code).mean()
    seen_codes = np.zeros(len(d), bool)
    for f in range(5):
        tr, te = d.fold != f, d.fold == f
        seen_codes[te.to_numpy()] = d.code[te].isin(set(d.code[tr])).to_numpy()
    print(sysn, 'kNN train-term acc', round(knn_acc, 4), '| gold code seen in train fold:', round(seen_codes.mean(), 4))
    dict_top = cs.codes[[code_idx[r][0] for r in range(len(d))]]
    for th in [0.5, 0.6, 0.7, 0.8, 0.9]:
        comb = np.where(simv >= th, pred, dict_top)
        print(sysn, f'combo th={th}', round((comb == d.code).mean(), 4))
    print(time.time() - t)
