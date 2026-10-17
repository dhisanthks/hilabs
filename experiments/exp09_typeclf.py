"""Exp09: concept-type classifier → reranker feature.

Predicts the answer's *type* from the term alone: SNOMED semantic tag (finding / procedure / substance ...)
or ICD-10 chapter (R, M, I ...). Inputs: SapBERT embedding of the term, char n-grams of the term,
subconcept one-hot (if available). Multinomial logistic regression, out-of-fold (same 5 folds).
The reranker then gets p_type = P(candidate's type | term) and type_top = candidate's type is the argmax.

Usage: CUDA_VISIBLE_DEVICES= python exp09_typeclf.py <cand_run>
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from src.dicts import load_train, load_system, SYSTEMS, CACHE
from src.cv import assign_folds, N_FOLDS
from src.normalize import clean
from src.engine import log
from src.evaluate import cv_rerank

run = sys.argv[1]
df = load_train(); df['fold'] = assign_folds(df)
emb_path = os.path.join(CACHE, 'entity_emb_sapbert.npy')
if os.path.exists(emb_path):
    E_all = np.load(emb_path)
else:
    from src.encoders import Encoder
    E_all = Encoder('sapbert').encode(df.entity.map(clean).tolist(), batch_size=128)
    np.save(emb_path, E_all)
summary = {}
for s in SYSTEMS:
    cs = load_system(s)
    m = (df.coding_system == s).to_numpy()
    d = df[m].reset_index(drop=True)
    E = E_all[m].astype(np.float32)
    gidx = d.code.map(cs.code_to_idx)
    grp_of = cs.tag_of if s == 'SNOMEDCT_US' else cs.meta['chapter'].to_numpy()
    y = np.where(gidx.notna(), grp_of[gidx.fillna(0).astype(int)], 'NA')
    classes = np.array(sorted(set(grp_of)))
    P = np.zeros((len(d), len(classes)), np.float32)
    for f in range(N_FOLDS):
        tr, te = (d.fold != f).to_numpy(), (d.fold == f).to_numpy()
        tf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2, sublinear_tf=True)
        Xs = [sp.csr_matrix(E), None]
        Xc = tf.fit_transform(d.entity.map(clean))
        parts = [sp.csr_matrix(E * 2.0), Xc]
        if s == 'SNOMEDCT_US':
            oh = OneHotEncoder(handle_unknown='ignore').fit(d[['subconcept']])
            parts.append(oh.transform(d[['subconcept']]) * 3.0)
        X = sp.hstack(parts).tocsr()
        clf = LogisticRegression(C=4.0, max_iter=300, random_state=7)
        clf.fit(X[tr], y[tr])
        pr = clf.predict_proba(X[te])
        col = {c: i for i, c in enumerate(classes)}
        for j, c in enumerate(clf.classes_):
            if c in col:
                P[te, col[c]] = pr[:, j]
    pred = classes[P.argmax(1)]
    acc_type = float((pred == y).mean())
    log(s, 'type classifier OOF accuracy', round(acc_type, 4), '| majority baseline', round(pd.Series(y).value_counts(normalize=True).iloc[0], 4))
    np.save(os.path.join(CACHE, f'typeprob_{s}.npy'), P)
    cand = pd.read_parquet(os.path.join(CACHE, f'oof_{run}_{s}.parquet'))
    col = {c: i for i, c in enumerate(classes)}
    ctype = np.array([col[g] for g in grp_of])[cand.code_idx.to_numpy()]
    cand['p_type'] = P[cand.qid.to_numpy(), ctype]
    cand['type_top'] = (P.argmax(1)[cand.qid.to_numpy()] == ctype).astype(np.int8)
    m0, *_ = cv_rerank(cand.drop(columns=['p_type', 'type_top']), d)
    m1, pred1, *_ = cv_rerank(cand, d)
    log(s, 'reranker without type', m0['acc'], '| with type classifier', m1['acc'], m1)
    cand.to_parquet(os.path.join(CACHE, f'oof_{run}+type_{s}.parquet'))
    pred1.to_parquet(os.path.join(CACHE, f'pred_{run}+type_{s}.parquet'))
    summary[s] = dict(type_acc=acc_type, without=m0, with_type=m1)
json.dump(summary, open(f'results/exp09_{run}.json', 'w'), indent=1)
print(summary)
