"""Exp18: chunk-context features for the ranker (the ranker previously saw no chart context).

Per term (fold-aware, out-of-fold):
  p_type_ctx      – P(candidate's type | entity char-ngrams + subconcept + chunk words) from a logistic regression
                    (type = SNOMED semantic tag / ICD chapter); type_ctx_top = candidate's type is the argmax
  ch_* flags      – chunk style: table markup, numeric density, units, lab / exercise / history / assessment /
                    device / social vocabulary, chunk length
Saved as cache/ctxfeat_<system>.parquet (qid, code_idx, ...) for exp13_stack.py --extra ctxfeat.
Usage: python exp18_context_features.py [systems]
"""
import sys, os, re, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from src.dicts import load_train, load_system, SYSTEMS, CACHE
from src.cv import assign_folds, N_FOLDS
from src.engine import log
from src.normalize import clean

systems = sys.argv[1].split(',') if len(sys.argv) > 1 else list(SYSTEMS)
VOCAB = {
    'ch_lab': r'\b(?:serum|plasma|blood|urine|result|reference|range|flag|abnormal|high|low|mg/dl|mmol|g/dl|u/l|meq|cbc|bmp|cmp|panel|specimen|lab)\b',
    'ch_exercise': r'\b(?:reps?|sets?|x\s?\d+|sec|seconds|minutes|theraband|exercise|stretch|strength|rom|gait|ambulat\w*|transfer\w*|pt|ot)\b',
    'ch_history': r'\b(?:history|hx|pmh|psh|past|prior|surgical|family|fhx)\b',
    'ch_assess': r'\b(?:assessment|plan|diagnos\w*|impression|problem|a/p|icd|chronic|acute)\b',
    'ch_device': r'\b(?:device|equipment|dme|walker|cane|wheelchair|pump|monitor|meter|lancet|pen|needle|cpap|oxygen)\b',
    'ch_social': r'\b(?:lives|living|spouse|alone|home|stairs|tobacco|alcohol|smok\w*|drinks|employ\w*|education)\b',
    'ch_procedure': r'\b(?:procedure|performed|surgery|operation|ultrasound|ct|mri|x-?ray|echo|ekg|injection|biopsy|scan)\b',
    'ch_vitals': r'\b(?:bp|pulse|temp|spo2|resp|weight|height|bmi|hr)\b',
}


def chunk_clean(c):
    return re.sub(r'header_\d+:|::|<br>', ' ', str(c)).lower()


df = load_train(); df['fold'] = assign_folds(df)
for s in systems:
    cs = load_system(s)
    d = df[df.coding_system == s].reset_index(drop=True)
    grp_of = cs.tag_of if s == 'SNOMEDCT_US' else cs.meta['chapter'].to_numpy()
    gi = d.code.map(cs.code_to_idx)
    y = np.where(gi.notna(), grp_of[gi.fillna(0).astype(int)], 'NA')
    ent = d.entity.map(clean); ch = d.chunk_text.map(chunk_clean)
    classes = np.array(sorted(set(grp_of)))
    col = {c: i for i, c in enumerate(classes)}
    P = np.zeros((len(d), len(classes)), np.float32)
    for f in range(N_FOLDS):
        tr, te = (d.fold != f).to_numpy(), (d.fold == f).to_numpy()
        parts = [TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2, sublinear_tf=True).fit(ent[tr]).transform(ent),
                 TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30000, sublinear_tf=True).fit(ch[tr]).transform(ch)]
        if s == 'SNOMEDCT_US':
            parts.append(OneHotEncoder(handle_unknown='ignore').fit(d[['subconcept']][tr]).transform(d[['subconcept']]) * 3)
        X = sp.hstack(parts).tocsr()
        clf = LogisticRegression(C=4, max_iter=400, random_state=7).fit(X[tr], y[tr])
        pr = clf.predict_proba(X[te])
        for j, c in enumerate(clf.classes_):
            if c in col:
                P[te, col[c]] = pr[:, j]
    log(s, 'context type classifier OOF accuracy', round(float((classes[P.argmax(1)] == y).mean()), 4))
    flags = pd.DataFrame({k: ch.str.count(v).clip(upper=20).astype(np.float32) for k, v in VOCAB.items()})
    flags['ch_table'] = d.chunk_text.str.contains('header_').astype(np.float32)
    flags['ch_digits'] = ch.str.count(r'\d').astype(np.float32) / (ch.str.len() + 1)
    flags['ch_len'] = np.log1p(ch.str.len()).astype(np.float32)
    # map onto candidate table keys
    keys = pd.concat([pd.read_parquet(os.path.join(CACHE, f'oofpart_ftsap2_{s}_f{f}.parquet'), columns=['qid', 'code_idx'])
                      for f in range(N_FOLDS)], ignore_index=True)
    ctype = np.array([col[g] for g in grp_of])[keys.code_idx.to_numpy()]
    q = keys.qid.to_numpy()
    out = keys.copy()
    out['p_type_ctx'] = P[q, ctype]
    out['type_ctx_top'] = (P.argmax(1)[q] == ctype).astype(np.int8)
    for c in flags.columns:
        out[c] = flags[c].to_numpy()[q]
    out.to_parquet(os.path.join(CACHE, f'ctxfeat_{s}.parquet'))
    log(s, 'saved', out.shape)
