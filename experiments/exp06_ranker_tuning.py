"""Exp06: reranker hyper-parameter / objective comparison on a saved OOF candidate table."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from src.dicts import load_train
from src.cv import assign_folds
from src.evaluate import cv_rerank
run, sysn = sys.argv[1], sys.argv[2]
df = load_train(); df['fold'] = assign_folds(df)
d = df[df.coding_system == sysn].reset_index(drop=True)
cand = pd.read_parquet(f'cache/oof_{run}_{sysn}.parquet')
grid = [
    ('base lambdarank 600', {}, 600),
    ('lambdarank lr.03 1000', dict(learning_rate=0.03), 1000),
    ('lambdarank leaves63', dict(num_leaves=63, min_data_in_leaf=10), 600),
    ('lambdarank leaves15', dict(num_leaves=15), 800),
    ('binary', dict(objective='binary', metric='auc'), 600),
    ('xendcg', dict(objective='rank_xendcg'), 600),
    ('lambdarank trunc5', dict(lambdarank_truncation_level=5), 600),
]
for name, p, r in grid:
    m, *_ = cv_rerank(cand, d, params=p, rounds=r)
    print(sysn, name, m, flush=True)
