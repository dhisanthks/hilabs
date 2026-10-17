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
for name, p, r in [('leaves63', dict(num_leaves=63, min_data_in_leaf=10), 600),
                   ('leaves127 lr.03', dict(num_leaves=127, min_data_in_leaf=10, learning_rate=0.03), 900),
                   ('leaves63 trunc30 lr.03', dict(num_leaves=63, min_data_in_leaf=10, learning_rate=0.03, lambdarank_truncation_level=30), 1000)]:
    m, *_ = cv_rerank(cand, d, params=p, rounds=r)
    print(sysn, name, m, flush=True)
