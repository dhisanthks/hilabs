"""Post-hoc feature augmentation of saved candidate tables (keeps experiments in sync with engine changes)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src.dicts import load_system
from src.features import CodeText, specificity_features
from src.query import variants

_ct = {}


def add_specificity(cand, d, system):
    if 'sp_extra_n' in cand:
        return cand
    ct = _ct.get(system) or _ct.setdefault(system, CodeText(load_system(system)))
    q = np.sort(cand.qid.unique())
    vl = {qq: variants(d.entity[qq]) for qq in q}
    tmp = cand[['qid', 'code_idx']].rename(columns={'qid': 'row'}).reset_index(drop=True)
    sp = specificity_features(tmp, vl, ct)
    sp.index = cand.index
    return pd.concat([cand, sp], axis=1)
