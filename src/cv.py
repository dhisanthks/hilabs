"""Cross-validation folds: grouped by source chunk so a mention's ICD and SNOMED rows (and its
chunk-mates) never leak across folds."""
import numpy as np
from sklearn.model_selection import GroupKFold

N_FOLDS = 5


def assign_folds(df, n=N_FOLDS, seed=13):
    rng = np.random.RandomState(seed)
    docs = df["doc_id"].unique()
    perm = rng.permutation(len(docs))
    doc_fold = {d: perm[i] % n for i, d in enumerate(docs)}
    return df["doc_id"].map(doc_fold).to_numpy()
