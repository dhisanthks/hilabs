"""Contrastive fine-tuning of a bi-encoder on (entity → gold code description) pairs.

Loss: MultipleNegativesRankingLoss (in-batch negatives) + one mined hard negative per pair
(a lexically close but wrong code). One shared model for both coding systems.
Deterministic: fixed seeds, no data-order randomness beyond the seeded sampler.
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch

from .dicts import load_system
from .normalize import clean, clean_desc
from .query import variants


def set_seed(seed=7):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_pairs(train: pd.DataFrame, hard_neg: dict | None = None, max_desc=3, seed=7):
    """Rows of (anchor, positive, negative). anchor = entity variant; positive = a gold description.
    hard_neg: {(system, entity): code} lexically-close wrong code."""
    rng = random.Random(seed)
    out = []
    for sysn, d in train.groupby("coding_system"):
        cs = load_system(sysn)
        descs = cs.desc.assign(t=cs.desc.description.map(clean_desc)).groupby("code")["t"].apply(list)
        for e, code in zip(d["entity"], d["code"]):
            if code not in descs.index:
                continue
            pos_list = list(dict.fromkeys(descs[code]))[:max_desc]
            vs = variants(e)
            neg_code = hard_neg.get((sysn, e)) if hard_neg else None
            negs = descs.get(neg_code, []) if neg_code else []
            for p in pos_list:
                a = vs[0] if rng.random() < 0.6 or len(vs) == 1 else rng.choice(vs[1:])
                n = negs[0] if negs else None
                out.append((a, p, n))
    rng.shuffle(out)
    return out


def finetune(base: str, pairs, out_dir=None, epochs=2, batch_size=64, lr=2e-5, seed=7, max_len=48):
    from sentence_transformers import SentenceTransformer, losses, models
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from datasets import Dataset

    from .encoders import ENCODERS
    set_seed(seed)
    hf, pooling, _, _ = ENCODERS.get(base, (base, "mean", "", ""))
    if any(k in hf.lower() for k in ("sapbert", "medcpt")):
        word = models.Transformer(hf, max_seq_length=max_len)
        model = SentenceTransformer(modules=[word, models.Pooling(word.get_word_embedding_dimension(), pooling)],
                                    device="cuda")
    else:
        model = SentenceTransformer(hf, device="cuda")
    model.max_seq_length = max_len
    with_neg = [p for p in pairs if p[2]]
    ds = Dataset.from_dict({"anchor": [p[0] for p in with_neg], "positive": [p[1] for p in with_neg],
                            "negative": [p[2] for p in with_neg]})
    loss = losses.MultipleNegativesRankingLoss(model, scale=20.0)
    args = SentenceTransformerTrainingArguments(
        output_dir=out_dir or "cache/ft_tmp", num_train_epochs=epochs, per_device_train_batch_size=batch_size,
        learning_rate=lr, warmup_ratio=0.1, fp16=True, seed=seed, data_seed=seed, save_strategy="no",
        logging_steps=50, report_to=[], batch_sampler="no_duplicates", dataloader_num_workers=0)
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss)
    trainer.train()
    if out_dir:   # atomic save (temp dir + rename)
        import os, shutil
        tmp = out_dir + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        model.save(tmp)
        shutil.rmtree(out_dir, ignore_errors=True)
        os.replace(tmp, out_dir)
    return model


def mine_hard_negatives(oof_cand: pd.DataFrame, train_sys: pd.DataFrame, system: str, rank_col="char_sim"):
    """Highest-scoring wrong candidate per query from a candidate table (qid aligned with train_sys index)."""
    cs = load_system(system)
    c = oof_cand[oof_cand.label == 0].sort_values(["qid", rank_col], ascending=[True, False])
    top = c.groupby("qid").head(1)
    ents = train_sys["entity"].to_numpy()
    return {(system, ents[q]): cs.codes[ci] for q, ci in zip(top.qid, top.code_idx)}


def iterative_hard_negatives(oof_tables: dict, per_sys: dict, k=3, score_col="dense_ft_sim"):
    """Round-2 negatives: for each training term, the top-k *wrong* codes by the round-1 fine-tuned model's
    out-of-fold dense similarity. {(system, entity): [code, ...]}"""
    out = {}
    for s, c in oof_tables.items():
        cs = load_system(s)
        c = c[c.label == 0].sort_values(["qid", score_col], ascending=[True, False]).groupby("qid").head(k)
        ents = per_sys[s]["entity"].to_numpy()
        for q, g in c.groupby("qid"):
            out[(s, ents[q])] = [cs.codes[i] for i in g.code_idx]
    return out


def build_pairs_multi(train: pd.DataFrame, hard_multi: dict, seed=7):
    """One (anchor, positive, negative) triplet per hard negative (≤k per term); positives cycle through the
    gold code's first 3 descriptions; anchors mix original and normalized variants."""
    rng = random.Random(seed)
    out = []
    for sysn, d in train.groupby("coding_system"):
        cs = load_system(sysn)
        descs = cs.desc.assign(t=cs.desc.description.map(clean_desc)).groupby("code")["t"].apply(list)
        for e, code in zip(d["entity"], d["code"]):
            if code not in descs.index:
                continue
            pos = list(dict.fromkeys(descs[code]))[:3]
            vs = variants(e)
            for j, nc in enumerate(hard_multi.get((sysn, e), [])):
                if nc not in descs.index:
                    continue
                a = vs[0] if rng.random() < 0.6 or len(vs) == 1 else rng.choice(vs[1:])
                out.append((a, pos[j % len(pos)], descs[nc][0]))
    rng.shuffle(out)
    return out
