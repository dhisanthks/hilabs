"""Joint ICD-10-CM ↔ SNOMED CT coupling (second stage).

For a mention that needs both codes, the first-stage prediction in one system is evidence for the
other. For each candidate of system A we add:
  cp_cooc      – P(candidate | other-system top-1) from training pairs (fold-aware), smoothed
  cp_cooc_w    – Σ_k P(other cand k) · P(candidate | k) over the other system's top-5
  cp_sim       – dense similarity between candidate name and other system's top-1 name
                 (generalizes to code pairs never seen together in training)
  cp_sim_w     – probability-weighted version over the other system's top-5
  cp_has       – whether the mention has an other-system prediction at all
Disagreement between the two systems therefore becomes a feature and a confidence signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def pair_table(train_pairs: pd.DataFrame):
    """train_pairs: columns mention_id, ICD10CM, SNOMEDCT_US (codes). → counts dict both directions."""
    c = train_pairs.groupby(["ICD10CM", "SNOMEDCT_US"]).size()
    tot_i = train_pairs.groupby("ICD10CM").size()
    tot_s = train_pairs.groupby("SNOMEDCT_US").size()
    return c, tot_i, tot_s


def coupling_features(cand_a, sys_a, top_b, cs_a, cs_b, name_emb_a, name_emb_b, pairs, alpha=1.0):
    """cand_a: candidate table of system A with 'mention_id'.
    top_b: DataFrame[mention_id, code_idx, prob] — other system's top-5 per mention (OOF).
    name_emb_*: torch fp16 [n_codes, dim] normalized embeddings of code display names (on GPU), or None."""
    c, tot_i, tot_s = pairs
    out = pd.DataFrame(index=cand_a.index)
    tb = top_b.sort_values(["mention_id", "prob"], ascending=[True, False])
    groups = {m: (g.code_idx.to_numpy(), g.prob.to_numpy()) for m, g in tb.groupby("mention_id")}
    a_codes = cs_a.codes; b_codes = cs_b.codes
    if sys_a == "ICD10CM":
        cnt = lambda a, b: c.get((a, b), 0); tot = lambda b: tot_s.get(b, 0)
    else:
        cnt = lambda a, b: c.get((b, a), 0); tot = lambda b: tot_i.get(b, 0)
    mids = cand_a["mention_id"].to_numpy(); cidx = cand_a["code_idx"].to_numpy()
    cooc = np.zeros(len(cand_a), np.float32); coocw = np.zeros(len(cand_a), np.float32)
    has = np.zeros(len(cand_a), np.int8)
    for i in range(len(cand_a)):
        g = groups.get(mids[i])
        if g is None:
            continue
        has[i] = 1
        ac = a_codes[cidx[i]]
        bcodes, bprob = g
        vals = []
        for bc_idx, p in zip(bcodes, bprob):
            bc = b_codes[bc_idx]
            vals.append(p * cnt(ac, bc) / (tot(bc) + alpha))
        cooc[i] = cnt(ac, b_codes[bcodes[0]]) / (tot(b_codes[bcodes[0]]) + alpha)
        coocw[i] = sum(vals)
    out["cp_cooc"] = cooc; out["cp_cooc_w"] = coocw; out["cp_has"] = has
    if name_emb_a is not None:
        sim1 = np.zeros(len(cand_a), np.float32); simw = np.zeros(len(cand_a), np.float32)
        # build per-row matrix of other-system top-5 (pad with -1)
        K = 5
        B = np.full((len(cand_a), K), -1, np.int64); P = np.zeros((len(cand_a), K), np.float32)
        for i in range(len(cand_a)):
            g = groups.get(mids[i])
            if g is None:
                continue
            k = min(K, len(g[0])); B[i, :k] = g[0][:k]; P[i, :k] = g[1][:k]
        dev = name_emb_a.device
        bs = 100_000
        for s in range(0, len(cand_a), bs):
            ea = name_emb_a[torch.from_numpy(cidx[s:s + bs]).to(dev)].float()
            bb = torch.from_numpy(np.maximum(B[s:s + bs], 0)).to(dev)
            eb = name_emb_b[bb].float()  # [n, K, d]
            sims = torch.einsum("nd,nkd->nk", ea, eb)
            sims = torch.where(torch.from_numpy(B[s:s + bs] >= 0).to(dev), sims, torch.zeros_like(sims))
            sim1[s:s + bs] = sims[:, 0].cpu().numpy()
            pw = torch.from_numpy(P[s:s + bs]).to(dev)
            simw[s:s + bs] = ((sims * pw).sum(1) / pw.sum(1).clamp_min(1e-6)).cpu().numpy()
        out["cp_sim"] = sim1; out["cp_sim_w"] = simw
    return out
