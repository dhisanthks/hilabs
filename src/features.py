"""Pairwise (query, candidate code) features for the learned reranker."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import torch
from rapidfuzz import fuzz

from .normalize import clean_desc

LAT = {"left": r"\b(left|lt|l)\b", "right": r"\b(right|rt|r)\b", "bilateral": r"\b(bilateral|bilat|b/l|bl|both|ble|bue)\b"}
QFLAGS = {
    "acute": r"\bacute\b", "chronic": r"\bchronic\b", "history": r"\b(history|h/o|hx|s/p|status post|prior)\b",
    "neg": r"\b(no|denies|negative|without|w/o|not)\b", "type1": r"\b(type 1|t1dm|dm1|iddm)\b",
    "type2": r"\b(type 2|t2dm|dm2|dmt2|niddm)\b", "severe": r"\bsevere\b", "mild": r"\bmild\b",
    "recurrent": r"\brecurrent\b", "exacerb": r"\bexacerbation\b", "pain": r"\b(pain|ache|aching|sore|soreness|tender)\b",
    "fracture": r"\b(fracture|fx)\b", "screen": r"\b(screen|screening)\b",
}
CFLAGS = dict(QFLAGS, unspecified=r"\bunspecified\b", other=r"^other\b", nos=r"\bnos\b",
              left=LAT["left"], right=LAT["right"], bilateral=LAT["bilateral"],
              with_=r"\bwith\b", without=r"\bwithout\b", due=r"\bdue to\b", encounter=r"\bencounter\b",
              history_of=r"\b(history of|h/o)\b")


class CodeText:
    """Per-code text resources: all cleaned descriptions + regex flags on the display name."""

    def __init__(self, cs):
        self.cs = cs
        desc = cs.desc.assign(t=cs.desc.description.map(clean_desc))
        g = desc.groupby("code_idx")["t"].apply(list)
        self.descs = g.reindex(range(len(cs))).to_numpy()
        self.n_desc = np.array([len(x) if isinstance(x, list) else 0 for x in self.descs], np.int16)
        names = pd.Series(cs.name_of).fillna("").str.lower()
        self.flags = pd.DataFrame({f"c_{k}": names.str.contains(v, regex=True).to_numpy() for k, v in CFLAGS.items()})
        self.name_len = names.str.split().str.len().to_numpy()
        # desc rows grouped by code (for dense max-over-synonyms)
        order = np.argsort(cs.desc.code_idx.to_numpy(), kind="stable")
        counts = np.bincount(cs.desc.code_idx.to_numpy(), minlength=len(cs))
        self.ptr = np.concatenate([[0], np.cumsum(counts)])
        self.rows_sorted = order


def query_flags(texts) -> pd.DataFrame:
    s = pd.Series(texts).str.lower()
    out = {f"q_{k}": s.str.contains(v, regex=True).to_numpy() for k, v in QFLAGS.items()}
    for k, v in LAT.items():
        out[f"q_{k}"] = s.str.contains(v, regex=True).to_numpy()
    out["q_ntok"] = s.str.split().str.len().to_numpy()
    out["q_nchar"] = s.str.len().to_numpy()
    out["q_hasdigit"] = s.str.contains(r"\d").to_numpy()
    return pd.DataFrame(out)


def fuzzy_features(cand: pd.DataFrame, var_lists, ct: CodeText) -> pd.DataFrame:
    """String similarity between the best query variant and the best description of each candidate."""
    rows = cand["row"].to_numpy(); codes = cand["code_idx"].to_numpy()
    n = len(cand)
    ratio = np.zeros(n, np.float32); tset = np.zeros(n, np.float32); psort = np.zeros(n, np.float32)
    jac = np.zeros(n, np.float32); cover = np.zeros(n, np.float32)
    cache_tok = {}
    for i in range(n):
        vs = var_lists[rows[i]]; ds = ct.descs[codes[i]]
        if not isinstance(ds, list):
            continue
        br = bt = bp = bj = bc = 0.0
        for v in vs:
            qt = cache_tok.get(v)
            if qt is None:
                qt = cache_tok[v] = set(v.split())
            for d in ds:
                r = fuzz.ratio(v, d)
                if r > br: br = r
                t = fuzz.token_set_ratio(v, d)
                if t > bt: bt = t
                p = fuzz.token_sort_ratio(v, d)
                if p > bp: bp = p
                dt = set(d.split())
                inter = len(qt & dt)
                if inter:
                    j = inter / len(qt | dt)
                    if j > bj: bj = j
                    c = inter / len(qt)
                    if c > bc: bc = c
        ratio[i], tset[i], psort[i], jac[i], cover[i] = br, bt, bp, bj, bc
    return pd.DataFrame({"fz_ratio": ratio, "fz_tset": tset, "fz_tsort": psort, "jaccard": jac, "q_cover": cover})


def dense_fill(cand, Qv, owner_rows, Dt: torch.Tensor, ct: CodeText, name="dense", bs=200_000):
    """Exact dense similarity for every candidate: max over query variants × code descriptions."""
    # variant embedding indices per row
    var_of_row = {}
    for vi, r in enumerate(owner_rows):
        var_of_row.setdefault(r, []).append(vi)
    rows = cand["row"].to_numpy(); codes = cand["code_idx"].to_numpy()
    pi, vi_l, di_l = [], [], []
    for i in range(len(cand)):
        d0, d1 = ct.ptr[codes[i]], ct.ptr[codes[i] + 1]
        drows = ct.rows_sorted[d0:d1]
        for v in var_of_row[rows[i]]:
            pi.extend([i] * len(drows)); vi_l.extend([v] * len(drows)); di_l.extend(drows)
    pi = np.asarray(pi, np.int64); vi_l = np.asarray(vi_l, np.int64); di_l = np.asarray(di_l, np.int64)
    dev = Dt.device
    Q = torch.from_numpy(Qv).to(dev)
    out = torch.full((len(cand),), -1.0, device=dev)
    for s in range(0, len(pi), bs):
        q = Q[torch.from_numpy(vi_l[s:s + bs]).to(dev)]
        d = Dt[torch.from_numpy(di_l[s:s + bs]).to(dev)]
        sims = (q.float() * d.float()).sum(1)
        out.scatter_reduce_(0, torch.from_numpy(pi[s:s + bs]).to(dev), sims, reduce="amax")
    return out.cpu().numpy()
