"""The term-mapping engine: candidate generation + features. Used identically by CV and inference.

Engine(system).set_memory(train_rows) installs the training "memory" (kNN terms + code priors);
Engine.candidates(query_rows) returns the candidate table with all reranker features.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import torch

from .candidates import char_knn_features, dense_candidates, knn_features, lexical_candidates, merge
from .dicts import CACHE, load_system
from .features import CodeText, dense_fill, fuzzy_features, query_flags, specificity_features
from .lexical import LexicalIndex
from .normalize import clean, clean_desc
from .query import variants

# SNOMED semantic tags that carry >0.5% of gold labels - retrieved separately so each concept type is in the pool
STRAT_TAGS = ["finding", "disorder", "procedure", "observable entity", "regime/therapy", "physical object",
              "qualifier value", "situation", "body structure", "substance", "environment", "event",
              "morphologic abnormality"]
SUBCONCEPTS = ["DiagnosisCode", "ProcedureCode", "LabService", "EncounterType", "Device", "MedicationProfileMedication"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class DenseIndex:
    """Dictionary description embeddings for one encoder (fp16, disk-cached). Kept in host memory and
    paged onto the GPU only while that encoder is being used, so several encoders fit a 4 GB card."""

    def __init__(self, cs, encoder, tag, cache=True):
        self.cs = cs
        path = os.path.join(CACHE, f"emb_{tag}_{cs.name}.npy")
        if not cache:
            D = encoder.encode(cs.desc.description.map(clean_desc).tolist(), query=False, batch_size=1024)
        elif os.path.exists(path):
            D = np.load(path, mmap_mode="r")   # memory-mapped: read from disk on demand, not committed to RAM
        else:
            t = time.time()
            D = encoder.encode(cs.desc.description.map(clean_desc).tolist(), query=False, batch_size=1024)
            np.save(path, D)
            log(f"encoded {len(D)} {cs.name} descriptions with {tag} in {time.time() - t:.0f}s")
        self._np = D
        self._gpu = None

    @property
    def D(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # torch warns that the memmap is read-only; it is never written
            return torch.from_numpy(np.asarray(self._np))

    @property
    def Dt(self):
        if self._gpu is None:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self._gpu = self.D.to(dev)
        return self._gpu

    def release(self):
        self._gpu = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class Engine:
    def __init__(self, system: str, encoders: dict | None = None, top_lex=60, top_dense=100, knn_k=25,
                 cache_dense=True):
        """encoders: {tag: Encoder} — every encoder yields dense_<tag> and knn_<tag> features."""
        self.system = system
        self.cs = load_system(system)
        self.ct = CodeText(self.cs)
        self.char = LexicalIndex(self.cs, "char")
        self.bm25 = LexicalIndex(self.cs, "bm25")
        self.encoders = encoders or {}
        self.dense = {tag: DenseIndex(self.cs, enc, tag, cache_dense) for tag, enc in self.encoders.items()}
        self.top_lex, self.top_dense, self.knn_k = top_lex, top_dense, knn_k
        self.meta = self.cs.meta
        self.groups = None
        self._build_groups()

    def _build_groups(self):
        """SNOMED tag-stratified retrieval groups (built once any dense encoder is attached)."""
        if self.groups is None and self.system == "SNOMEDCT_US" and self.encoders:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            tag_desc = self.cs.tag_of[self.cs.desc.code_idx.to_numpy()]
            self.groups = {t: torch.from_numpy(np.where(tag_desc == t)[0]).to(dev) for t in STRAT_TAGS}

    # ---------------------------------------------------------------- memory
    def set_memory(self, mem: pd.DataFrame):
        """mem: training rows of this system (entity, code, optional subconcept)."""
        mem = mem[mem["code"].isin(self.cs.code_to_idx)].reset_index(drop=True)
        self.mem = mem
        self.mem_text = mem["entity"].map(clean).tolist()
        self.mem_codes = mem["code"].map(self.cs.code_to_idx).to_numpy()
        self.mem_emb = {}
        for tag, enc in self.encoders.items():
            self.mem_emb[tag] = torch.from_numpy(enc.encode(self.mem_text)).to(
                "cuda" if torch.cuda.is_available() else "cpu")
        cnt = np.bincount(self.mem_codes, minlength=len(self.cs)).astype(np.float32)
        self.code_prior = np.log1p(cnt)
        if self.system == "ICD10CM":
            cat = self.meta["category"].to_numpy()
            s = pd.Series(cnt).groupby(cat).transform("sum").to_numpy()
            self.group_prior = np.log1p(s)
            chap = self.meta["chapter"].to_numpy()
            self.chap_prior = np.log1p(pd.Series(cnt).groupby(chap).transform("sum").to_numpy())
        else:
            tag = self.cs.tag_of
            self.group_prior = np.log1p(pd.Series(cnt).groupby(tag).transform("sum").to_numpy())
            self.chap_prior = self.group_prior
        # P(group | subconcept) — semantic tag (SNOMED) / chapter (ICD) given the subconcept label
        grp = self.cs.tag_of if self.system == "SNOMEDCT_US" else self.meta["chapter"].to_numpy()
        self.grp_of = grp
        if "subconcept" in mem:
            t = pd.crosstab(mem["subconcept"], grp[self.mem_codes])
            self.p_grp_sub = (t + 0.5).div(t.sum(1) + 0.5 * t.shape[1], axis=0)
        else:
            self.p_grp_sub = None
        return self

    # ------------------------------------------------------------ candidates
    def candidates(self, q: pd.DataFrame, exclude_self=False) -> pd.DataFrame:
        """q: rows with entity [, chunk_text, subconcept]. Returns candidate feature table."""
        t0 = time.time()
        self._build_groups()
        n = len(q)
        var_lists = [variants(e) for e in q["entity"]]
        flat, owner = [], []
        for r, vs in enumerate(var_lists):
            flat += vs; owner += [r] * len(vs)
        owner = np.array(owner)
        frames = [lexical_candidates(self.char, var_lists, top=self.top_lex),
                  lexical_candidates(self.bm25, var_lists, top=self.top_lex)]
        qv = {}
        for tag, enc in self.encoders.items():
            qv[tag] = enc.encode(flat)
            frames.append(dense_candidates(qv[tag], owner, self.dense[tag].Dt, self.cs.desc.code_idx.to_numpy(), n,
                                           top=self.top_dense, name=f"dense_{tag}", groups=self.groups))
            first = np.array([vs_i for vs_i in np.unique(owner, return_index=True)[1]])
            frames.append(knn_features(qv[tag][first], self.mem_emb[tag], self.mem_codes, k=self.knn_k,
                                       name=f"knn_{tag}"))
            self.dense[tag].release()
        frames.append(char_knn_features([vs[0] for vs in var_lists], self.mem_text, self.mem_codes, k=self.knn_k))
        cand = merge(frames)
        log(f"{self.system}: {n} rows -> {len(cand)} candidates ({len(cand) / max(n, 1):.0f}/row) "
            f"retrieval {time.time() - t0:.0f}s")

        # ---- fill missing retriever scores with exact values
        for tag in self.encoders:
            cand[f"dense_{tag}_sim"] = dense_fill(cand, qv[tag], owner, self.dense[tag].Dt, self.ct)
            self.dense[tag].release()
        for name in ("char", "bm25"):
            col = f"{name}_sim"
            miss = cand[col].isna().to_numpy()
            if miss.any():
                idx = self.char if name == "char" else self.bm25
                cand.loc[miss, col] = _lex_fill(idx, cand[miss], var_lists, self.ct)
        rank_cols = [c for c in cand.columns if c.endswith("rank")]
        cand[rank_cols] = cand[rank_cols].fillna(999)
        cand = cand.fillna({c: 0.0 for c in cand.columns if c.startswith(("knn_", "kchar_"))})

        # ---- string similarity
        fz = fuzzy_features(cand, var_lists, self.ct)
        cand = pd.concat([cand.reset_index(drop=True), fz, specificity_features(cand, var_lists, self.ct)], axis=1)
        # ---- query flags
        qf = query_flags([vs[-1] + " | " + vs[0] for vs in var_lists])
        base_tok = np.array([len(vs[0].split()) for vs in var_lists])
        qf["q_ntok"] = base_tok
        qf["q_nchar"] = np.array([len(vs[0]) for vs in var_lists])
        cand = cand.join(qf, on="row")
        cand["n_variants"] = np.array([len(v) for v in var_lists])[cand["row"].to_numpy()]
        # ---- code features
        ci = cand["code_idx"].to_numpy()
        cand = pd.concat([cand, self.ct.flags.iloc[ci].reset_index(drop=True)], axis=1)
        cand["c_ndesc"] = self.ct.n_desc[ci]
        cand["c_namelen"] = self.ct.name_len[ci]
        cand["code_prior"] = self.code_prior[ci]
        cand["group_prior"] = self.group_prior[ci]
        cand["chap_prior"] = self.chap_prior[ci]
        if self.system == "ICD10CM":
            m = self.meta
            cand["c_len"] = m["length"].to_numpy()[ci]
            cand["c_leaf"] = m["is_leaf"].to_numpy()[ci]
            cand["c_range"] = m["is_range"].to_numpy()[ci]
            cand["c_chapter"] = pd.Categorical(m["chapter"].to_numpy()[ci], categories=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        else:
            tags = sorted(set(self.cs.tag_of))
            cand["c_tag"] = pd.Categorical(self.cs.tag_of[ci], categories=tags)
        # laterality agreement
        for k in ("left", "right", "bilateral"):
            cand[f"lat_{k}_match"] = (cand[f"q_{k}"].astype(int) - cand[f"c_{k}"].astype(int)).astype(np.int8)
        # P(group | subconcept)
        if self.p_grp_sub is not None and "subconcept" in q:
            sub = q["subconcept"].to_numpy()[cand["row"].to_numpy()]
            grp = self.grp_of[ci]
            P = self.p_grp_sub
            lut = {(s, g): P.at[s, g] for s in P.index for g in P.columns}
            cand["p_grp_sub"] = np.array([lut.get((s, g), 0.0) for s, g in zip(sub, grp)], np.float32)
            cand["subconcept"] = pd.Categorical(sub, categories=SUBCONCEPTS)
        # rank-normalized dense score within row
        for tag in self.encoders:
            c = f"dense_{tag}_sim"
            cand[f"{c}_gap"] = cand[c] - cand.groupby("row")[c].transform("max")
        cand["char_sim_gap"] = cand["char_sim"] - cand.groupby("row")["char_sim"].transform("max")
        log(f"{self.system}: features done {time.time() - t0:.0f}s, {cand.shape[1]} cols")
        return cand


def _lex_fill(index, sub, var_lists, ct):
    rows = sub["row"].to_numpy(); codes = sub["code_idx"].to_numpy()
    out = np.zeros(len(sub), np.float32)
    if not hasattr(index, "D"):
        index.D = index.DT.T.tocsr()
    # group by row to batch one query matrix per row
    order = np.argsort(rows, kind="stable")
    i = 0
    while i < len(order):
        r = rows[order[i]]
        j = i
        while j < len(order) and rows[order[j]] == r:
            j += 1
        idxs = order[i:j]
        drows, owners = [], []
        for k, p in enumerate(idxs):
            dr = ct.rows_sorted[ct.ptr[codes[p]]:ct.ptr[codes[p] + 1]]
            drows.extend(dr); owners.extend([k] * len(dr))
        Q = index.query_matrix(var_lists[r])
        S = np.asarray((Q @ index.D[drows].T).todense()).max(0)
        best = np.zeros(len(idxs), np.float32)
        np.maximum.at(best, np.array(owners), S)
        out[idxs] = best
        i = j
    return out
