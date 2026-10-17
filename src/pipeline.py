"""End-to-end training and inference for the full validated stack.

  candidates (engine)  →  R1 LightGBM  →  cross-encoder on R1 top-K  →  R2 LightGBM (+CE features)
  →  ICD↔SNOMED coupling features from the other system's R2 top-5  →  R3 LightGBM  →  code + confidence

train():  builds every out-of-fold artifact with exactly the code used in the experiments (checkpointed in cache/,
          so re-runs resume), fits R1/R2/R3 on the OOF tables (with- and without-subconcept variants), and trains
          the final bi-encoder and cross-encoder on all training data.
predict(): maps an input file in either accepted schema.
"""
from __future__ import annotations

import gc
import json
import os
import time

import lightgbm as lgb
import numpy as np
from rapidfuzz import fuzz
import pandas as pd
import torch

from . import ranker
from .coupling import coupling_features, pair_table
from .crossenc import CrossEncoder, cand_text, context_window, make_query
from .cv import N_FOLDS, assign_folds
from .dicts import CACHE, ROOT, SYSTEMS, load_system, load_train
from .engine import DenseIndex, Engine, log
from .evaluate import cv_rerank
from .normalize import clean, clean_desc

ART = os.path.join(ROOT, "models")
RUN1 = "ftsap"     # round-1 OOF candidate tables (experiments/exp05_finetune_cv.py)
RUN = "ftsap2"     # round-2 OOF tables: fine-tuned with iterative hard negatives mined from RUN1
CE_RUN = "cectx"   # OOF cross-encoder score name shared with experiments/exp07_crossenc_cv.py
BASE = "sapbert"
CE_CFG = dict(max_len=96, neg=9, gpb=6, epochs=1, topk=30)
SUB_FEATS = ("p_grp_sub", "subconcept")
CE_FEATS = ["ce", "ce_rank", "ce_gap", "r1"]
CP_FEATS = ["cp_cooc", "cp_cooc_w", "cp_has", "cp_sim", "cp_sim_w"]


# ============================================================================ input
def read_input(path: str) -> pd.DataFrame:
    """Accepts the training schema (entity, chunk_text, coding_system[, subconcept]) or the brief's schema
    (chunk_id, clinical_entity[, chunk_text]). Returns entity, chunk_text, [coding_system], [subconcept], ..."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    ren = {}
    for c in df.columns:
        lc = c.strip().lower()
        if lc in ("clinical_entity", "entity", "term"):
            ren[c] = "entity"
        elif lc in ("chunk_text", "chunk", "context", "text"):
            ren[c] = "chunk_text"
        elif lc in ("coding_system", "system"):
            ren[c] = "coding_system"
        elif lc == "subconcept":
            ren[c] = "subconcept"
    df = df.rename(columns=ren)
    if "entity" not in df:
        raise ValueError(f"input needs an entity / clinical_entity column, got {list(df.columns)}")
    if "chunk_text" not in df:
        df["chunk_text"] = ""
    return df


def _ce_queries(entities, chunks):
    return [make_query(clean(e), None, context_window(e, c)) for e, c in zip(entities, chunks)]


def _add_ce_feats(cand, ce_scores):
    """cand sorted by (qid, r1 score); ce_scores aligned with cand.index (NaN outside top-K)."""
    cand = cand.copy()
    cand["ce"] = ce_scores
    cand["ce_rank"] = cand.groupby("qid")["ce"].rank(ascending=False, method="first")
    cand["ce_gap"] = cand["ce"] - cand.groupby("qid")["ce"].transform("max")
    return cand


def _probs(cand, col="score"):
    ex = np.exp(cand[col] - cand.groupby("qid")[col].transform("max"))
    return ex / ex.groupby(cand["qid"]).transform("sum")


def _name_emb(cs):
    path = os.path.join(CACHE, f"emb_names_{BASE}_{cs.name}.npy")
    if not os.path.exists(path):
        from .encoders import Encoder
        enc = Encoder(BASE)
        np.save(path, enc.encode([clean_desc(str(x)) for x in cs.name_of], query=False, batch_size=1024))
        del enc; torch.cuda.empty_cache()
    return torch.from_numpy(np.load(path)).to("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================ training
def _lexical_oof(df):
    from .finetune import mine_hard_negatives
    hard = {}
    for s in SYSTEMS:
        d = df[df.coding_system == s].reset_index(drop=True)
        p = os.path.join(CACHE, f"oof_lex_{s}.parquet")
        if not os.path.exists(p):
            eng = Engine(s, {})
            parts = []
            for f in range(N_FOLDS):
                eng.set_memory(d[d.fold != f])
                te = d[d.fold == f].reset_index()
                c = eng.candidates(te)
                c["qid"] = te["index"].to_numpy()[c["row"].to_numpy()]
                c["fold"] = f
                parts.append(c)
            cand = pd.concat(parts, ignore_index=True)
            gold = d.code.map(eng.cs.code_to_idx).fillna(-1).astype(np.int64).to_numpy()
            cand["label"] = (cand.code_idx.to_numpy() == gold[cand.qid.to_numpy()]).astype(np.int8)
            cand.to_parquet(p)
        hard.update(mine_hard_negatives(pd.read_parquet(p), d, s))
    return hard


def _oof_candidates(df, run, pairs_fn, epochs=2):
    """5-fold candidate tables: per fold, fine-tune the bi-encoder on the other folds (pairs_fn(train_rows)),
    build candidates + features for the held-out fold. Checkpointed per fold as cache/oofpart_<run>_*."""
    from .encoders import Encoder
    from .finetune import finetune
    per = {s: df[df.coding_system == s].reset_index(drop=True) for s in SYSTEMS}
    base = None
    for f in range(N_FOLDS):
        outp = {s: os.path.join(CACHE, f"oofpart_{run}_{s}_f{f}.parquet") for s in SYSTEMS}
        if all(os.path.exists(p) for p in outp.values()):
            continue
        base = base or Encoder(BASE)
        st = finetune(BASE, pairs_fn(df[df.fold != f]), epochs=epochs)
        ft = Encoder.from_model(st)
        for s in SYSTEMS:
            d = per[s]
            eng = Engine(s, {"ft": ft}, cache_dense=False)
            eng.encoders["base"] = base
            eng.dense["base"] = DenseIndex(eng.cs, base, BASE, cache=True)
            eng.set_memory(d[d.fold != f])
            te = d[d.fold == f].reset_index()
            c = eng.candidates(te)
            c["qid"] = te["index"].to_numpy()[c["row"].to_numpy()]
            c["fold"] = f
            gold = d.code.map(eng.cs.code_to_idx).fillna(-1).astype(np.int64).to_numpy()
            c["label"] = (c.code_idx.to_numpy() == gold[c.qid.to_numpy()]).astype(np.int8)
            c.to_parquet(outp[s])
            del eng; gc.collect(); torch.cuda.empty_cache()
        del st, ft; gc.collect(); torch.cuda.empty_cache()
    return {s: pd.concat([pd.read_parquet(os.path.join(CACHE, f"oofpart_{run}_{s}_f{f}.parquet"))
                          for f in range(N_FOLDS)], ignore_index=True) for s in SYSTEMS}


def _iter_negatives(df, round1):
    from .finetune import iterative_hard_negatives
    per = {s: df[df.coding_system == s].reset_index(drop=True) for s in SYSTEMS}
    return iterative_hard_negatives({s: round1[s][["qid", "code_idx", "label", "dense_ft_sim"]] for s in SYSTEMS},
                                    per, k=3)


def _ce_groups(X, folds_mask_fn, neg):
    groups = []
    for s, x in X.items():
        c, d = x["cand"], x["d"]
        tr = c[folds_mask_fn(c) & (c.label == 0) & (c.r1 < neg + 5)]
        negs = tr.groupby("qid").code_idx.apply(list)
        for qi in np.where(folds_mask_fn(d) & (x["gold"] >= 0))[0]:
            ng = negs.get(qi, [])[:neg]
            if ng:
                groups.append((x["q"][qi], [cand_text(x["cs"], x["gold"][qi])] + [cand_text(x["cs"], k) for k in ng]))
    return groups


def train(train_csv=None, epochs=2):
    t0 = time.time()
    os.makedirs(ART, exist_ok=True)
    df = load_train(train_csv)
    df["fold"] = assign_folds(df)
    from .finetune import build_pairs, build_pairs_multi, finetune
    log("[1/6] lexical OOF + lexical hard negatives"); hard = _lexical_oof(df)
    log("[2/6] round-1 fine-tuned candidate tables (lexical negatives)")
    round1 = _oof_candidates(df, RUN1, lambda tr: build_pairs(tr, hard), epochs)
    hard_multi = _iter_negatives(df, round1)
    log("      round-2 fine-tuned candidate tables (iterative hard negatives)")
    oof = _oof_candidates(df, RUN, lambda tr: build_pairs_multi(tr, hard_multi), epochs)
    # ensemble: round-1 encoder features join the round-2 tables as *_ft1_* (complementary: better term↔term kNN)
    for s in SYSTEMS:
        cols = [c for c in round1[s].columns if c.startswith(("dense_ft_", "knn_ft_"))]
        r1f = round1[s][["qid", "code_idx"] + cols].rename(columns={c: c.replace("_ft_", "_ft1_") for c in cols})
        oof[s] = oof[s].merge(r1f, on=["qid", "code_idx"], how="left")
    del round1; gc.collect()
    X, meta = {}, {}
    log("[3/6] R1 reranker (OOF scores → cross-encoder candidates)")
    for s in SYSTEMS:
        d = df[df.coding_system == s].reset_index(drop=True)
        cs = load_system(s)
        cand = oof[s]
        if "sp_extra_n" not in cand:
            from .features import CodeText, specificity_features
            from .query import variants
            ct = CodeText(cs)
            vl = {q: variants(d.entity[q]) for q in cand.qid.unique()}
            sp = specificity_features(cand[["qid", "code_idx"]].rename(columns={"qid": "row"}), vl, ct)
            cand = pd.concat([cand.reset_index(drop=True), sp], axis=1)
        _, _, _, _, cand = cv_rerank(cand, d)
        cand = cand.sort_values(["qid", "score"], ascending=[True, False])
        cand["r1"] = cand.groupby("qid").cumcount().astype(np.int16)
        f1 = ranker.feature_cols(cand.drop(columns=["r1"]))
        for v, fts in (("sub", f1), ("nosub", [f for f in f1 if f not in SUB_FEATS])):
            m, fts = ranker.fit(cand, feats=fts)
            m.save_model(os.path.join(ART, f"r1_{s}_{v}.txt")); meta[f"r1_{s}_{v}"] = fts
        X[s] = dict(d=d, cs=cs, cand=cand.drop(columns=["score"]),
                    q=_ce_queries(d.entity, d.chunk_text),
                    gold=d.code.map(cs.code_to_idx).fillna(-1).astype(int).to_numpy())
    log("[4/6] cross-encoder OOF scores (keyed by term id + code)")
    ce_path = {s: os.path.join(CACHE, f"ce_scores_{CE_RUN}_{s}.parquet") for s in SYSTEMS}
    if not all(os.path.exists(p) for p in ce_path.values()):
        parts = {s: [] for s in SYSTEMS}
        for f in range(N_FOLDS):
            groups = _ce_groups(X, lambda t: (t.fold != f).to_numpy(), CE_CFG["neg"])
            ce = CrossEncoder(BASE, max_len=CE_CFG["max_len"]).fit(groups, epochs=CE_CFG["epochs"],
                                                                   group_size=CE_CFG["neg"] + 1,
                                                                   groups_per_batch=CE_CFG["gpb"])
            for s, x in X.items():
                te = x["cand"][(x["cand"].fold == f) & (x["cand"].r1 < CE_CFG["topk"])]
                sc = ce.score([x["q"][i] for i in te.qid], [cand_text(x["cs"], k) for k in te.code_idx])
                parts[s].append(te[["qid", "code_idx", "fold"]].assign(ce=sc))
            del ce; gc.collect(); torch.cuda.empty_cache()
        for s in SYSTEMS:
            pd.concat(parts[s], ignore_index=True).to_parquet(ce_path[s])
    log("[5/6] R2 (+cross-encoder) and R3 (+ICD↔SNOMED coupling)")
    tops = {}
    for s, x in X.items():
        c = x["cand"]
        ce = pd.read_parquet(ce_path[s])[["qid", "code_idx", "ce"]]
        sc = c[["qid", "code_idx"]].merge(ce, on=["qid", "code_idx"], how="left")["ce"].to_numpy()
        c = _add_ce_feats(c, sc)
        _, _, _, _, c2 = cv_rerank(c, x["d"])
        c["prob"] = _probs(c2)
        c["mention_id"] = x["d"].mention_id.to_numpy()[c.qid.to_numpy()]
        f2 = ranker.feature_cols(c.drop(columns=["prob", "mention_id"]))
        for v, fts in (("sub", f2), ("nosub", [f for f in f2 if f not in SUB_FEATS])):
            m, fts = ranker.fit(c.drop(columns=["prob", "mention_id"]), feats=fts)
            m.save_model(os.path.join(ART, f"r2_{s}_{v}.txt")); meta[f"r2_{s}_{v}"] = fts
        tops[s] = c.sort_values(["qid", "prob"], ascending=[True, False]).groupby("qid").head(5)[
            ["mention_id", "code_idx", "prob", "fold"]]
        x["cand"] = c
    pairs = df.pivot_table(index="mention_id", columns="coding_system", values="code", aggfunc="first").dropna()
    pairs["fold"] = df.groupby("mention_id").fold.first().reindex(pairs.index)
    pairs = pairs.reset_index()
    pairs.drop(columns=["fold"]).to_csv(os.path.join(ART, "pairs.csv"), index=False)
    emb = {s: _name_emb(X[s]["cs"]) for s in SYSTEMS}
    for s, x in X.items():
        o = [t for t in SYSTEMS if t != s][0]
        c = x["cand"]
        feats = [coupling_features(c[c.fold == f], s, tops[o][tops[o].fold == f], x["cs"], X[o]["cs"],
                                   emb[s], emb[o], pair_table(pairs[pairs.fold != f])) for f in range(N_FOLDS)]
        c = c.join(pd.concat(feats)).drop(columns=["prob", "mention_id"])
        f3 = ranker.feature_cols(c)
        for v, fts in (("sub", f3), ("nosub", [f for f in f3 if f not in SUB_FEATS])):
            m, fts = ranker.fit(c, feats=fts)
            m.save_model(os.path.join(ART, f"r3_{s}_{v}.txt")); meta[f"r3_{s}_{v}"] = fts
    json.dump(meta, open(os.path.join(ART, "ranker_features.json"), "w"), indent=1)
    log("[6/6] final bi-encoder and cross-encoder on all training data")
    if not os.path.exists(os.path.join(ART, "biencoder_r1", "config.json")):
        finetune(BASE, build_pairs(df, hard), out_dir=os.path.join(ART, "biencoder_r1"), epochs=epochs)
    if not os.path.exists(os.path.join(ART, "biencoder", "config.json")):
        finetune(BASE, build_pairs_multi(df, hard_multi), out_dir=os.path.join(ART, "biencoder"), epochs=epochs)
    if not os.path.exists(os.path.join(ART, "crossencoder", "config.json")):
        groups = _ce_groups(X, lambda t: np.ones(len(t), bool), CE_CFG["neg"])
        ce = CrossEncoder(BASE, max_len=CE_CFG["max_len"]).fit(groups, epochs=CE_CFG["epochs"],
                                                               group_size=CE_CFG["neg"] + 1, groups_per_batch=CE_CFG["gpb"])
        ce.save(os.path.join(ART, "crossencoder"))
    log(f"training done in {(time.time() - t0) / 60:.1f} min")


# ============================================================================ inference
def _display_names(train, cs):
    """Human-readable display per code for the *output* only: the coders' display string when the code occurs in
    training, else the dictionary description without index-style parenthetical lists."""
    disp = train[train.coding_system == cs.name].groupby("code")["display"].agg(lambda x: x.value_counts().index[0])
    desc = cs.desc[~cs.desc.description.str.contains(r"\) \(", regex=True)]
    first = desc.assign(L=desc.description.str.len()).sort_values("L", ascending=False).groupby("code_idx")["description"].first()
    names = pd.Series(cs.name_of, index=range(len(cs)))
    names.update(first)
    out = names.to_numpy().astype(object)
    for code, dname in disp.items():
        i = cs.code_to_idx.get(code)
        if i is not None:
            out[i] = dname
    return out


class Mapper:
    def __init__(self):
        from .encoders import Encoder
        self.train = load_train()
        self.enc = {"ft": Encoder(os.path.join(ART, "biencoder"), pooling="cls"),
                    "ft1": Encoder(os.path.join(ART, "biencoder_r1"), pooling="cls"),
                    "base": Encoder(BASE)}
        self.ce = CrossEncoder(BASE, max_len=CE_CFG["max_len"], path=os.path.join(ART, "crossencoder"))
        self.feats = json.load(open(os.path.join(ART, "ranker_features.json")))
        self.models = {k: lgb.Booster(model_file=os.path.join(ART, f"{k}.txt")) for k in self.feats}
        pairs = pd.read_csv(os.path.join(ART, "pairs.csv"), dtype=str)
        self.pairs = pair_table(pairs)
        self.engines, self.emb = {}, {}

    def engine(self, s):
        if s not in self.engines:
            eng = Engine(s, {})
            for tag, cache_tag in (("base", BASE), ("ft", "ftfinal"), ("ft1", "ft1final")):
                eng.encoders[tag] = self.enc[tag]
                eng.dense[tag] = DenseIndex(eng.cs, self.enc[tag], cache_tag, cache=True)
            eng.set_memory(self.train[self.train.coding_system == s])
            self.engines[s] = eng
            self.emb[s] = _name_emb(eng.cs)
            self.display = getattr(self, "display", {})
            self.display[s] = _display_names(self.train, eng.cs)
        return self.engines[s]

    def _score(self, cand, stage, s, variant):
        k = f"{stage}_{s}_{variant}"
        return self.models[k].predict(cand[self.feats[k]])

    def stage12(self, q: pd.DataFrame, s: str):
        """Candidates → R1 → cross-encoder → R2. Returns candidate table with prob (R2) per row of q."""
        eng = self.engine(s)
        q = q.reset_index(drop=True)
        variant = "sub" if "subconcept" in q and (q["subconcept"] != "").any() else "nosub"
        cand = eng.candidates(q)
        cand["qid"] = cand["row"]
        cand["score"] = self._score(cand, "r1", s, variant)
        cand = cand.sort_values(["qid", "score"], ascending=[True, False])
        cand["r1"] = cand.groupby("qid").cumcount().astype(np.int16)
        top = cand[cand.r1 < CE_CFG["topk"]]
        queries = _ce_queries(q.entity, q.chunk_text)
        ce = pd.Series(np.nan, index=cand.index)
        ce.loc[top.index] = self.ce.score([queries[i] for i in top.qid], [cand_text(eng.cs, k) for k in top.code_idx])
        cand = _add_ce_feats(cand.drop(columns=["score"]), ce.to_numpy())
        cand["score"] = self._score(cand, "r2", s, variant)
        cand["prob"] = _probs(cand)
        return cand, variant

    def run(self, q: pd.DataFrame, systems_per_row: dict):
        """systems_per_row: {system: index array of q rows needing it}. Mentions = rows of q."""
        st = {}
        for s, idx in systems_per_row.items():
            if len(idx):
                cand, variant = self.stage12(q.iloc[idx], s)
                cand["mention_id"] = np.asarray(idx)[cand["qid"].to_numpy()]
                st[s] = (cand, variant, idx)
        out = {}
        for s, (cand, variant, idx) in st.items():
            o = [t for t in SYSTEMS if t != s][0]
            if o in st:
                oc = st[o][0]
                topb = oc.sort_values(["qid", "prob"], ascending=[True, False]).groupby("qid").head(5)[
                    ["mention_id", "code_idx", "prob"]]
            else:
                topb = pd.DataFrame(columns=["mention_id", "code_idx", "prob"])
            eng = self.engines[s]
            self.engine(o)
            cp = coupling_features(cand, s, topb, eng.cs, self.engines[o].cs, self.emb[s], self.emb[o], self.pairs)
            c3 = cand.drop(columns=["score", "prob"]).join(cp)
            c3["score"] = self._score(c3, "r3", s, variant)
            best = ranker.top1(c3).reindex(range(len(idx)))
            ok = best["code_idx"].notna()
            ci = best["code_idx"].fillna(0).astype(np.int64).to_numpy()
            # runner-up + ambiguity flag (close call or near-synonym twins) for review / set-based scoring
            c3["p"] = _probs(c3)
            second = (c3.sort_values(["qid", "p"], ascending=[True, False]).groupby("qid").nth(1)
                      .set_index("qid").reindex(range(len(idx))))
            ok2 = second["code_idx"].notna()
            ci2 = second["code_idx"].fillna(0).astype(np.int64).to_numpy()
            p1 = best["prob"].fillna(0).to_numpy(); p2 = second["p"].fillna(0).to_numpy()
            n1 = [clean_desc(str(x)) for x in eng.cs.name_of[ci]]; n2 = [clean_desc(str(x)) for x in eng.cs.name_of[ci2]]
            twin = np.array([ok2.iloc[i] and fuzz.token_set_ratio(n1[i], n2[i]) >= 90 for i in range(len(ci))])
            out[s] = pd.DataFrame({
                "code": np.where(ok, eng.cs.codes[ci], ""),
                "display": np.where(ok, self.display[s][ci], ""),
                "confidence": np.round(p1, 4),
                "alt_code": np.where(ok2, eng.cs.codes[ci2], ""),
                "alt_display": np.where(ok2, self.display[s][ci2], ""),
                "alt_confidence": np.round(p2, 4),
                "ambiguous": (p2 >= 0.5 * np.maximum(p1, 1e-9)) | twin,
            }, index=np.asarray(idx))
        return out


def predict(input_csv, output_csv, review_threshold=0.5):
    t0 = time.time()
    q = read_input(input_csv)
    m = Mapper()
    out = q.copy()
    if "coding_system" in q:
        # long format: rows of the same (entity, chunk) mention are coupled; one requested system per row
        mention = q.groupby(["entity", "chunk_text"], sort=False).ngroup().to_numpy()
        uq = q.assign(_m=mention).drop_duplicates("_m").reset_index(drop=True)
        need = {s: np.unique(mention[q.coding_system.to_numpy() == s]) for s in SYSTEMS}
        res = m.run(uq, {s: need[s] for s in SYSTEMS})
        for col in ("code", "display", "alt_code", "alt_display"):
            out[col] = pd.Series([""] * len(out), index=out.index, dtype=object)
        out["confidence"] = np.nan
        out["alt_confidence"] = np.nan
        out["ambiguous"] = False
        for s, r in res.items():
            rows = np.where(q.coding_system.to_numpy() == s)[0]
            for col in r:
                kind = float if col in ("confidence", "alt_confidence") else (bool if col == "ambiguous" else object)
                out.loc[out.index[rows], col] = r[col].reindex(mention[rows]).to_numpy(dtype=kind)
        out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce").fillna(0)
        out["needs_review"] = out["confidence"] < review_threshold
    else:
        allrows = np.arange(len(q))
        res = m.run(q, {s: allrows for s in SYSTEMS})
        for s, r in res.items():
            key = "icd10cm" if s == "ICD10CM" else "snomed"
            out[f"{key}_code"] = r["code"].to_numpy()
            out[f"{key}_display"] = r["display"].to_numpy()
            out[f"{key}_confidence"] = r["confidence"].to_numpy()
            out[f"{key}_alt_code"] = r["alt_code"].to_numpy()
            out[f"{key}_alt_display"] = r["alt_display"].to_numpy()
            out[f"{key}_ambiguous"] = r["ambiguous"].to_numpy()
        out["needs_review"] = (out["icd10cm_confidence"] < review_threshold) | (out["snomed_confidence"] < review_threshold)
    out.to_csv(output_csv, index=False)
    log(f"wrote {len(out)} rows to {output_csv} in {time.time() - t0:.0f}s")
    return out
