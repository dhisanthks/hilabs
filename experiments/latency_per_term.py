"""Per-term latency (median / p95), one term at a time, models already loaded (steady state).

Phase A (retrieval -> R1 -> cross-encoder -> R2) is timed per term for both systems with the encoders on the GPU;
then the LLM is loaded once and phase B (LLM on the R2 top-10 -> R2+LLM) is timed per term. Latency with the LLM =
A + B per term; without the LLM = A only. Coupling/output formatting (< 5 ms) is not included.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from src.dicts import SYSTEMS, load_train
from src.pipeline import Mapper

N, WARM = 60, 3
df = load_train()
d = df[df.subconcept == "DiagnosisCode"].drop_duplicates(["entity", "chunk_text"]).sample(N + WARM, random_state=0)
q = d[["entity", "chunk_text"]].assign(subconcept="DiagnosisCode").reset_index(drop=True)

m = Mapper()
for s in SYSTEMS:
    m.engine(s)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


ta = np.zeros(len(q)); tb = np.zeros(len(q)); saved = {}
for i in range(len(q)):
    t0 = time.perf_counter()
    for s in SYSTEMS:
        saved[(i, s)] = m.stage_a(q.iloc[[i]], s)
    sync(); ta[i] = time.perf_counter() - t0
m._gpu_encoders(False); m._load_llm()
for i in range(len(q)):
    t0 = time.perf_counter()
    for s in SYSTEMS:
        cand, variant, qq = saved[(i, s)]
        m.stage_b(cand, variant, qq, s)
    sync(); tb[i] = time.perf_counter() - t0
ta, tb = ta[WARM:], tb[WARM:]
res = {"n_terms": N, "both_systems_per_term": True,
       "without_llm_s": {"median": round(float(np.median(ta)), 3), "p95": round(float(np.percentile(ta, 95)), 3)},
       "with_llm_s": {"median": round(float(np.median(ta + tb)), 3), "p95": round(float(np.percentile(ta + tb, 95)), 3)}}
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join("results", "latency_per_term.json"), "w"), indent=1)
