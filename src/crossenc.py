"""Cross-encoder reranker: scores (query, candidate description) jointly.

Trained listwise: each training group = 1 gold + (G-1) hard negatives taken from the engine's own
candidate pool, softmax cross-entropy over the group. The query string can carry the subconcept
and a short context window; the candidate string carries the code's display name (+ SNOMED tag).
"""
from __future__ import annotations

import math
import os
import random

import numpy as np
import pandas as pd
import torch
from torch import nn

from .encoders import ENCODERS


def set_seed(seed=7):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def cand_text(cs, code_idx):
    name = cs.name_of[code_idx]
    tag = cs.tag_of[code_idx]
    return f"{name} ({tag})" if tag else name


class CrossEncoder:
    def __init__(self, base="sapbert", max_len=64, path=None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        hf = path or ENCODERS.get(base, (base,))[0]
        self.tok = AutoTokenizer.from_pretrained(hf)
        self.model = AutoModelForSequenceClassification.from_pretrained(hf, num_labels=1).cuda()
        self.max_len = max_len

    def _batch(self, q, c):
        return self.tok(q, c, truncation=True, max_length=self.max_len, padding=True, return_tensors="pt").to("cuda")

    def fit(self, groups, epochs=1, group_size=16, groups_per_batch=4, lr=3e-5, seed=7, log_every=200):
        """groups: list of (query, [pos_text, neg1, neg2, ...])."""
        set_seed(seed)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        steps = epochs * math.ceil(len(groups) / groups_per_batch)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, s / max(1, int(0.1 * steps))) * max(0.0, (steps - s) / steps))
        scaler = torch.amp.GradScaler()
        self.model.train()
        step = 0
        rng = random.Random(seed)
        for ep in range(epochs):
            order = list(range(len(groups)))
            rng.shuffle(order)
            for b in range(0, len(order), groups_per_batch):
                gs = [groups[i] for i in order[b:b + groups_per_batch]]
                qs, cs = [], []
                for q, cands in gs:
                    cands = (cands + cands[-1:] * group_size)[:group_size]
                    qs += [q] * group_size; cs += cands
                enc = self._batch(qs, cs)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = self.model(**enc).logits.view(len(gs), group_size).float()
                loss = nn.functional.cross_entropy(logits, torch.zeros(len(gs), dtype=torch.long, device="cuda"))
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step()
                step += 1
                if step % log_every == 0:
                    print(f"  ce step {step}/{steps} loss {loss.item():.3f}", flush=True)
        self.model.eval()
        return self

    @torch.no_grad()
    def score(self, queries, cands, bs=256):
        self.model.eval()
        out = np.zeros(len(queries), np.float32)
        order = np.argsort([len(q) + len(c) for q, c in zip(queries, cands)])
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            enc = self._batch([queries[i] for i in idx], [cands[i] for i in idx])
            with torch.autocast("cuda", dtype=torch.float16):
                out[idx] = self.model(**enc).logits.view(-1).float().cpu().numpy()
        return out

    def save(self, path):
        """Atomic save: write to <path>.tmp, then rename — a failed save never leaves a half-written model."""
        import shutil
        tmp = path + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        self.model.save_pretrained(tmp); self.tok.save_pretrained(tmp)
        shutil.rmtree(path, ignore_errors=True)
        os.replace(tmp, path)


def make_query(entity, subconcept=None, context=None):
    q = entity
    if subconcept:
        q = f"[{subconcept}] {q}"
    if context:
        q = f"{q} | {context}"
    return q


def context_window(entity: str, chunk: str, width=120):
    """Text around the entity in its chunk (cleaned of table markup)."""
    if not isinstance(chunk, str) or not chunk:
        return ""
    ch = chunk.replace("<br>", " ").replace("::", " ")
    import re
    ch = re.sub(r"header_\d+:", " ", ch)
    ch = re.sub(r"\s+", " ", ch).strip()
    i = ch.lower().find(entity.lower()[:30])
    if i < 0:
        return ch[:2 * width]
    return ch[max(0, i - width): i + len(entity) + width]
