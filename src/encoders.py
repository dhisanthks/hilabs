"""Dense text encoders (biomedical / general sentence embedders) behind one interface."""
from __future__ import annotations

import numpy as np
import torch

# name -> (hf id, pooling, query prefix, doc prefix)
ENCODERS = {
    "sapbert": ("cambridgeltl/SapBERT-from-PubMedBERT-fulltext", "cls", "", ""),
    "biolord": ("FremyCompany/BioLORD-2023", "mean", "", ""),
    "gte": ("thenlper/gte-base", "mean", "", ""),
    "bge": ("BAAI/bge-base-en-v1.5", "cls", "", ""),
    "e5": ("intfloat/e5-base-v2", "mean", "query: ", "query: "),
    "medembed": ("abhinand/MedEmbed-base-v0.1", "cls", "", ""),
    "medcpt": ("ncbi/MedCPT-Query-Encoder", "cls", "", ""),
}


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class Encoder:
    def __init__(self, name_or_path: str, pooling: str = "mean", qprefix: str = "", dprefix: str = "",
                 max_len: int = 64):
        from sentence_transformers import SentenceTransformer, models

        if name_or_path in ENCODERS:
            hf, pooling, qprefix, dprefix = ENCODERS[name_or_path]
        else:
            hf = name_or_path
        # SapBERT / MedCPT ship as plain HF checkpoints: build CLS pooling explicitly
        raw_hf = any(k in hf.lower() for k in ("sapbert", "medcpt"))
        if not raw_hf:
            st = SentenceTransformer(hf, device=device())
        else:
            word = models.Transformer(hf, max_seq_length=max_len)
            pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode=pooling)
            st = SentenceTransformer(modules=[word, pool], device=device())
        st.max_seq_length = max_len
        if device() == "cuda":
            st.half()
        self.model = st
        self.qprefix, self.dprefix = qprefix, dprefix

    @classmethod
    def from_model(cls, st, qprefix="", dprefix=""):
        self = cls.__new__(cls)
        if device() == "cuda":
            st.half()
        self.model, self.qprefix, self.dprefix = st, qprefix, dprefix
        return self

    def encode(self, texts, query=True, batch_size=512, show=False) -> np.ndarray:
        p = self.qprefix if query else self.dprefix
        texts = [p + t for t in texts]
        # sort by length for speed, restore order after
        order = np.argsort([len(t) for t in texts])
        emb = self.model.encode([texts[i] for i in order], batch_size=batch_size, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=show).astype(np.float16)
        out = np.empty_like(emb)
        out[order] = emb
        return out


def topk_dense(Q: np.ndarray, D: np.ndarray | torch.Tensor, k: int = 100, bs: int = 256):
    """Exact inner-product top-k on GPU (D kept resident in fp16)."""
    dev = device()
    Dt = D if isinstance(D, torch.Tensor) else torch.from_numpy(D).to(dev)
    I, S = [], []
    for s in range(0, len(Q), bs):
        q = torch.from_numpy(Q[s:s + bs]).to(dev)
        sc = q @ Dt.T
        v, i = torch.topk(sc.float(), k, dim=1)
        I.append(i.cpu().numpy()); S.append(v.cpu().numpy())
    return np.vstack(I), np.vstack(S)


def topk_dense_grouped(Q: np.ndarray, Dt: torch.Tensor, groups: dict, k: int = 300, kg: int = 20, bs: int = 128):
    """Global top-k plus top-kg inside each group of description rows (e.g. SNOMED semantic tags).
    groups: {name: LongTensor of desc-row indices on Dt.device}. Returns (I, S, {name: (I_g, S_g)})."""
    dev = Dt.device
    I, S = [], []
    GI = {g: [] for g in groups}; GS = {g: [] for g in groups}
    for s in range(0, len(Q), bs):
        q = torch.from_numpy(Q[s:s + bs]).to(dev)
        sc = q @ Dt.T
        v, i = torch.topk(sc.float(), k, dim=1)
        I.append(i.cpu().numpy()); S.append(v.cpu().numpy())
        for g, idx in groups.items():
            sub = sc[:, idx]
            v, i = torch.topk(sub.float(), min(kg, len(idx)), dim=1)
            GI[g].append(idx[i].cpu().numpy()); GS[g].append(v.cpu().numpy())
        del sc
    return np.vstack(I), np.vstack(S), {g: (np.vstack(GI[g]), np.vstack(GS[g])) for g in groups}
