"""Fine-tuned LLM reranker — local inference (adapter trained in the cloud by kaggle/llm_rerank_kaggle.py).

Scores a term's top-k candidates with Qwen3-4B-Instruct-2507 (4-bit) + the fine-tuned LoRA adapter. The prompt and
letter scoring are identical to the training job. Output per candidate: llm_lp (log-probability of its letter).
Features for the ranker: llm_lp, llm_rank, llm_gap (NaN for candidates outside the scored top-k).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch

BASE_MODEL = "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit"
LETTERS = "ABCDEFGHIJKLMNOPQRST"
SYSTEM = ("You are an expert clinical coder. Map the clinical term from the chart to exactly one code of the requested "
          "coding system, following this organisation's coding conventions. Use an unspecified code when the text gives "
          "no further detail; do not invent specificity.")


def cand_text(cs, code_idx):
    if cs.name == "ICD10CM":
        return f"{cs.codes[code_idx]} {cs.name_of[code_idx]}"
    tag = cs.tag_of[code_idx]
    return f"{cs.name_of[code_idx]} ({tag})" if tag else str(cs.name_of[code_idx])


class LLMReranker:
    def __init__(self, adapter_dir, base=BASE_MODEL, max_len=640):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(adapter_dir if os.path.exists(os.path.join(adapter_dir, "tokenizer_config.json")) else base)
        self.tok.padding_side = "left"; self.tok.truncation_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(base, device_map={"": 0}, torch_dtype=torch.float16)
        self.model = PeftModel.from_pretrained(model, adapter_dir).eval()
        self.letter_ids = [self.tok.encode(L, add_special_tokens=False)[-1] for L in LETTERS]
        self.max_len = max_len
        self.adapters = ["default"]

    def add_adapter(self, adapter_dir, name):
        """Second fine-tuned adapter on the same 4-bit base (no extra VRAM for the base weights)."""
        self.model.load_adapter(adapter_dir, adapter_name=name)
        self.adapters.append(name)

    def use(self, name="default"):
        self.model.set_adapter(name)

    def prompt(self, system, subconcept, term, context, texts):
        sysname = "ICD-10-CM" if system == "ICD10CM" else "SNOMED CT"
        lines = [f"Coding system: {sysname}", f"Entity type: {subconcept}", f"Term: {term}"]
        if context:
            lines.append(f"Chart context: ...{context[:300]}...")
        lines.append("\nCandidates:")
        lines += [f"{LETTERS[i]}. {t}" for i, t in enumerate(texts)]
        lines.append("\nAnswer with the single letter of the correct candidate.")
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(lines)}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def score(self, prompts, n_cands, bs=4):
        out = []
        for s in range(0, len(prompts), bs):
            enc = self.tok(prompts[s:s + bs], return_tensors="pt", padding=True, truncation=True,
                           max_length=self.max_len).to("cuda")
            try:
                lg = self.model(**enc, logits_to_keep=1).logits[:, -1, :].float()
            except TypeError:
                lg = self.model(**enc).logits[:, -1, :].float()
            for j in range(lg.shape[0]):
                k = n_cands[s + j]
                out.append(torch.log_softmax(lg[j, self.letter_ids[:k]], -1).cpu().numpy())
        return out


def add_llm_feats(cand: pd.DataFrame, lp: np.ndarray, prefix: str = "llm") -> pd.DataFrame:
    """lp aligned with cand.index (NaN where not scored)."""
    cand = cand.copy()
    cand[f"{prefix}_lp"] = lp
    cand[f"{prefix}_rank"] = cand.groupby("qid")[f"{prefix}_lp"].rank(ascending=False, method="first")
    cand[f"{prefix}_gap"] = cand[f"{prefix}_lp"] - cand.groupby("qid")[f"{prefix}_lp"].transform("max")
    return cand


def load_oof_scores(llm_dir, system, code_to_idx):
    """Out-of-fold scores from the cloud job → DataFrame[qid, code_idx, llm_lp] for one system."""
    parts = [pd.read_csv(os.path.join(llm_dir, f"llm_scores_f{f}.csv"), dtype={"code": str}) for f in range(5)
             if os.path.exists(os.path.join(llm_dir, f"llm_scores_f{f}.csv"))]
    df = pd.concat(parts, ignore_index=True)
    df = df[df.system == system].copy()
    df["code_idx"] = df.code.map(code_to_idx)
    return df.dropna(subset=["code_idx"]).astype({"code_idx": np.int64})[["qid", "code_idx", "llm_lp"]]
