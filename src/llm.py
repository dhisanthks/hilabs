"""Local LLM adjudicator (Qwen3-4B-Instruct-2507, 4-bit) for the final choice among top candidates.

The prompt gives the term, its chunk context, the target coding system, a few *training* examples of how the
coders mapped the most similar terms (retrieval-augmented few-shot = local convention), and up to 10 lettered
candidates. One forward pass; the log-probability of each candidate letter is read from the next-token logits
(no sampling → deterministic). Output is used as a feature by the final reranker, not as a hard decision.
Runs fully locally: no chart text leaves the machine.
"""
from __future__ import annotations

import numpy as np
import torch

MODEL_ID = "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit"
LETTERS = "ABCDEFGHIJ"

SYSTEM = ("You are an expert clinical coder. You map a clinical term extracted from a medical chart to exactly one "
          "code of the requested coding system, following the coding conventions shown in the examples. "
          "Prefer the candidate the examples' conventions would choose; use an unspecified code when the text "
          "gives no further detail; do not invent specificity.")


class LLMJudge:
    def __init__(self, model_id=MODEL_ID, max_len=1024):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", torch_dtype=torch.float16)
        self.model.eval()
        self.max_len = max_len
        self.letter_ids = [self.tok.encode(L, add_special_tokens=False)[0] for L in LETTERS]

    def prompt(self, term, context, system, candidates, examples, subconcept=None):
        sysname = "ICD-10-CM" if system == "ICD10CM" else "SNOMED CT"
        lines = [f"Coding system: {sysname}"]
        if subconcept:
            lines.append(f"Entity type: {subconcept}")
        lines.append(f"Term: {term}")
        if context:
            lines.append(f"Chart context: ...{context[:300]}...")
        if examples:
            lines.append("\nHow our coders mapped similar terms:")
            lines += [f"- \"{t}\" -> {c}" for t, c in examples]
        lines.append("\nCandidates:")
        lines += [f"{LETTERS[i]}. {c}" for i, c in enumerate(candidates)]
        lines.append("\nAnswer with the single letter of the best candidate.")
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(lines)}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return text

    @torch.no_grad()
    def score(self, prompts, n_cands, bs=4):
        """Returns list of arrays: log-softmax over the candidate letters for each prompt."""
        out = []
        for s in range(0, len(prompts), bs):
            batch = prompts[s:s + bs]
            enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len).to("cuda")
            logits = self.model(**enc).logits[:, -1, :].float()
            for j in range(len(batch)):
                k = n_cands[s + j]
                lg = logits[j, self.letter_ids[:k]]
                out.append(torch.log_softmax(lg, -1).cpu().numpy())
        return out
