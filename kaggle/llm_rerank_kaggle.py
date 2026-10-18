"""Fine-tuned LLM reranker — cloud GPU job (Kaggle 2x T4 / Colab).

QLoRA fine-tunes Qwen3-4B-Instruct-2507 (4-bit, Apache-2.0) as a *listwise* reranker: given the term, chart context,
entity type and the stack's top-10 candidates (letters A–J), one forward pass; loss = cross-entropy over the 10
letter logits at the answer position (no generation, deterministic scoring).

Honest 5-fold: for fold f, train on the other folds' terms and score fold f → out-of-fold scores for every training
term (used locally as ranker features). Then 'final' trains on all terms and saves the LoRA adapter for test-time
scoring. Each finished unit is written immediately, so a cut session keeps completed folds.

  python llm_rerank_kaggle.py --data /kaggle/input/<dataset>/llm_prompts.jsonl --plan auto      # both GPUs
  python llm_rerank_kaggle.py --data ... --units 0 --limit 200                                  # smoke test / timing
Outputs (in --out, default /kaggle/working/llm_out): llm_scores_f{0..4}.csv, adapter_final/ (+ zip), logs.
"""
import argparse, json, math, os, random, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="/kaggle/working/llm_out")
ap.add_argument("--model", default="unsloth/Qwen3-4B-Instruct-2507-bnb-4bit")
ap.add_argument("--plan", default=None, help="auto = split folds 0-4 + final across 2 GPUs")
ap.add_argument("--units", default=None, help="comma list of units to run in this process: 0..4 and/or final")
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--bs", type=int, default=2)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--max-len", type=int, default=640)
ap.add_argument("--limit", type=int, default=0, help="use only N training examples (smoke test)")
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

if a.plan == "auto":   # launcher: two worker processes, one per GPU, balanced by work
    plan = {0: "0,1,2", 1: "3,4,final"}
    procs = []
    for gpu, units in plan.items():
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        cmd = [sys.executable, __file__, "--data", a.data, "--out", a.out, "--model", a.model, "--units", units,
               "--epochs", str(a.epochs), "--lr", str(a.lr), "--bs", str(a.bs), "--accum", str(a.accum),
               "--max-len", str(a.max_len), "--seed", str(a.seed)]
        log = open(os.path.join(a.out, f"worker_gpu{gpu}.log"), "a")
        procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT))
        print("started GPU", gpu, "units", units, flush=True)
    for p in procs:
        p.wait()
    print("all workers finished; return codes", [p.returncode for p in procs], flush=True)
    sys.exit(0)

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

LETTERS = "ABCDEFGHIJKLMNOPQRST"   # up to 20 candidates (A–T); prompts with fewer use a prefix
SYSTEM = ("You are an expert clinical coder. Map the clinical term from the chart to exactly one code of the requested "
          "coding system, following this organisation's coding conventions. Use an unspecified code when the text gives "
          "no further detail; do not invent specificity.")


def log(*x):
    print(time.strftime("%H:%M:%S"), *x, flush=True)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


recs = [json.loads(l) for l in open(a.data, encoding="utf-8")]
log("records", len(recs))
tok = AutoTokenizer.from_pretrained(a.model)
tok.padding_side = "left"
tok.truncation_side = "left"   # never cut the candidates / answer slot at the end of long prompts
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
LETTER_IDS = [tok.encode(L, add_special_tokens=False)[-1] for L in LETTERS]
assert len(set(LETTER_IDS)) == len(LETTERS), "letter tokens must be distinct for this tokenizer"


def prompt(r, order):
    sysname = "ICD-10-CM" if r["system"] == "ICD10CM" else "SNOMED CT"
    lines = [f"Coding system: {sysname}", f"Entity type: {r['subconcept']}", f"Term: {r['term']}"]
    if r.get("context"):
        lines.append(f"Chart context: ...{r['context'][:300]}...")
    lines.append("\nCandidates:")
    lines += [f"{LETTERS[i]}. {r['cand_text'][j]}" for i, j in enumerate(order)]
    lines.append("\nAnswer with the single letter of the correct candidate.")
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(lines)}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:   # chat templates without a system role: fold it into the user turn
        msgs = [{"role": "user", "content": SYSTEM + "\n\n" + "\n".join(lines)}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def load_model(train=True):
    model = AutoModelForCausalLM.from_pretrained(a.model, device_map={"": 0}, torch_dtype=torch.float16)
    if train:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        # "all-linear" works for any architecture (Qwen q/k/v/o/gate/up/down, Phi qkv_proj/gate_up_proj, Llama ...)
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                         target_modules="all-linear")
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
    return model


def last_logits(model, enc):
    try:
        out = model(**enc, logits_to_keep=1)
    except TypeError:
        out = model(**enc)
    return out.logits[:, -1, :].float()


def train_unit(train_recs, tag):
    seed_all(a.seed)
    rng = random.Random(a.seed)
    ex = [r for r in train_recs if r["gold"] >= 0]
    if a.limit:
        ex = ex[: a.limit]
    model = load_model(train=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = math.ceil(len(ex) * a.epochs / (a.bs * a.accum))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, int(0.05 * steps))) *
                                              max(0.0, (steps - s) / steps))
    scaler = torch.amp.GradScaler()
    model.train()
    n_ex = int(len(ex) * a.epochs)
    order_ex = [ex[i % len(ex)] for i in range(n_ex)]
    rng.shuffle(order_ex)
    t0 = time.time(); step = 0; micro = 0; run_loss = 0.0
    for b in range(0, n_ex, a.bs):
        batch = order_ex[b:b + a.bs]
        texts, targets, ks = [], [], []
        for r in batch:
            k = len(r["cand_text"]); order = list(range(k)); rng.shuffle(order)   # shuffle to avoid position bias
            texts.append(prompt(r, order)); targets.append(order.index(r["gold"])); ks.append(k)
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=a.max_len).to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            lg = last_logits(model, enc)
        letter = lg[:, LETTER_IDS]
        mask = torch.full_like(letter, float("-inf"))
        for i, k in enumerate(ks):
            mask[i, :k] = 0
        loss = nn.functional.cross_entropy(letter + mask, torch.tensor(targets, device="cuda")) / a.accum
        scaler.scale(loss).backward()
        run_loss += loss.item() * a.accum; micro += 1
        if micro % a.accum == 0:
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); step += 1
            if step % 20 == 0 or step == steps:
                el = time.time() - t0
                log(f"[{tag}] step {step}/{steps} loss {run_loss / micro:.3f} | {el / 60:.1f} min elapsed, "
                    f"~{el / step * (steps - step) / 60:.1f} min left")
                run_loss = 0.0; micro = 0
    model.eval()
    return model


@torch.no_grad()
def score_unit(model, recs_te, path):
    rows = []; gold_ok = 0
    for b in range(0, len(recs_te), 4):
        batch = recs_te[b:b + 4]
        texts = [prompt(r, list(range(len(r["cand_text"])))) for r in batch]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=a.max_len).to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            lg = last_logits(model, enc)
        for i, r in enumerate(batch):
            k = len(r["cand_text"])
            lp = torch.log_softmax(lg[i, LETTER_IDS[:k]], -1).cpu().numpy()
            gold_ok += int(r["gold"] >= 0 and int(np.argmax(lp)) == r["gold"])
            for j in range(k):
                rows.append(f"{r['qid']},{r['system']},{r['cand_codes'][j]},{j},{lp[j]:.5f}")
    with open(path + ".tmp", "w") as fh:
        fh.write("qid,system,code,stack_rank,llm_lp\n" + "\n".join(rows) + "\n")
    os.replace(path + ".tmp", path)
    log(f"scored {len(recs_te)} terms → {path}; LLM top-1 accuracy (gold in top-10 counted) {gold_ok / len(recs_te):.4f}")


for unit in a.units.split(","):
    done = os.path.join(a.out, f"llm_scores_f{unit}.csv") if unit != "final" else os.path.join(a.out, "adapter_final", "adapter_config.json")
    if os.path.exists(done) and not a.limit:
        log("skip finished unit", unit); continue
    t = time.time()
    if unit == "final":
        model = train_unit(recs, "final")
        model.save_pretrained(os.path.join(a.out, "adapter_final"))
        tok.save_pretrained(os.path.join(a.out, "adapter_final"))
        subprocess.run(["zip", "-qr", os.path.join(a.out, "adapter_final.zip"), "adapter_final"], cwd=a.out)
        log("saved final adapter")
    else:
        f = int(unit)
        tr = [r for r in recs if r["fold"] != f]; te = [r for r in recs if r["fold"] == f]
        model = train_unit(tr, f"fold{f}")
        if not a.limit:   # keep per-fold adapters (needed to re-score OOF features with a different k later)
            model.save_pretrained(os.path.join(a.out, f"adapter_f{f}"))
        out = os.path.join(a.out, f"llm_scores_f{f}.csv" if not a.limit else f"smoke_scores_f{f}.csv")
        score_unit(model, te if not a.limit else te[:200], out)
    log(f"unit {unit} done in {(time.time() - t) / 60:.1f} min")
    del model; torch.cuda.empty_cache()
