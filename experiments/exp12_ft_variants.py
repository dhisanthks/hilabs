"""Exp12: bi-encoder fine-tuning variants, fast proxy evaluation on fold 0.

Train on folds 1-4; on fold 0 measure (a) top-1 accuracy of the encoder alone when re-ranking each term's R1 top-50
candidates (dense max-over-descriptions), (b) kNN@1 over training terms. Variants:
  base2   – current recipe: 2 epochs, 1 lexical hard negative
  ep4     – 4 epochs
  iter2   – iterative hard negatives: top-3 wrong candidates by the round-1 fine-tuned model's OOF dense score
  iter3   – iter2 with 3 epochs
Usage: python exp12_ft_variants.py base2 ep4 iter2 iter3
"""
import sys, os, gc, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch
from src.dicts import load_train, load_system, SYSTEMS, CACHE
from src.cv import assign_folds
from src.engine import log
from src.encoders import Encoder
from src.finetune import build_pairs, finetune, mine_hard_negatives
from src.normalize import clean, clean_desc
from src.query import variants

df = load_train(); df['fold'] = assign_folds(df)
per = {s: df[df.coding_system == s].reset_index(drop=True) for s in SYSTEMS}
cands = {s: pd.read_parquet(os.path.join(CACHE, f'oof_ftsap+cectx_{s}.parquet'),
                            columns=['qid', 'code_idx', 'label', 'fold', 'r1', 'dense_ft_sim']) for s in SYSTEMS}
cs_ = {s: load_system(s) for s in SYSTEMS}

lex_hard = {}
for s in SYSTEMS:
    lex_hard.update(mine_hard_negatives(pd.read_parquet(os.path.join(CACHE, f'oof_lex_{s}.parquet')), per[s], s))


def iter_hard(k=3):
    """{(system, entity): [codes]} top-k wrong candidates by OOF fine-tuned dense similarity."""
    out = {}
    for s in SYSTEMS:
        c = cands[s]
        c = c[c.label == 0].sort_values(['qid', 'dense_ft_sim'], ascending=[True, False]).groupby('qid').head(k)
        ents = per[s].entity.to_numpy()
        for q, g in c.groupby('qid'):
            out[(s, ents[q])] = [cs_[s].codes[i] for i in g.code_idx]
    return out


def pairs_multi(train, hard_multi, seed=7):
    """One triplet per (positive description, hard negative) — up to 3 negatives per term."""
    rng = random.Random(seed)
    out = []
    for s, d in train.groupby('coding_system'):
        cs = cs_[s]
        descs = cs.desc.assign(t=cs.desc.description.map(clean_desc)).groupby('code')['t'].apply(list)
        for e, code in zip(d.entity, d.code):
            if code not in descs.index:
                continue
            pos = list(dict.fromkeys(descs[code]))[:3]
            vs = variants(e)
            for j, nc in enumerate(hard_multi.get((s, e), [])):
                if nc not in descs.index:
                    continue
                a = vs[0] if rng.random() < 0.6 or len(vs) == 1 else rng.choice(vs[1:])
                out.append((a, pos[j % len(pos)], descs[nc][0]))
    rng.shuffle(out)
    return out


def proxy_eval(enc):
    res = {}
    for s in SYSTEMS:
        cs, d, c = cs_[s], per[s], cands[s]
        c0 = c[(c.fold == 0) & (c.r1 < 50)]
        codes = np.unique(c0.code_idx)
        dsel = cs.desc[cs.desc.code_idx.isin(codes)]
        D = enc.encode(dsel.description.map(clean_desc).tolist(), query=False, batch_size=512)
        code_rows = pd.Series(np.arange(len(dsel))).groupby(dsel.code_idx.to_numpy()).apply(np.array).to_dict()
        q0 = np.sort(c0.qid.unique())
        Q = enc.encode([clean(x) for x in d.entity[q0]])
        qpos = {q: i for i, q in enumerate(q0)}
        sims = np.array([float((D[code_rows[ci]].astype(np.float32) @ Q[qpos[q]].astype(np.float32)).max())
                         for q, ci in zip(c0.qid, c0.code_idx)])
        top = c0.assign(sim=sims).sort_values(['qid', 'sim'], ascending=[True, False]).groupby('qid').head(1)
        acc_pool = float(top.label.sum() / len(q0))
        mem = np.where(d.fold != 0)[0]
        M = enc.encode([clean(x) for x in d.entity[mem]])
        nn = (Q.astype(np.float32) @ M.astype(np.float32).T).argmax(1)
        knn = float((d.code.to_numpy()[mem][nn] == d.code.to_numpy()[q0]).mean())
        res[s] = dict(dense_top1_in_top50=round(acc_pool, 4), knn1=round(knn, 4))
    return res


train = df[df.fold != 0]
out = {}
for v in sys.argv[1:]:
    if v == 'base2':
        pairs, ep = build_pairs(train, lex_hard), 2
    elif v == 'ep4':
        pairs, ep = build_pairs(train, lex_hard), 4
    elif v.startswith('iter'):
        pairs, ep = pairs_multi(train, iter_hard(3)), int(v[-1])
    log(v, len(pairs), 'pairs', ep, 'epochs')
    st = finetune('sapbert', pairs, epochs=ep)
    r = proxy_eval(Encoder.from_model(st))
    log(v, r)
    out[v] = r
    del st; gc.collect(); torch.cuda.empty_cache()
    json.dump(out, open('results/exp12_ft_variants.json', 'w'), indent=1)
print(out)
