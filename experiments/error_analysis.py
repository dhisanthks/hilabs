"""Error analysis of out-of-fold predictions, broken down by failure mode.

Usage: python error_analysis.py <pred_run>    (reads cache/pred_<pred_run>_<system>.parquet)
Writes results/error_analysis_<run>.md and .json.
"""
import sys, os, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src.dicts import load_train, load_system, SYSTEMS
from src.cv import assign_folds
from src.normalize import clean
from src.abbrev import ABBREV
from src.query import get_speller

run = sys.argv[1]
df = load_train(); df['fold'] = assign_folds(df)
sp = get_speller()
report, md = {}, [f"# Error analysis — `{run}` (5-fold out-of-fold predictions)\n"]


def pct(x):
    return f"{100 * x:.1f}%"


def table(series_dict, title):
    md.append(f"\n**{title}**\n\n| bucket | n | accuracy |\n|---|---|---|")
    for k, (n, a) in series_dict.items():
        md.append(f"| {k} | {n} | {pct(a)} |")


for s in SYSTEMS:
    cs = load_system(s)
    d = df[df.coding_system == s].reset_index(drop=True)
    p = pd.read_parquet(f'cache/pred_{run}_{s}.parquet').reindex(range(len(d)))
    ci = p.code_idx.fillna(0).astype(int).to_numpy()
    d['pred'] = np.where(p.code_idx.notna(), cs.codes[ci], '')
    d['pred_name'] = np.where(p.code_idx.notna(), cs.name_of[ci], '')
    d['ok'] = (d.pred == d.code)
    d['conf'] = p.prob.fillna(0).to_numpy()
    d['in_pool'] = p.in_pool.fillna(0).astype(bool).to_numpy()
    d['seen'] = p.seen.fillna(False).astype(bool).to_numpy()
    q = d.entity.map(clean)
    toks = q.str.split()
    d['is_abbrev'] = toks.map(lambda t: any(x in ABBREV and len(x) <= 5 for x in t))
    d['is_noisy'] = d.entity.str.contains(r'<br>|[^\x00-\x7f]|\?|\(hcc\)', case=False) | (q.map(sp.fix) != q)
    d['is_history'] = q.str.contains(r'\b(?:history|h/o|hx|s/p|status post|prior)\b')
    d['is_negated'] = q.str.contains(r'\b(?:no|denies|negative|without|not)\b')
    d['is_lateral'] = q.str.contains(r'\b(?:left|right|bilateral|lt|rt|b/l|ble|lle|rle)\b')
    d['is_long'] = toks.str.len() >= 6
    acc = d.ok.mean()
    md.append(f"\n## {s}\n\nExact match **{pct(acc)}** on {len(d)} terms. Candidate recall {pct(d.in_pool.mean())}.\n")
    r = dict(acc=acc, n=len(d))
    err = d[~d.ok]
    modes = {'retrieval miss (gold not in candidates)': (~err.in_pool).sum()}
    if s == 'ICD10CM':
        same_cat = err.in_pool & (err.pred.str[:3] == err.code.str[:3])
        same_chap = err.in_pool & ~same_cat & (err.pred.str[0] == err.code.str[0])
        modes['specificity (right 3-char category, wrong code)'] = same_cat.sum()
        modes['right chapter, wrong category'] = same_chap.sum()
        modes['wrong chapter'] = (err.in_pool & ~same_cat & ~same_chap).sum()
        r['category_acc'] = float((d.pred.str[:3] == d.code.str[:3]).mean())
        r['chapter_acc'] = float((d.pred.str[0] == d.code.str[0]).mean())
        md.append(f"Partial credit: 3-char category accuracy **{pct(r['category_acc'])}**, chapter accuracy {pct(r['chapter_acc'])}.\n")
    else:
        tag = pd.Series(cs.tag_of, index=cs.codes)
        wrong_tag = err.in_pool & (err.pred.map(tag) != err.code.map(tag))
        modes['wrong semantic tag (e.g. substance vs procedure)'] = wrong_tag.sum()
        modes['same semantic tag, wrong concept'] = (err.in_pool & ~wrong_tag).sum()
        r['tag_acc'] = float((d.pred.map(tag) == d.code.map(tag)).mean())
        md.append(f"Partial credit: semantic-tag accuracy **{pct(r['tag_acc'])}**.\n")
    # tolerant metrics: how much of the gap is gold-standard ambiguity
    from rapidfuzz import fuzz
    from src.normalize import clean_desc
    gi = d.code.map(cs.code_to_idx).fillna(0).astype(int).to_numpy()
    gname = [clean_desc(str(x)) for x in cs.name_of[gi]]
    pname = [clean_desc(str(x)) for x in d.pred_name]
    sim = np.array([fuzz.token_set_ratio(a, b) for a, b in zip(pname, gname)])
    same_name = np.array([a == b for a, b in zip(pname, gname)])
    tol = {'exact match': float(acc),
           'exact or same-name twin (identical preferred name)': float((d.ok | same_name).mean()),
           'exact or near-synonym (name similarity >= 90)': float((d.ok | (sim >= 90)).mean())}
    if s == 'ICD10CM':
        tol['correct 3-character category'] = float((d.pred.str[:3] == d.code.str[:3]).mean())
    r['tolerant'] = tol
    md.append("\n**Tolerant metrics (how much of the gap is gold-standard ambiguity)**\n\n| metric | accuracy |\n|---|---|")
    for k, v in tol.items():
        md.append(f"| {k} | {pct(v)} |")
    md.append("\n**Failure modes (share of errors)**\n\n| mode | errors | share |\n|---|---|---|")
    for k, v in modes.items():
        md.append(f"| {k} | {v} | {pct(v / max(len(err), 1))} |")
    r['modes'] = {k: int(v) for k, v in modes.items()}
    slices = {}
    for col, name in [('seen', 'gold code seen in training'), ('is_abbrev', 'contains abbreviation'),
                      ('is_noisy', 'OCR / typo / markup noise'), ('is_history', 'history / status-post'),
                      ('is_negated', 'negation cue'), ('is_lateral', 'laterality'), ('is_long', '≥6 tokens')]:
        for val in (True, False):
            m = d[col] == val
            if m.sum():
                slices[f"{name} = {val}"] = (int(m.sum()), float(d.ok[m].mean()))
    table(slices, 'Accuracy by input slice')
    r['slices'] = slices
    if 'subconcept' in d:
        sub = {k: (int(len(g)), float(g.ok.mean())) for k, g in d.groupby('subconcept') if len(g) >= 5}
        table(sub, 'Accuracy by subconcept'); r['subconcept'] = sub
    # calibration and abstention
    d['bin'] = pd.cut(d.conf, [0, .2, .4, .6, .8, .9, 1.0001], include_lowest=True)
    cal = {str(k): (int(len(g)), float(g.ok.mean())) for k, g in d.groupby('bin', observed=True)}
    table(cal, 'Calibration: accuracy by confidence bin')
    r['calibration'] = cal
    order = d.sort_values('conf', ascending=False).ok.to_numpy()
    cum = np.cumsum(order) / np.arange(1, len(order) + 1)
    cov = {}
    md.append("\n**Coverage vs accuracy (abstain on lowest-confidence terms)**\n\n| coverage | accuracy |\n|---|---|")
    for c in (1.0, 0.9, 0.8, 0.7, 0.5, 0.3):
        k = max(1, int(round(c * len(order))))
        cov[c] = float(cum[k - 1]); md.append(f"| {pct(c)} | {pct(cum[k - 1])} |")
    r['coverage_accuracy'] = cov
    md.append("\n**Sample errors**\n\n| entity | gold | predicted | conf |\n|---|---|---|---|")
    for _, e in err.sample(min(15, len(err)), random_state=0).iterrows():
        md.append(f"| {e.entity[:60]} | {e.code} {e.display[:45]} | {e.pred} {str(e.pred_name)[:45]} | {e.conf:.2f} |")
    report[s] = r
open(f'results/error_analysis_{run}.md', 'w', encoding='utf-8').write("\n".join(md))
json.dump(report, open(f'results/error_analysis_{run}.json', 'w'), indent=1, default=str)
print("\n".join(md[:80]))
