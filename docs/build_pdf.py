"""Build the technical write-up PDF: python docs/build_pdf.py  → docs/HiLabs_TermMapping_Writeup.pdf
Numbers live in the RESULTS dict below (all 5-fold cross-validated, see docs/WORKLOG.md)."""
import os

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "HiLabs_TermMapping_Writeup.pdf")

RESULTS = dict(icd=71.1, snomed=58.2, snomed_dx=67.4, icd_dx=71.1,
               icd_unseen=50.1, snomed_unseen=47.0, icd_cat=81.6, icd_chap=89.7, snomed_tag=76.9,
               icd_syn=74.4, snomed_syn=65.6, demo_link="(link to be added)")

INK = colors.HexColor("#1f2933"); ACCENT = colors.HexColor("#1d4ed8"); MUTED = colors.HexColor("#52606d")
LIGHT = colors.HexColor("#eef2ff"); GRID = colors.HexColor("#cbd2d9")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, textColor=INK, spaceBefore=4, spaceAfter=4)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=10.5, textColor=ACCENT, spaceBefore=5, spaceAfter=2)
P = ParagraphStyle("P", parent=ss["BodyText"], fontSize=8.2, leading=10.0, textColor=INK, spaceAfter=1.5)
SMALL = ParagraphStyle("S", parent=P, fontSize=7.6, leading=9.6, textColor=MUTED)
CELL = ParagraphStyle("C", parent=P, fontSize=7.4, leading=8.9, spaceAfter=0)
TITLE = ParagraphStyle("T", parent=ss["Title"], fontSize=17, textColor=INK, spaceAfter=2)
SUB = ParagraphStyle("Sub", parent=P, alignment=TA_CENTER, textColor=MUTED, fontSize=8.5)


def table(rows, widths, header=True, bold_last=False):
    data = [[Paragraph(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, GRID), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), LIGHT)]
    if bold_last:
        st += [("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#fef3c7"))]
    t.setStyle(TableStyle(st))
    return t


def arrow(d, x1, y1, x2, y2):
    d.add(Line(x1, y1, x2, y2, strokeColor=MUTED, strokeWidth=1))
    d.add(Polygon([x2, y2, x2 - 4, y2 + 6, x2 + 4, y2 + 6], fillColor=MUTED, strokeColor=MUTED))


def box(d, x, y, w, h, title, lines=(), fill=LIGHT):
    d.add(Rect(x, y, w, h, rx=4, ry=4, fillColor=fill, strokeColor=ACCENT, strokeWidth=0.8))
    d.add(String(x + w / 2, y + h - 11, title, fontName="Helvetica-Bold", fontSize=8, fillColor=INK, textAnchor="middle"))
    for i, ln in enumerate(lines):
        d.add(String(x + w / 2, y + h - 22 - 9 * i, ln, fontName="Helvetica", fontSize=6.6, fillColor=MUTED,
                     textAnchor="middle"))


def architecture():
    W, H = 17.2 * cm, 9.6 * cm
    d = Drawing(W, H)
    cx = W / 2
    box(d, cx - 95, H - 30, 190, 24, "Input: extracted term + chunk context", ())
    arrow(d, cx, H - 30, cx, H - 40)
    box(d, cx - 190, H - 78, 380, 38, "1  Normalize  →  query variants (original always kept)",
        ("unidecode, markup/(hcc) strip  |  ~350 multi-sense clinical abbreviations (MS → 3 senses)",
         "spelling/OCR repair: glued words, stutter, edit distance to dictionary vocabulary"))
    arrow(d, cx, H - 78, cx, H - 88)
    bw, gap = 88, 5
    names = [("char n-gram", "TF-IDF"), ("BM25", "words"), ("fine-tuned", "SapBERT r1 + r2"),
             ("base SapBERT", "dense"), ("kNN over", "training terms")]
    x0 = cx - (5 * bw + 4 * gap) / 2
    for i, (a, b) in enumerate(names):
        box(d, x0 + i * (bw + gap), H - 128, bw, 36, a, (b,), fill=colors.white)
    d.add(String(cx, H - 136, "2  Candidate generation: union ≈250 (ICD) / ≈475 (SNOMED) codes; SNOMED adds top-k per "
                 "semantic tag; recall 95.8% / 92.8%", fontName="Helvetica-Oblique", fontSize=6.8, fillColor=INK,
                 textAnchor="middle"))
    arrow(d, cx, H - 140, cx, H - 150)
    box(d, cx - 190, H - 184, 380, 34, "3  R1  LightGBM LambdaRank  (~110 features)",
        ("retriever scores/ranks, fuzzy similarity, specificity (extra/missing qualifiers), laterality,",
         "ICD structure, SNOMED tag, code/category priors, P(tag | subconcept)"))
    arrow(d, cx, H - 184, cx, H - 192)
    box(d, cx - 190, H - 220, 380, 28, "4  Cross-encoder (SapBERT, listwise) on R1 top-30  →  R2",
        ("reads term + ±120 chars of chunk context jointly with each candidate name",))
    arrow(d, cx, H - 220, cx, H - 230)
    box(d, cx - 190, H - 258, 380, 28, "5  Fine-tuned LLM on R2 top-10  →  R2+LLM;   ICD ↔ SNOMED coupling  →  R3",
        ("QLoRA Qwen3-4B listwise reranker (cloud-trained, runs locally)  |  coupling kept for ICD (CV)",))
    arrow(d, cx, H - 258, cx, H - 266)
    d.add(String(cx, H - 274, "Output: code + display + calibrated confidence + runner-up + needs_review / ambiguous flags",
                 fontName="Helvetica-Bold", fontSize=7.6, fillColor=ACCENT, textAnchor="middle"))
    return d


def build():
    R = RESULTS
    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=1.4 * cm, rightMargin=1.4 * cm, topMargin=1.1 * cm,
                            bottomMargin=1.1 * cm, title="Clinical Term Mapping Engine — Write-up",
                            author="Team dhisanthks / HiLabs Hackathon 2026")
    s = []
    B = lambda txt: s.append(Paragraph("• " + txt, P))
    s.append(Paragraph("Clinical Term Mapping Engine: ICD-10-CM &amp; SNOMED CT", TITLE))
    s.append(Paragraph(f"HiLabs Hackathon 2026 · technical write-up · demo video: {R['demo_link']} · repo: "
                       "dhisanthks/hilabs (private)", SUB))
    s.append(Spacer(1, 5))
    s.append(Paragraph("All numbers are 5-fold cross-validated on the training data. The evaluation set contains <b>diagnosis / "
                       "condition terms only</b>, so every SNOMED CT number below is on diagnosis terms (all ICD rows are diagnoses). <b>Headline (the lower of the two "
                       "systems, as the brief scores it): 67.4%, set by SNOMED CT</b> — which is why most late work targeted SNOMED. "
                       "SNOMED partial credit uses the semantic tag because the supplied dictionary has no IS-A relationships, so "
                       "hierarchy distance cannot be computed without a UMLS/SNOMED licence.", SMALL))
    s.append(table([
        ["5-fold CV (diagnosis terms)", "ICD-10-CM", "SNOMED CT"],
        ["<b>Exact match</b>", "<b>71.1%</b>", "<b>67.4%</b>"],
        ["Codes never seen in training (long tail)", "50.1%", "57.4%"],
        ["Rare or never-seen codes (≤ 2 training examples)", "61.5%", "63.3%"],
        ["Partial credit: ICD 3-character category / SNOMED semantic tag", "81.6%", "84.4%"],
        ["Exact or near-synonym (name similarity ≥ 90)", "74.4%", "73.9%"],
        ["Most-confident 30% of terms (abstain on the rest)", "96.4%", "95.9%"],
    ], [10.2 * cm, 3.7 * cm, 3.7 * cm]))

    s.append(Paragraph("1. Approach — problem and what the data told us", H2))
    s.append(Paragraph(
        "Input: an entity extracted by CodEx plus its source chunk; output: one ICD-10-CM code and one SNOMED CT concept from the "
        "supplied dictionaries (98,222 ICD codes; 382,124 SNOMED concepts, 984,585 descriptions). 9,646 labelled rows (2,306 ICD, "
        "7,340 SNOMED of which 2,700 diagnosis terms) shaped every decision:", P))
    B("<b>Every entity string occurs once</b> — lookup is useless; the engine must generalize to unseen wording.")
    B("<b>Local conventions dominate</b> — only 6% (ICD) / 15% (SNOMED) of terms match a dictionary string exactly "
      "(<i>can not squat</i> → R26.2 Difficulty in walking; <i>rashes</i> → Eruption).")
    B("<b>Long tail</b> — 24% (ICD) / 46% (SNOMED diagnosis) of held-out terms have a gold code never seen in training folds.")
    B("<b>Label ambiguity</b> — see §6: the gold standard's own consistency is 93.7% / 84.6%.")
    s.append(Paragraph("<b>Protocol.</b> 5-fold CV grouped by source chunk; every learned component (encoders, cross-encoder, LLM, "
                       "kNN memory, priors, rankers) is re-trained inside each fold, so every number is on terms the system never saw.", P))

    s.append(Paragraph("2. Architecture", H2))
    s.append(architecture())

    s.append(Paragraph("3. How accuracy was built up (each step measured, 5-fold CV)", H2))
    s.append(table([
        ["Step", "ICD", "SNOMED (diagnosis)"],
        ["char TF-IDF + nearest-training-term baseline", "41.2", "n/m*"],
        ["+ normalization + R1 LambdaRank over lexical retrievers", "55.9", "53.4"],
        ["+ SapBERT dense retrieval and kNN (best of 7 encoders benchmarked)", "63.1", "n/m*"],
        ["+ per-fold fine-tuned SapBERT + tag-stratified SNOMED retrieval", "66.7", "62.0"],
        ["+ cross-encoder with chunk context (R2)", "67.5", "62.3"],
        ["+ ICD↔SNOMED coupling (R3)", "67.7", "62.2"],
        ["+ round-2 fine-tuning (iterative hard negatives), ensembled with round-1 encoder", "68.3", "62.7"],
        ["+ cross-encoder v2 (negatives from the final ranker, 2 epochs)", "68.4", "63.2"],
        ["+ chunk-context features (rankers see the chart context)", "68.9", "63.6"],
        ["<b>+ fine-tuned LLM reranker (R2+LLM) — final</b>", "<b>71.1</b>", "<b>67.4</b>"],
    ], [12.2 * cm, 2.4 * cm, 3.0 * cm], bold_last=True))
    s.append(Paragraph("* n/m: these two early stages were scored only on all SNOMED rows, not on the diagnosis subset.", SMALL))

    s.append(Paragraph("4. How the models are trained", H2))
    B("<b>Fine-tuned SapBERT bi-encoders (retrieval + kNN features, feed R1).</b> Start from SapBERT (PubMedBERT trained on UMLS "
      "synonyms). Contrastive fine-tuning on (term → gold code description) pairs with MultipleNegativesRanking loss: every other "
      "pair in the batch is a negative, plus explicit hard negatives; 2 epochs, lr 2e-5, batch 64, re-trained per fold. "
      "<i>Round 1</i> uses the lexically closest wrong code as the hard negative. <i>Round 2</i> uses the 3 wrong codes the round-1 "
      "model itself ranked highest (iterative hard negatives) — typically sibling codes differing only in specificity "
      "(E11.40 vs E11.42, unspecified vs specific), so the encoder learns the coders' specificity and wording conventions. "
      "Both rounds are kept as complementary feature sets (ensemble).")
    B("<b>R1 / R2 / R2+LLM / R3 — LightGBM LambdaRank.</b> Each term's candidates form one ranking group (label 1 = gold). "
      "LambdaRank grows trees that push the gold above wrong candidates, weighting each pair by its effect on top-1 (NDCG). "
      "This is how retriever weights are learned: the 'weight' of a retriever is not a fixed number — trees split on each "
      "retriever's score and rank <i>in combination</i> with other features (e.g. trust nearest-training-term votes when several "
      "neighbours agree; distrust dense similarity when the candidate adds an unsupported qualifier), and the criterion is purely "
      "which combination puts the gold first on training folds. A fixed quality-weighted fusion was tested and lost "
      "(38.0% vs 51.5% top-1). Each ranker is 5-seed bagged (70% rows / features per model).")
    B("<b>Cross-encoder (R2).</b> SapBERT as a pair classifier reading [term | ±120 characters of chunk context] with each "
      "candidate name; listwise softmax loss over the gold + 9 hard negatives taken from R1's own top-30; 2 epochs, per fold.")
    B("<b>Fine-tuned LLM reranker (R2+LLM).</b> Qwen3-4B-Instruct-2507 (Apache-2.0) loaded in 4-bit and fine-tuned with QLoRA "
      "(LoRA rank 16, α 32, on all attention and MLP projections; ~0.8% of weights trainable). Prompt: coding system, entity "
      "type, term, chart context and R2's top-10 candidates lettered A–J; the training loss is cross-entropy over the 10 letter "
      "logits at the answer position (listwise, no text generation); candidate order is shuffled during training so it cannot "
      "learn position. 1 epoch, lr 1e-4, honest 5-fold on Kaggle 2× T4 + a final adapter trained on all data. At inference "
      "one forward pass gives a log-probability per candidate (deterministic) → features llm_lp / llm_rank / llm_gap for the "
      "R2+LLM ranker. Gain: +2.2 ICD, +3.8 SNOMED diagnosis.")

    s.append(Paragraph("5. Design decisions (and the stage each one lives in)", H2))
    B("<b>[Retrieval] Many cheap retrievers, union of candidates</b> (char n-gram, BM25, fine-tuned + base SapBERT, kNN over "
      "training terms; SNOMED also top-k inside each semantic tag) → 96% of gold codes in the pool.")
    B("<b>[Normalization] Variants, never replacements</b> — abbreviation senses (MS → 3 senses), spelling/OCR repair, glued words.")
    B("<b>[R1] Retriever weighting learned, not hand-set</b> — LambdaRank over ~110 per-candidate features (see §4).")
    B("<b>[R1] Specificity calibration</b> — handled by the round-2 fine-tuned SapBERT (trained against sibling codes) plus explicit "
      "R1 features: qualifier words a candidate adds (over-coding) or misses (under-coding), unspecified / other / leaf flags. "
      "Specificity errors are balanced (46 over- vs 60 under-coded), i.e. no systematic bias.")
    B("<b>[R1] Context</b> — chunk-context type classifier and chunk-style flags let the ranker use the chart context.")
    B("<b>[R2] Cross-encoder</b> reads term + context jointly with each candidate — resolves abbreviation collisions from context.")
    B("<b>[R2+LLM] LLM where judgement is needed</b> — only on R2's top-10, as features, not as a hard decision.")
    B("<b>[R3] Independent prediction, then coupling</b> — a learned crosswalk recovers only 37% of ICD codes even from the gold "
      "SNOMED code, so each system is predicted on its own; the other system's top-5 enters R3 as evidence. Kept for ICD, "
      "not for SNOMED (final stage R2+LLM).")
    B("<b>[Deployment] Deterministic and private</b> — fixed seeds, no sampling; everything runs locally on a 4 GB GPU: ~0.2 s/term "
      "without the LLM, ~0.85 s/term with it (two-phase inference); dictionary files are fingerprinted for version drift.")

    s.append(Paragraph("6. Error analysis, label ambiguity and flagging", H2))
    s.append(table([
        ["Failure mode (share of errors)", "ICD (667)", "SNOMED diagnosis (879)"],
        ["gold never retrieved", "14%", "13%"],
        ["right ICD category, wrong specificity", "34%", "—"],
        ["right chapter, wrong category / same SNOMED tag, wrong concept", "24%", "48%"],
        ["wrong chapter / wrong SNOMED semantic tag (mostly disorder ↔ finding)", "28%", "39%"],
        ["…of which the predicted name is a near-synonym of the gold (≥ 90)", "10%", "20%"],
    ], [10.6 * cm, 3.0 * cm, 4.0 * cm]))
    B("<b>Label ambiguity.</b> Near-identical training terms share the majority gold code only <b>93.7% (ICD) / 84.6% (SNOMED)</b> of the "
      "time (<i>ble edema</i> → R60.9 vs <i>mild ble edema</i> → R60.0; <i>phq-9</i> vs <i>phq 9</i> → different concepts) — matching the "
      "~0.85 inter-annotator agreement of the SNOMED CT Entity Linking Challenge. 20% of SNOMED diagnosis errors are near-synonyms "
      "(<i>Hyperhidrosis</i> vs <i>Diaphoresis</i>); crediting them gives 73.9%. "
      "Twin-preference features and label cleaning could not learn the coders' tie-breaking (±0.2).")
    B("<b>Hard slices.</b> ICD chapter Z (history/status) 47.5%, T (injury) 51.4%; SNOMED diagnosis terms whose gold is a rare type "
      "(situation 18.5%, procedure 20.6%); context-dependent abbreviations still fail on ICD (<i>MS</i> in an echo report → multiple "
      "sclerosis; SNOMED correctly gives mitral valve stenosis).")
    B("<b>Flagging mechanism.</b> confidence = softmax probability of the chosen code over the final ranker's scores. "
      "<i>needs_review</i> = confidence &lt; 0.5 → flags 30% of SNOMED diagnosis terms, which are only 32.8% correct, while the "
      "unflagged 70% are 82.4% correct. <i>ambiguous</i> = runner-up probability ≥ half the winner's, or runner-up name a "
      "near-synonym (≥ 90); every row also carries the runner-up code (alt_code). Confidence &gt; 0.9 is right 96.1% (SNOMED dx); "
      "accuracy at 70% / 50% / 30% coverage: SNOMED dx 82.4 / 90.1 / 95.9%, ICD 85.4 / 92.3 / 96.4%.")

    s.append(Paragraph("7. Core requirement: the seven hard cases", H2))
    s.append(table([
        ["Failure mode", "How the engine handles it", "Evidence"],
        ["1 Nearest string is wrong", "~110 signals beyond string similarity; specificity features penalize extra qualifiers",
         "HTN → I10; pulmonary hypertension → I27.20"],
        ["2 Abbreviation collision", "all senses retrieved; cross-encoder + LLM read the chunk context; low confidence → review",
         "MS (echo) → mitral valve stenosis"],
        ["3 Specificity calibration", "round-2 fine-tuned SapBERT (sibling negatives) + R1 add/miss-qualifier features",
         "diabetes → E11.9; t2dm with diabetic neuropathy → E11.40"],
        ["4 Granularity mismatch", "both systems predicted independently; coupling only as R3 evidence", "crosswalk alone: 37%"],
        ["5 Long tail", "dictionary-wide retrieval, fine-tuned encoders, LLM knowledge", "never-seen codes 50.1% / 57.4%"],
        ["6 Upstream noise", "markup strip, glued-word split, stutter cut, spelling repair (as variants)",
         "chronickidney disease stag 3 → N18.30"],
        ["7 Version drift", "dictionary fingerprint → all indexes rebuilt; new codes retrievable at once", "README §Version drift"],
    ], [3.6 * cm, 8.8 * cm, 5.2 * cm]))

    s.append(Paragraph("8. Answers to the brief's open questions", H2))
    B("<b>Partial credit?</b> Exact match as headline; report hierarchical credit (ICD category 81.6%, SNOMED tag 84.4%).")
    B("<b>Ambiguous gold?</b> Accept sets of valid codes (20% of SNOMED diagnosis errors are near-synonyms); rows carry alt_code.")
    B("<b>Inter-annotator agreement?</b> Publish it as the ceiling: 93.7% / 84.6% measured; 95% exact match exceeds it.")
    B("<b>Per-category floors?</b> Most ICD chapters 67–82%; a ~50% floor would flag chapter Z (47.5%) and rare SNOMED gold types.")

    s.append(KeepTogether([Paragraph("9. What we tried and abandoned", H2), table([
        ["Idea", "Result", "Decision"],
        ["Zero-shot local LLM judge (Qwen3-4B, few-shot)", "CV blend +0.25", "replaced by fine-tuned LLM"],
        ["2nd LLM: SNOMED-diagnosis specialist over the top-20", "stacked 67.44% = without it", "dropped (redundant)"],
        ["Cross-encoder v3 with the entity type in the query", "ICD +0.6, SNOMED −0.7", "dropped"],
        ["Quality-weighted retriever fusion (RRF, MRR weights)", "38.0% vs 51.5% top-1; as feature ±0.2", "dropped"],
        ["Concept-type classifier / two-level type ranker", "type accuracy = R1's own", "dropped"],
        ["Context-kNN; twin / majority-convention features; cleaned labels", "±0.2 (noise)", "dropped"],
        ["Diagnosis-only specialized ranker", "62.9% vs 63.2% (all-rows ranker)", "dropped"],
        ["Longer encoder training (4 epochs); bigger LightGBM", "±0 / ±1", "harder negatives instead"],
    ], [6.8 * cm, 6.8 * cm, 4.0 * cm])]))

    s.append(Paragraph("10. Challenges &amp; learnings", H2))
    B("<b>The data sets the ceiling:</b> measuring label consistency early would have focused effort sooner.")
    B("<b>New signals beat new features:</b> fine-tuning, cross-encoder and the fine-tuned LLM gave the gains; eight feature ideas gave ≈ 0.")
    B("<b>Only trust full 5-fold numbers:</b> a one-fold check overstated the cross-encoder (+3.5 vs +0.6).")
    B("<b>A 4 GB GPU shaped the engineering:</b> two disk-full incidents (page-file growth), memory-mapped caches, atomic saves, "
      "two-phase inference (5.7 → 0.85 s/term).")

    s.append(Paragraph("11. Innovations", H2))
    s.append(Paragraph("Tag-stratified SNOMED retrieval; iterative hard-negative fine-tuning and a two-round encoder ensemble; "
                       "explicit specificity features; chunk-context features for the rankers; a cross-encoder trained on the final "
                       "ranker's own confusions; a listwise QLoRA-tuned LLM reranker read by letter log-probabilities (deterministic, "
                       "cloud-trained, runs locally); calibrated confidence with review/ambiguity flags and runner-up codes; measured "
                       "label-consistency ceiling and tolerant metrics; dictionary fingerprinting for version drift.", P))

    s.append(Paragraph("12. Future improvements — a plan for the remaining gap, by error bucket", H2))
    s.append(table([
        ["Remaining SNOMED diagnosis errors (879)", "Next step", "Why it should help"],
        ["Wrong semantic tag, 39% (mostly disorder ↔ finding twins)",
         "UMLS/SNOMED licence → IS-A hierarchy; twin-aware features, hierarchical scoring",
         "many are the same clinical idea under two tags; the hierarchy tells the ranker they are siblings"],
        ["Same tag, wrong concept, 48% (close candidates; 20% of all errors are near-synonyms)",
         "stronger LLM judge (7–8B, more epochs) applied only to the 30% flagged terms",
         "the 4B LLM gave the largest late gain (+3.8); the gold is in the top-10 for 88% of terms"],
        ["Gold never retrieved, 13%",
         "synthetic chart-style training terms for unseen codes (local LLM, no patient data); more retrievers",
         "46% of terms have a gold code never seen in training"],
        ["Label ambiguity (ceiling 84.6%)",
         "set-based gold for ambiguous terms; published coding conventions; coder feedback (active learning)",
         "raises the measurable ceiling; flagged terms go to coders, whose decisions become training data"],
    ], [5.4 * cm, 6.4 * cm, 5.8 * cm]))

    s.append(PageBreak())
    s.append(Paragraph("13. External resources", H2))
    s.append(Paragraph(
        "SapBERT (cambridgeltl/SapBERT-from-PubMedBERT-fulltext, Apache-2.0) — bi-encoders and cross-encoder. Qwen3-4B-Instruct-2507 "
        "(Apache-2.0; unsloth 4-bit build) fine-tuned with QLoRA (PEFT, bitsandbytes) on Kaggle 2× T4 GPUs. Benchmarked only: "
        "BioLORD-2023, gte-base, MedCPT, BGE-base-en-v1.5, MedEmbed-base-v0.1, e5-base-v2. ICD-10-CM and SNOMED CT US dictionaries "
        "as supplied; hand-curated clinical abbreviation list; PyTorch, sentence-transformers, transformers, LightGBM, scikit-learn, "
        "RapidFuzz. Related work: DrivenData SNOMED CT Entity Linking Challenge, n2c2 2019 Track 3, xMEN, MedCAT (docs/RESEARCH.md).", P))
    s.append(Paragraph("<b>Reproduce:</b> see README — <font face='Courier'>python run.py predict --input eval.csv --output "
                       "predictions.csv --assume-subconcept DiagnosisCode</font> with the released models.", SMALL))
    s.append(Paragraph("Appendix A — Long-tail generalization (accuracy by gold-code frequency in the training folds)", H2))
    s.append(table([
        ["Gold code seen in training", "ICD share", "ICD accuracy", "SNOMED dx share", "SNOMED dx accuracy"],
        ["absent (never)", "23.8%", "50.1%", "46.2%", "57.4%"],
        ["rare (1–2 times)", "24.4%", "72.6%", "29.8%", "72.4%"],
        ["3–5 times", "21.9%", "80.0%", "16.4%", "80.8%"],
        ["6–20 times", "29.2%", "80.0%", "7.6%", "80.9%"],
        ["rare or absent (≤ 2)", "48%", "61.5%", "76%", "63.3%"],
    ], [5.0 * cm, 2.9 * cm, 2.9 * cm, 3.4 * cm, 3.4 * cm]))
    s.append(Paragraph("Appendix B — Accuracy by category (final model)", H2))
    s.append(table([
        ["ICD chapter", "n", "accuracy", "SNOMED diagnosis gold type", "n", "accuracy"],
        ["N / E / B", "68 / 131 / 30", "82.4 / 80.2 / 80.0%", "disorder", "1,276", "76.1%"],
        ["C / G / K", "50 / 79 / 102", "78.0 / 77.2 / 76.5%", "finding", "1,035", "68.5%"],
        ["J / I / R", "143 / 292 / 690", "74.1 / 74.0 / 72.0%", "situation", "65", "18.5%"],
        ["M / D / H / F", "293 / 28 / 62 / 46", "67.9 / 67.9 / 67.7 / 67.4%", "procedure", "34", "20.6%"],
        ["S / L / T / Z", "50 / 86 / 37 / 80", "66.0 / 62.8 / 51.4 / 47.5%", "", "", ""],
    ], [2.6 * cm, 3.0 * cm, 3.8 * cm, 4.2 * cm, 1.8 * cm, 2.2 * cm]))
    s.append(Paragraph("Appendix C — Zero-shot encoder benchmark (retrieval recall over all rows, kNN@1)", H2))
    s.append(table([
        ["Encoder", "ICD R@1", "ICD R@100", "ICD kNN@1", "SNOMED R@1", "SNOMED R@100", "SNOMED kNN@1"],
        ["SapBERT (chosen)", "32.8", "76.5", "48.5", "38.6", "79.2", "26.7"],
        ["BioLORD-2023", "33.6", "81.1", "46.5", "36.7", "75.4", "25.3"],
        ["gte-base", "28.3", "75.3", "47.1", "38.0", "80.2", "25.8"],
        ["MedCPT", "25.3", "79.3", "47.2", "35.2", "77.6", "25.0"],
    ], [4.2 * cm, 2.4 * cm, 2.4 * cm, 2.0 * cm, 2.2 * cm, 2.4 * cm, 2.0 * cm]))
    doc.build(s)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
