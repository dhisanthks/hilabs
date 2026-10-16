"""Clinical abbreviation expansion.

Curated from general clinical-documentation knowledge (common chart abbreviations). Ambiguous
abbreviations map to several senses; the engine queries every sense and lets the reranker (which
sees the chunk context and subconcept) decide. Expansions are *added* as alternative query
variants — the original string is always kept, so expansion can never remove signal.
"""
from __future__ import annotations

import re

ABBREV: dict[str, list[str]] = {
    # cardio
    "htn": ["hypertension"], "hld": ["hyperlipidemia"], "hlp": ["hyperlipidemia"],
    "chf": ["congestive heart failure"], "hf": ["heart failure"],
    "hfref": ["heart failure with reduced ejection fraction"], "hfpef": ["heart failure with preserved ejection fraction"],
    "cad": ["coronary artery disease"], "ashd": ["atherosclerotic heart disease"], "mlad": ["mid left anterior descending artery disease coronary artery"],
    "lad": ["left anterior descending artery"], "mi": ["myocardial infarction"], "nstemi": ["non st elevation myocardial infarction"],
    "stemi": ["st elevation myocardial infarction"], "afib": ["atrial fibrillation"], "a-fib": ["atrial fibrillation"], "af": ["atrial fibrillation"],
    "afl": ["atrial flutter"], "aflutter": ["atrial flutter"], "vtach": ["ventricular tachycardia"], "vt": ["ventricular tachycardia"],
    "nsvt": ["nonsustained ventricular tachycardia"], "svt": ["supraventricular tachycardia"], "tach": ["tachycardia"], "brady": ["bradycardia"],
    "pvc": ["premature ventricular contraction"], "pvcs": ["premature ventricular contractions"], "pac": ["premature atrial contraction"],
    "mvp": ["mitral valve prolapse"], "mr": ["mitral regurgitation"], "tr": ["tricuspid regurgitation"], "ar": ["aortic regurgitation"],
    "as": ["aortic stenosis"], "ms": ["multiple sclerosis", "mitral stenosis", "mental status"], "pad": ["peripheral arterial disease"],
    "pvd": ["peripheral vascular disease"], "dvt": ["deep vein thrombosis"], "pe": ["pulmonary embolism", "physical examination"],
    "vte": ["venous thromboembolism"], "icm": ["ischemic cardiomyopathy"], "nicm": ["nonischemic cardiomyopathy"],
    "cm": ["cardiomyopathy"], "lvh": ["left ventricular hypertrophy"], "ef": ["ejection fraction"], "lbbb": ["left bundle branch block"],
    "rbbb": ["right bundle branch block"], "aaa": ["abdominal aortic aneurysm"], "cabg": ["coronary artery bypass graft"],
    "pci": ["percutaneous coronary intervention"], "aicd": ["automatic implantable cardioverter defibrillator"],
    "icd": ["implantable cardioverter defibrillator"], "ppm": ["permanent pacemaker"], "ascvd": ["atherosclerotic cardiovascular disease"],
    "doe": ["dyspnea on exertion"], "pnd": ["paroxysmal nocturnal dyspnea"], "jvd": ["jugular venous distension"],
    "ttp": ["tenderness to palpation", "thrombotic thrombocytopenic purpura"], "ekg": ["electrocardiogram"], "ecg": ["electrocardiogram"],
    "tte": ["transthoracic echocardiogram"], "tee": ["transesophageal echocardiogram"], "echo": ["echocardiogram"],
    "evar": ["endovascular aortic aneurysm repair"], "svg": ["saphenous vein graft"], "trop": ["troponin"], "trops": ["troponin"],
    "repol": ["repolarization"], "ste": ["st elevation"], "nst": ["nonspecific st"],
    # pulm
    "copd": ["chronic obstructive pulmonary disease"], "sob": ["shortness of breath"], "osa": ["obstructive sleep apnea"],
    "pna": ["pneumonia"], "cap": ["community acquired pneumonia"], "ards": ["acute respiratory distress syndrome"],
    "uri": ["upper respiratory infection"], "urti": ["upper respiratory tract infection"], "ild": ["interstitial lung disease"],
    "ph": ["pulmonary hypertension"], "pft": ["pulmonary function test"], "cpap": ["continuous positive airway pressure"],
    "bipap": ["bilevel positive airway pressure"], "nippv": ["noninvasive positive pressure ventilation"], "o2": ["oxygen"],
    "ahi": ["apnea hypopnea index"], "nebs": ["nebulizer"], "neb": ["nebulizer"], "trach": ["tracheostomy"], "rsv": ["respiratory syncytial virus"],
    "cxr": ["chest x-ray"], "ggo": ["ground glass opacity"],
    # endo / renal
    "dm": ["diabetes mellitus"], "dm1": ["type 1 diabetes mellitus"], "dm2": ["type 2 diabetes mellitus"], "t1dm": ["type 1 diabetes mellitus"],
    "t2dm": ["type 2 diabetes mellitus"], "dmt2": ["type 2 diabetes mellitus"], "niddm": ["type 2 diabetes mellitus"],
    "iddm": ["type 1 diabetes mellitus"], "gdm": ["gestational diabetes mellitus"], "dka": ["diabetic ketoacidosis"],
    "pre dm": ["prediabetes"], "predm": ["prediabetes"], "a1c": ["hemoglobin a1c"], "hba1c": ["hemoglobin a1c"], "hgba1c": ["hemoglobin a1c"],
    "cgm": ["continuous glucose monitor"], "bg": ["blood glucose"], "fsbg": ["fingerstick blood glucose"], "glu": ["glucose"],
    "ckd": ["chronic kidney disease"], "esrd": ["end stage renal disease"], "aki": ["acute kidney injury"], "arf": ["acute renal failure"],
    "cki": ["chronic kidney insufficiency"], "uti": ["urinary tract infection"], "bph": ["benign prostatic hyperplasia"],
    "egfr": ["estimated glomerular filtration rate"], "gfr": ["glomerular filtration rate"], "bun": ["blood urea nitrogen"],
    "cr": ["creatinine"], "ua": ["urinalysis"], "tsh": ["thyroid stimulating hormone"], "hypothyroid": ["hypothyroidism"],
    "hyperthyroid": ["hyperthyroidism"], "bmp": ["basic metabolic panel"], "cmp": ["comprehensive metabolic panel"], "cbc": ["complete blood count"],
    "hco3": ["bicarbonate"], "k": ["potassium"], "na": ["sodium"], "mg": ["magnesium"], "phos": ["phosphate"],
    "anc": ["absolute neutrophil count"], "wbc": ["white blood cell count"], "rbc": ["red blood cell count"], "hgb": ["hemoglobin"],
    "hct": ["hematocrit"], "plt": ["platelet count"], "inr": ["international normalized ratio"], "ptt": ["partial thromboplastin time"],
    "bnp": ["brain natriuretic peptide"], "probnp": ["n-terminal pro brain natriuretic peptide"], "ldl": ["low density lipoprotein"],
    "hdl": ["high density lipoprotein"], "rpr": ["rapid plasma reagin"], "esr": ["erythrocyte sedimentation rate"], "crp": ["c-reactive protein"],
    "abg": ["arterial blood gas"], "vbg": ["venous blood gas"], "poct": ["point of care test"], "poc": ["point of care"],
    "lft": ["liver function test"], "lfts": ["liver function tests"], "ast": ["aspartate aminotransferase"], "alt": ["alanine aminotransferase"],
    "psa": ["prostate specific antigen"], "mrsa": ["methicillin resistant staphylococcus aureus"], "c-diff": ["clostridium difficile"],
    "cdiff": ["clostridium difficile"],
    # GI
    "gerd": ["gastroesophageal reflux disease"], "pud": ["peptic ulcer disease"], "ibs": ["irritable bowel syndrome"],
    "ibd": ["inflammatory bowel disease"], "gib": ["gastrointestinal bleeding"], "gi": ["gastrointestinal"], "brbpr": ["bright red blood per rectum"],
    "brpr": ["bright red blood per rectum"], "n/v": ["nausea and vomiting"], "nv": ["nausea and vomiting"], "n/v/d": ["nausea vomiting diarrhea"],
    "abd": ["abdominal"], "nash": ["nonalcoholic steatohepatitis"], "nafld": ["nonalcoholic fatty liver disease"], "sbo": ["small bowel obstruction"],
    "egd": ["esophagogastroduodenoscopy"], "peg": ["percutaneous endoscopic gastrostomy"], "npo": ["nothing by mouth"],
    # neuro / psych
    "cva": ["cerebrovascular accident"], "tia": ["transient ischemic attack"], "ich": ["intracerebral hemorrhage"], "sah": ["subarachnoid hemorrhage"],
    "sz": ["seizure"], "ams": ["altered mental status"], "loc": ["loss of consciousness"], "ha": ["headache"], "tbi": ["traumatic brain injury"],
    "pd": ["parkinson disease"], "als": ["amyotrophic lateral sclerosis"], "rls": ["restless legs syndrome"], "eds": ["excessive daytime sleepiness"],
    "mdd": ["major depressive disorder"], "gad": ["generalized anxiety disorder"], "ptsd": ["post traumatic stress disorder"],
    "adhd": ["attention deficit hyperactivity disorder"], "ocd": ["obsessive compulsive disorder"], "si": ["suicidal ideation"],
    "hi": ["homicidal ideation"], "etoh": ["alcohol"], "aud": ["alcohol use disorder"], "oud": ["opioid use disorder"], "sud": ["substance use disorder"],
    "bpd": ["bipolar disorder", "borderline personality disorder"], "mci": ["mild cognitive impairment"], "phq": ["patient health questionnaire"],
    "aox3": ["alert and oriented"], "aox4": ["alert and oriented"], "a&o": ["alert and oriented"],
    # MSK / rehab
    "oa": ["osteoarthritis"], "ra": ["rheumatoid arthritis", "right atrium"], "djd": ["degenerative joint disease"], "ddd": ["degenerative disc disease"],
    "lbp": ["low back pain"], "rom": ["range of motion"], "arom": ["active range of motion"], "prom": ["passive range of motion"],
    "aarom": ["active assisted range of motion"], "fx": ["fracture"], "tka": ["total knee arthroplasty"], "tkr": ["total knee replacement"],
    "tha": ["total hip arthroplasty"], "thr": ["total hip replacement"], "orif": ["open reduction internal fixation"], "slr": ["straight leg raise"],
    "adl": ["activities of daily living"], "adls": ["activities of daily living"], "iadl": ["instrumental activities of daily living"],
    "iadls": ["instrumental activities of daily living"], "lob": ["loss of balance"], "wb": ["weight bearing"], "wbat": ["weight bearing as tolerated"],
    "nwb": ["non weight bearing"], "fwb": ["full weight bearing"], "pwb": ["partial weight bearing"], "wbing": ["weight bearing"],
    "sts": ["sit to stand"], "eob": ["edge of bed"], "oob": ["out of bed"], "rw": ["rolling walker"], "fww": ["front wheeled walker"],
    "w/c": ["wheelchair"], "wc": ["wheelchair"], "hep": ["home exercise program"], "pt": ["physical therapy", "patient"],
    "ot": ["occupational therapy"], "slp": ["speech language pathology"], "pt/ot": ["physical therapy occupational therapy"],
    "dme": ["durable medical equipment"], "ue": ["upper extremity"], "ues": ["upper extremities"], "le": ["lower extremity"],
    "les": ["lower extremities"], "ble": ["bilateral lower extremities"], "bue": ["bilateral upper extremities"],
    "lle": ["left lower extremity"], "rle": ["right lower extremity"], "lue": ["left upper extremity"], "rue": ["right upper extremity"],
    "ub": ["upper body"], "lb": ["lower body"], "c-spine": ["cervical spine"], "l-spine": ["lumbar spine"], "scap": ["scapular"],
    "tfesi": ["transforaminal epidural steroid injection"], "esi": ["epidural steroid injection"], "mmt": ["manual muscle testing"],
    "gtb": ["green theraband"], "rtb": ["red theraband"], "ytb": ["yellow theraband"], "btb": ["blue theraband"],
    # general / misc
    "h/o": ["history of"], "w": ["with"], "ho": ["history of"], "hx": ["history"], "pmh": ["past medical history"], "psh": ["past surgical history"], "fhx": ["family history"],
    "fh": ["family history"], "sh": ["social history"], "s/p": ["status post"], "c/o": ["complains of"], "f/u": ["follow up"],
    "fu": ["follow up"], "r/o": ["rule out"], "w/": ["with"], "w/o": ["without"], "wo": ["without"], "b/l": ["bilateral"], "bilat": ["bilateral"],
    "bl": ["bilateral"], "2/2": ["secondary to"], "d/c": ["discharge"], "dc": ["discharge"], "snf": ["skilled nursing facility"],
    "ltc": ["long term care"], "hhc": ["home health care"], "icu": ["intensive care unit"], "micu": ["medical intensive care unit"],
    "ed": ["emergency department", "erectile dysfunction"], "er": ["emergency room"], "pcp": ["primary care physician"],
    "dnr": ["do not resuscitate"], "dni": ["do not intubate"], "goc": ["goals of care"], "acp": ["advance care planning"],
    "bmi": ["body mass index"], "wt": ["weight"], "ht": ["height"], "bp": ["blood pressure"], "hr": ["heart rate"], "rr": ["respiratory rate"],
    "spo2": ["oxygen saturation"], "sat": ["saturation"], "sats": ["oxygen saturation"], "temp": ["temperature"], "lbs": ["pounds"],
    "iv": ["intravenous"], "ivs": ["intravenous"], "im": ["intramuscular"], "po": ["by mouth"], "sq": ["subcutaneous"], "prn": ["as needed"],
    "xr": ["x-ray"], "u/s": ["ultrasound"], "us": ["ultrasound"], "ct": ["computed tomography"], "mri": ["magnetic resonance imaging"],
    "cta": ["computed tomography angiography"], "mra": ["magnetic resonance angiography"], "angio": ["angiography"],
    "hiv": ["human immunodeficiency virus"], "hcv": ["hepatitis c virus"], "hbv": ["hepatitis b virus"], "std": ["sexually transmitted disease"],
    "sti": ["sexually transmitted infection"], "pid": ["pelvic inflammatory disease", "prolapsed intervertebral disc"],
    "ca": ["cancer", "calcium"], "mets": ["metastases"], "hcc": ["hepatocellular carcinoma"], "nsclc": ["non small cell lung cancer"],
    "sclc": ["small cell lung cancer"], "ckd3": ["chronic kidney disease stage 3"], "aaox3": ["alert and oriented"],
    "ugib": ["upper gastrointestinal bleeding"], "lgib": ["lower gastrointestinal bleeding"], "sirs": ["systemic inflammatory response syndrome"],
    "ssi": ["surgical site infection"], "cellulitis": ["cellulitis"], "ptx": ["pneumothorax"], "pna/uti": ["pneumonia urinary tract infection"],
    "abx": ["antibiotics"], "rx": ["prescription"], "tx": ["treatment"], "dx": ["diagnosis"], "sx": ["symptoms"], "ok": ["normal"],
    "wnl": ["within normal limits"], "nad": ["no acute distress"], "nkda": ["no known drug allergies"], "mar": ["medication administration record"],
    "ems": ["emergency medical services"], "crna": ["certified registered nurse anesthetist"], "mac": ["monitored anesthesia care"],
    "asa": ["aspirin", "american society of anesthesiologists"], "nsaid": ["nonsteroidal anti-inflammatory drug"],
    "ppi": ["proton pump inhibitor"], "lmp": ["last menstrual period"], "g1p1": ["gravida para"], "ivf": ["intravenous fluids"],
    "tpn": ["total parenteral nutrition"], "ngt": ["nasogastric tube"], "foley": ["indwelling urinary catheter"],
    "vs": ["vital signs"], "vss": ["vital signs stable"], "rrr": ["regular rate and rhythm"], "ctab": ["clear to auscultation bilaterally"],
    "nt": ["nontender"], "nd": ["nondistended"], "ntnd": ["nontender nondistended"], "hepatosplenomegaly": ["hepatosplenomegaly"],
    "irreg": ["irregular"], "abnrm": ["abnormal"], "abnl": ["abnormal"], "dec": ["decreased"], "incr": ["increased"], "hypox": ["hypoxia"],
}

# merged-token / OCR helpers: common glued words seen in chart text
_GLUE = re.compile(r"(chronic|acute|severe|mild|moderate|left|right|bilateral)(?=[a-z]{4,})")


def expand(text: str) -> list[str]:
    """Return query variants: original first, then every abbreviation-expanded form."""
    variants = [text]
    toks = text.split()
    # multi-token keys first (e.g. "pre dm")
    joined = " " + text + " "
    for k, senses in ABBREV.items():
        if " " in k and f" {k} " in joined:
            for s in senses:
                variants.append(joined.replace(f" {k} ", f" {s} ").strip())
    # single-token keys; ambiguous tokens fan out to every sense (cap the cross product)
    options = []
    for t in toks:
        key = t.strip(".,;:()")
        if key in ABBREV and ABBREV[key] != [key]:
            options.append(ABBREV[key])
        elif "/" in key and all(p in ABBREV for p in key.split("/") if p):
            options.append([" ".join(ABBREV[p][0] for p in key.split("/") if p)])
        else:
            options.append([t])
    if any(len(o) > 1 or o[0] != toks[i] for i, o in enumerate(options)):
        combos = [[]]
        for o in options:
            combos = [c + [s] for c in combos for s in o][:6]
        variants += [" ".join(c) for c in combos]
    unglued = _GLUE.sub(r"\1 ", text)
    if unglued != text:
        variants.append(unglued)
    out, seen = [], set()
    for v in variants:
        if v not in seen:
            seen.add(v); out.append(v)
    return out
