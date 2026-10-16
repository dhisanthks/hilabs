import pandas as pd, re
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',120)
n = pd.read_excel('SNOMEDCT_US_codes_and_descriptions.xlsx', sheet_name='Notes', header=None, dtype=str)
for r in n.itertuples(): print(r[1], '|', r[2])
icd = pd.read_excel('icd10cm_codes.xlsx', dtype=str); icd.columns=['code','description']
sn = pd.read_excel('SNOMEDCT_US_codes_and_descriptions.xlsx', sheet_name='SNOMED codes', dtype=str)
sn.columns=['code','description','order']
icd.to_parquet('cache/icd.parquet'); sn.to_parquet('cache/snomed.parquet')
df = pd.read_csv('hackathon_train.csv', dtype=str)
norm = lambda s: re.sub(r'\s+',' ', str(s).lower().strip())
for sysn, dic in [('ICD10CM',icd),('SNOMEDCT_US',sn)]:
    d = df[df.coding_system==sysn]
    codes=set(dic.code)
    print(sysn, 'gold in dict:', d.code.isin(codes).mean(), d[~d.code.isin(codes)].code.unique()[:20])
    dd = dic.assign(n=dic.description.map(norm))
    m = dd.groupby('n').code.agg(set)
    hit = d.entity.map(norm).map(m)
    ex = hit.notna()
    corr = [c in h if isinstance(h,set) else False for c,h in zip(d.code,hit)]
    print(sysn,'entity exact-matches a description:',ex.mean(),' and gold among those:', sum(corr)/max(ex.sum(),1))
    # display equals which description?
    print(sysn, 'display in dict descriptions of gold code:', (d.display.map(norm)+'|'+d.code).isin(set(dd.n+'|'+dd.code)).mean())
# ICD code kinds: header ranges, non-billable categories
print(icd[icd.code.str.contains('-')].head())
icdc = icd.code.unique(); print('icd unique codes', len(icdc))
d = df[df.coding_system=='ICD10CM']
print('gold ICD length dist', d.code.str.replace('.','').str.len().value_counts())
# is gold code a parent of other codes (non-leaf)?
leaf = set(c for c in icdc if not any(o!=c and o.startswith(c) for o in [] ))
