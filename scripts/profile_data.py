import pandas as pd
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',80)
df = pd.read_csv('hackathon_train.csv')
print(df.shape); print(df.dtypes)
print(df.isna().sum())
print(df.coding_system.value_counts())
print(df.subconcept.value_counts())
print(pd.crosstab(df.subconcept, df.coding_system))
print('unique entities', df.entity.nunique(), 'unique chunks', df.chunk_text.nunique())
print('unique codes per system', df.groupby('coding_system').code.nunique())
# pairs
g = df.groupby(['entity','chunk_text'])
print('entity+chunk groups', g.ngroups)
print(g.coding_system.apply(lambda s: tuple(sorted(s))).value_counts().head(10))
# entity ambiguity: same entity -> multiple codes
for sys in ['ICD10CM','SNOMEDCT_US']:
    d = df[df.coding_system==sys]
    amb = d.groupby('entity').code.nunique()
    print(sys, 'entities', len(amb), 'ambiguous', (amb>1).sum(), 'rows', len(d))
    vc = d.code.value_counts()
    print(sys, 'codes freq==1:', (vc==1).sum(), 'of', len(vc))
print(df.sample(25, random_state=0)[['entity','coding_system','code','display','subconcept']])
print(df.chunk_text.str.len().describe())
