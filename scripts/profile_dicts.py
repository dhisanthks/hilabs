import pandas as pd, time
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',100)
t=time.time()
for f in ['icd10cm_codes.xlsx','SNOMEDCT_US_codes_and_descriptions.xlsx']:
    x = pd.ExcelFile(f)
    print(f, x.sheet_names)
    for s in x.sheet_names:
        d = x.parse(s, dtype=str)
        print(s, d.shape, list(d.columns)); print(d.head(8)); print(d.tail(3))
        d.to_parquet(f'data_{s}.parquet') if False else None
print(time.time()-t)
