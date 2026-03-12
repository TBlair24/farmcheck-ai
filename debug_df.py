import pandas as pd
from pathlib import Path

INV = Path('data/processed/inventory.csv')
print('reading inventory at', INV)
df = pd.read_csv(INV)
print('initial columns:', df.columns.tolist())

# normalize same logic as sample_dataset
if "class_label" not in df.columns:
    if "class" in df.columns:
        df = df.rename(columns={"class": "class_label"})
    else:
        raise KeyError("inventory missing both class_label and class")
print('after rename columns:', df.columns.tolist())


# inspect group objects to see their columns

def inspect_and_sample(x):
    print('group columns before sample', x.columns.tolist())
    return x.sample(min(len(x), 100), random_state=42)

# try sampling by index and then using original df to keep class_label
sampled_by_idx = (
    df.groupby('class_label', group_keys=False)
    .apply(lambda g: g.sample(min(len(g), 100), random_state=42))
)
print('sampled_by_idx index head:', sampled_by_idx.index[:5])

# now gather rows from original df by the indices we sampled
sampled2 = df.loc[sampled_by_idx.index].reset_index(drop=True)
print('recovered columns using original df:', sampled2.columns.tolist())
print('first recovered row:', sampled2.iloc[0].to_dict())

# stop further diagnostics here
exit(0)
print('after apply, sampled columns:', sampled.columns.tolist())
print('index names:', sampled.index.names)
print('sampled head:')
print(sampled.head())
print('\nnow reset_index without drop')
sampled2 = sampled.reset_index()
print('columns after reset_index:', sampled2.columns.tolist())
print('first sampled2 row:')
print(sampled2.iloc[0].to_dict())
