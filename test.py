#%%
import pandas as pd

df = pd.read_excel('this_season_excel.xlsx')

# %%
def add_cols(row):
    return 50, row["SOW"] + row["SOL"], row["SOW"] - row["SOL"]

# %%
df[['constant','sum','diff']] = df.apply(add_cols, axis = 1, result_type='expand')
# %%
