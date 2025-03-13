import pandas as pd

df = pd.read_csv("data/1120918-3BP_Baseline-and-events_MHH.csv")
# df_MH = df[df[df.columns[1]]==1]
# print(len(df_MH))
# df_WCH = df[df[df.columns[2]]==1]
# print(len(df_WCH))

# count_MH = ((df_MH[df.columns[21]] >= 140) | (df_MH[df.columns[22]] >= 90)).sum()
# print(count_MH)

# count_WCH = ((df_WCH[df.columns[21]] < 140) & (df_WCH[df.columns[22]] < 90)).sum()
# print(count_WCH)

df_low = df[(df[df.columns[21]] < 140) & (df[df.columns[22]] < 90)]
print((df_low[df.columns[1]]==0).sum())