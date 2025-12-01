import pandas as pd

df = pd.read_csv("SCF2022_READY.csv")

df["liquid_assets"] = df[["LIQ"]].sum(axis=1, skipna=True)
df["monthly_expenses"] = (df["FOODSPEND"] / 12) + df["RENT"]
df["fragile"] = (df["liquid_assets"] < 3 * df["monthly_expenses"]).astype(int)
df = df.drop(columns=df[["LIQ", "FOODSPEND", "RENT", "liquid_assets", "monthly_expenses"]])

df.to_csv("SCF2022_fragility.csv", index=False)