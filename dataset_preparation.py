import pandas as pd
import re

# Load raw SCF file
df = pd.read_csv("SCFPcopy.csv")
df_cln = df.copy()

# Detect binary variables still coded as 1/2 and convert them to 0/1
binary_12_cols = [
    col for col in df_cln.columns
    if set(df_cln[col].dropna().unique()).issubset({1, 2})
]

# Convert from 1/2 → 0/1
df_cln[binary_12_cols] = df_cln[binary_12_cols].replace({1: 0, 2: 1})
df_cln[binary_12_cols] = df_cln[binary_12_cols].astype("Int64")

# =========================================================
# ====================== POOLING ==========================
# =========================================================

# If implicate ID exists, drop before pooling
if "Y1" in df_cln.columns:
    df_cln = df_cln.drop(columns=["Y1"])

# Identify continuous and binary variables
binary_cols = [c for c in df_cln.columns if set(df_cln[c].dropna().unique()).issubset({0, 1})]

continuous_cols = [
    c for c in df_cln.columns
    if c not in binary_cols and df_cln[c].dtype != "object" and c != "YY1"
]

# Pooling: mean for all numeric variables
agg_dict = {col: "mean" for col in continuous_cols + binary_cols}

pooled = df_cln.groupby("YY1", as_index=False).agg(agg_dict)

# Re-binarize: pooled binary vars (mean → 0/1)
pooled[binary_cols] = (pooled[binary_cols] >= 0.5).astype("Int64")

# Reorder columns to match original order
original_order = [c for c in df_cln.columns if c in pooled.columns]
pooled = pooled[original_order]

# Select only the variables needed for modeling
selected_vars = [
    # IDs
    "Y1", "YY1",

    # Demographics
    "HHSEX", "AGE", "AGECL", "EDCL", "MARRIED", "KIDS",
    "LIFECL", "FAMSTRUCT", "RACE",

    # Labor
    "LF", "OCCAT1", "OCCAT2",

    # Income
    "INCOME", "NORMINC", "WAGEINC", "BUSSEFARMINC",
    "INTDIVINC", "KGINC", "SSRETINC", "TRANSFOTHINC",

    # Wealth / assets
    "VEHIC", "HOUSES", "ORESRE", "NNRESRE", "BUS", "OTHNFIN",
    "CDS", "NMMF",  "STOCKS", "BOND", "RETQLIQ", "SAVBND", "OTHMA", "OTHFIN", 
    "STMUTF", "TFBMUTF", "GBMUTF", "OBMUTF", "COMUTF", "DEBT", "BNPL",

    # Financial stress / credit access
    "LATE", "LATE60", "HPAYDAY", "BNKRUPLAST5", "KNOWL",

    # Emergency coping strategies
    "EMERGBORR", "EMERGSAV", "EMERGPSTP", "EMERGCUT", "EMERGWORK",

    # Spending behavior,
    "SPENDMOR", "SPENDLESS", "EXPENSHILO",

    # Fragility vars (keep for now)
    "LIQ", "RENT", "FOODHOME", "FOODAWAY", "FOODDELV"
]

# Keep only variables that exist in the dataframe
selected_vars = [v for v in selected_vars if v in pooled.columns]
df_cln = pooled[selected_vars].copy()

# Create total food expenditure as a single variable
food_cols = [c for c in ["FOODHOME", "FOODAWAY", "FOODDELV"] if c in df_cln.columns]
df_cln["FOODSPEND"] = df_cln[food_cols].sum(axis=1, skipna=True)
df_cln = df_cln.drop(columns=food_cols)

# One-hot encode key categorical variables
onehot_vars = [v for v in [
    "RACE", "EDCL", "OCCAT1", "OCCAT2", "AGECL",
    "LIFECL", "FAMSTRUCT", "KNOWL", "SPENDMOR",
    "SPENDLESS", "EXPENSHILO"] if v in df_cln.columns]

df_cln = pd.get_dummies(
    df_cln,
    columns=onehot_vars,
    prefix=onehot_vars,
    drop_first=False,
    dtype=int  # ensure output is 0/1 and not True/False
)

# Identify one-hot dummy families (pattern: NAME_NUMBER)
dummy_pattern = re.compile(r"^(.+?)_\d+(\.\d+)?$")

dummy_groups = {}

for col in df_cln.columns:
    m = dummy_pattern.match(col)
    if m:
        prefix = m.group(1)
        dummy_groups.setdefault(prefix, []).append(col)

# For each family: merge rare categories (<5 cases) into OTHER
for prefix, cols in dummy_groups.items():
    # Count how many 1s in each dummy
    counts = df_cln[cols].sum()

    # Identify rare dummies
    rare_cols = counts[counts <= 15].index.tolist()

    if len(rare_cols) == 0:
        continue  # nothing to 

    # Create OTHER column = 1 if any of the rare columns == 1
    df_cln[f"{prefix}_OTHER"] = df_cln[rare_cols].max(axis=1)

    # Drop the original rare dummy columns
    df_cln.drop(columns=rare_cols, inplace=True)

# Save final cleaned dataset
df_cln.to_csv("SCF2022_READY.csv", index=False)

print("\nSaved READY dataset as SCF2022_READY.csv")
