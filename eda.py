from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.dpi"] = 100
sns.set(style="whitegrid")

OUTPUT_DIR = Path("EDA")
OUTPUT_DIR.mkdir(exist_ok=True)


def save_text(filename: str, text: str, mode: str = "w") -> None:
    """Small helper to write text files."""
    with open(OUTPUT_DIR / filename, mode, encoding="utf-8") as f:
        f.write(text)

df = pd.read_csv("SCF2022_fragility.csv")

# COLUMN TYPE
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

binary_cols = [
    col for col in numeric_cols
    if df[col].dropna().nunique() == 2
]

discrete_cats = [
    col for col in numeric_cols
    if 2 < df[col].dropna().nunique() <= 10
]

continuous_cols = [
    col for col in numeric_cols
    if df[col].dropna().nunique() > 10
]

categorical_like_cols = binary_cols + discrete_cats

save_text(
    "column_groups.txt",
    "BINARY COLUMNS:\n"
    + ", ".join(binary_cols) + "\n\n"
    + "DISCRETE CATEGORICAL-LIKE COLUMNS:\n"
    + ", ".join(discrete_cats) + "\n\n"
    + "CONTINUOUS NUMERIC COLUMNS:\n"
    + ", ".join(continuous_cols) + "\n\n"
)

print("Binary cols:", len(binary_cols))
print("Discrete categorical-like cols:", len(discrete_cats))
print("Continuous numeric cols:", len(continuous_cols))

# SUMMARY STATISTICS
numeric_summary = df[numeric_cols].describe()
numeric_summary.to_csv(OUTPUT_DIR / "numeric_summary.csv")

with open(OUTPUT_DIR / "value_counts.txt", "w", encoding="utf-8") as f:
    for col in categorical_like_cols:
        f.write(f"\n===== Value counts for {col} =====\n")
        f.write(str(df[col].value_counts(dropna=False)))
        f.write("\n")

if continuous_cols:
    corr = df[continuous_cols].corr()
    corr.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=300)
    plt.show()
else:
    print("No continuous columns available for correlation matrix.")

# OUTLIER ANALYSIS
def compute_iqr_outlier_stats(series: pd.Series) -> dict:

    s = series.dropna()
    if s.empty:
        return None

    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    mask = (series < lower) | (series > upper)
    n_outliers = mask.sum()
    n_non_missing = series.notna().sum()
    prop_outliers = n_outliers / n_non_missing if n_non_missing > 0 else np.nan

    return {
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "lower_bound": lower,
        "upper_bound": upper,
        "n_non_missing": n_non_missing,
        "n_outliers": n_outliers,
        "prop_outliers": prop_outliers,
    }


outlier_stats = []
any_outlier_mask = pd.Series(False, index=df.index)

for col in continuous_cols:
    stats = compute_iqr_outlier_stats(df[col])
    if stats is None:
        continue

    stats["column"] = col
    outlier_stats.append(stats)

    lower = stats["lower_bound"]
    upper = stats["upper_bound"]
    col_mask = (df[col] < lower) | (df[col] > upper)
    any_outlier_mask |= col_mask

if outlier_stats:
    outlier_df = pd.DataFrame(outlier_stats).set_index("column")
    outlier_df.to_csv(OUTPUT_DIR / "outlier_summary.csv")

    df_any_outlier = df[any_outlier_mask]
    df_any_outlier.to_csv(OUTPUT_DIR / "rows_with_any_outlier.csv", index=False)

    print(
        f"Outlier analysis done for {len(outlier_df)} continuous columns. "
        f"Rows with any outlier: {any_outlier_mask.sum()}."
    )
else:
    print("No continuous columns available for outlier analysis.")


def plot_hist_smart(series: pd.Series, colname: str):
    s = series.dropna()
    if s.nunique() <= 1:
        return 

    zero_count = (series == 0).sum()
    nonzero = s[s > 0]

    if nonzero.empty:
        return

    lower_clip = nonzero.quantile(0.005)
    upper_clip = nonzero.quantile(0.995)
    s_plot = nonzero[(nonzero >= lower_clip) & (nonzero <= upper_clip)]

    if s_plot.nunique() <= 1:
        s_plot = nonzero

    plt.figure(figsize=(6, 4))
    sns.histplot(s_plot, bins=40, stat="count")

    plt.title(
        f"{colname}: Histogram of Non-Zero Values\n"
        f"(zeros = {zero_count:,})"
    )
    plt.xlabel(colname)
    plt.ylabel("Count")

    plt.xlim(s_plot.min(), s_plot.max())

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"hist_{colname}_nonzero.png", dpi=300)
    plt.close()


hist_cols = continuous_cols if continuous_cols else numeric_cols
for col in hist_cols:
    plot_hist_smart(df[col], col)

box_cols = continuous_cols if continuous_cols else numeric_cols

if box_cols:
    plt.figure(figsize=(max(12, len(box_cols) * 0.4), 6))
    sns.boxplot(data=df[box_cols])
    plt.xticks(rotation=90)
    plt.title("Boxplots of Numeric Variables")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "boxplots.png", dpi=300)
    plt.show()
else:
    print("No columns available for boxplots.")

for col in categorical_like_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, data=df)
    plt.title(f"Countplot for {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"countplot_{col}.png", dpi=300)
    plt.close()

print("EDA complete — all results saved in the 'EDA' folder.")
