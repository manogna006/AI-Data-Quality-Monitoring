import pandas as pd
import numpy as np
import os


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "datasets/corrupted/online_retail_corrupted.csv"
OUTPUT_FILE = "results/metrics/dq_rule_results.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading corrupted dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df)}")
print(f"Columns loaded: {len(df.columns)}")


# ============================================================
# INITIALIZE DQ FLAGS
# ============================================================

df["DQ_Missing_Value"] = 0
df["DQ_Invalid_Value"] = 0
df["DQ_Outlier"] = 0
df["DQ_Inconsistency"] = 0
df["DQ_Duplicate"] = 0


# ============================================================
# 1. COMPLETENESS CHECK
# ============================================================
# Missing Customer ID is treated as a completeness violation.

print("\nChecking completeness...")

df["DQ_Missing_Value"] = (
    df["Customer ID"].isna()
).astype(int)


# ============================================================
# 2. VALIDITY CHECK
# ============================================================
# Price should not be negative or zero.

print("Checking validity...")

df["DQ_Invalid_Value"] = (
    pd.to_numeric(df["Price"], errors="coerce") <= 0
).astype(int)


# ============================================================
# 3. OUTLIER CHECK
# ============================================================
# Use an IQR-based rule on Quantity.
#
# This is intentionally independent of the injected ground truth.
# Extreme quantities are flagged as potential outliers.

print("Checking numerical outliers...")

quantity = pd.to_numeric(df["Quantity"], errors="coerce")

q1 = quantity.quantile(0.25)
q3 = quantity.quantile(0.75)

iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

df["DQ_Outlier"] = (
    (quantity < lower_bound) |
    (quantity > upper_bound)
).astype(int)

print(f"Quantity Q1          : {q1:.4f}")
print(f"Quantity Q3          : {q3:.4f}")
print(f"Quantity IQR         : {iqr:.4f}")
print(f"Lower outlier bound  : {lower_bound:.4f}")
print(f"Upper outlier bound  : {upper_bound:.4f}")


# ============================================================
# 4. CONSISTENCY CHECK
# ============================================================
# For each StockCode, determine the most common Description.
#
# A different description for the same StockCode is considered
# a potential consistency violation.

print("\nChecking consistency...")

# Remove rows where StockCode or Description is missing
valid_mapping = df[
    df["StockCode"].notna() &
    df["Description"].notna()
].copy()

# Find the most common description for each StockCode
description_mode = (
    valid_mapping
    .groupby("StockCode")["Description"]
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
)

df["Expected_Description"] = df["StockCode"].map(description_mode)

df["DQ_Inconsistency"] = (
    df["Description"].notna() &
    df["Expected_Description"].notna() &
    (df["Description"] != df["Expected_Description"])
).astype(int)


# ============================================================
# 5. DUPLICATE CHECK
# ============================================================
# Original_Row_ID is intentionally excluded because duplicate
# injected records share the same Original_Row_ID.
#
# We identify duplicate business records using the transaction
# attributes.

print("Checking duplicate records...")

duplicate_columns = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country"
]

existing_duplicate_columns = [
    col for col in duplicate_columns if col in df.columns
]

df["DQ_Duplicate"] = (
    df.duplicated(
        subset=existing_duplicate_columns,
        keep=False
    )
).astype(int)


# ============================================================
# TOTAL DQ ERRORS
# ============================================================

dq_columns = [
    "DQ_Missing_Value",
    "DQ_Invalid_Value",
    "DQ_Outlier",
    "DQ_Inconsistency",
    "DQ_Duplicate"
]

df["DQ_Error_Count"] = df[dq_columns].sum(axis=1)

df["DQ_Anomaly"] = (
    df["DQ_Error_Count"] > 0
).astype(int)


# ============================================================
# DQ SCORE
# ============================================================
# Five rule categories are considered.
#
# 100 = no detected DQ problem
# 0   = all five categories violated

df["DQ_Row_Score"] = (
    100 -
    (df["DQ_Error_Count"] / len(dq_columns) * 100)
).clip(lower=0)


# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("       DQ RULE ENGINE RESULTS")
print("========================================")

print(f"Total records          : {len(df)}")

for col in dq_columns:
    print(
        f"{col.replace('DQ_', ''):22}: "
        f"{df[col].sum()}"
    )

print("----------------------------------------")
print(
    f"Rows with DQ errors   : "
    f"{df['DQ_Anomaly'].sum()}"
)

print(
    f"Average DQ row score  : "
    f"{df['DQ_Row_Score'].mean():.2f}"
)

print("========================================")


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

output_columns = [
    "Original_Row_ID",
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
    "DQ_Missing_Value",
    "DQ_Invalid_Value",
    "DQ_Outlier",
    "DQ_Inconsistency",
    "DQ_Duplicate",
    "DQ_Error_Count",
    "DQ_Anomaly",
    "DQ_Row_Score"
]

# Keep only columns that actually exist
output_columns = [
    col for col in output_columns
    if col in df.columns
]

df[output_columns].to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved DQ rule results to:")
print(OUTPUT_FILE)