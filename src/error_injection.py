import pandas as pd
import numpy as np
import os

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "datasets/raw/online_retail_II.xlsx"

CORRUPTED_FILE = "datasets/corrupted/online_retail_corrupted.csv"
GROUND_TRUTH_FILE = "results/metrics/ground_truth.csv"

ERROR_RATE = 0.01
RANDOM_STATE = 42

os.makedirs("datasets/corrupted", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading original dataset...")

df = pd.read_excel(INPUT_FILE)

print(f"Original rows: {len(df)}")
print(f"Original columns: {len(df.columns)}")


# ============================================================
# ADD UNIQUE ORIGINAL ROW ID
# ============================================================

df.insert(
    0,
    "Original_Row_ID",
    np.arange(len(df))
)

# Keep a clean copy for comparison
original_df = df.copy()


# ============================================================
# CREATE ROW-LEVEL GROUND TRUTH
# ============================================================

ground_truth = pd.DataFrame({
    "Original_Row_ID": df["Original_Row_ID"],
    "Missing_Value": 0,
    "Invalid_Value": 0,
    "Outlier": 0,
    "Inconsistency": 0,
    "Duplicate": 0
})


# ============================================================
# RANDOM NUMBER GENERATOR
# ============================================================

rng = np.random.default_rng(RANDOM_STATE)

n_rows = len(df)

error_count = int(n_rows * ERROR_RATE)

print(f"\nTarget errors per category: {error_count}")


# ============================================================
# 1. MISSING VALUE ERROR
# ============================================================

print("\nInjecting missing-value errors...")

# Only select rows where Customer ID currently exists
valid_missing_indices = df[
    df["Customer ID"].notna()
].index.to_numpy()

missing_count = min(
    error_count,
    len(valid_missing_indices)
)

missing_indices = rng.choice(
    valid_missing_indices,
    size=missing_count,
    replace=False
)

df.loc[
    missing_indices,
    "Customer ID"
] = np.nan

ground_truth.loc[
    ground_truth["Original_Row_ID"].isin(
        df.loc[missing_indices, "Original_Row_ID"]
    ),
    "Missing_Value"
] = 1


# ============================================================
# 2. INVALID VALUE ERROR
# ============================================================

print("Injecting invalid-value errors...")

available_indices = np.setdiff1d(
    df.index.to_numpy(),
    missing_indices
)

invalid_indices = rng.choice(
    available_indices,
    size=error_count,
    replace=False
)

# Negative price is invalid
df.loc[
    invalid_indices,
    "Price"
] = -abs(
    df.loc[invalid_indices, "Price"]
)

ground_truth.loc[
    ground_truth["Original_Row_ID"].isin(
        df.loc[invalid_indices, "Original_Row_ID"]
    ),
    "Invalid_Value"
] = 1


# ============================================================
# 3. OUTLIER ERROR
# ============================================================

print("Injecting outlier errors...")

available_indices = np.setdiff1d(
    available_indices,
    invalid_indices
)

outlier_indices = rng.choice(
    available_indices,
    size=error_count,
    replace=False
)

# Create extreme Quantity values
original_quantity = df.loc[
    outlier_indices,
    "Quantity"
].abs()

outlier_values = original_quantity * 100

# Prevent zero from remaining zero
outlier_values = outlier_values.replace(
    0,
    1000
)

df.loc[
    outlier_indices,
    "Quantity"
] = outlier_values

ground_truth.loc[
    ground_truth["Original_Row_ID"].isin(
        df.loc[outlier_indices, "Original_Row_ID"]
    ),
    "Outlier"
] = 1


# ============================================================
# 4. INCONSISTENCY ERROR
# ============================================================

print("Injecting inconsistency errors...")

available_indices = np.setdiff1d(
    available_indices,
    outlier_indices
)

inconsistency_indices = rng.choice(
    available_indices,
    size=error_count,
    replace=False
)

df.loc[
    inconsistency_indices,
    "Description"
] = "INCORRECT_PRODUCT_DESCRIPTION"

ground_truth.loc[
    ground_truth["Original_Row_ID"].isin(
        df.loc[inconsistency_indices, "Original_Row_ID"]
    ),
    "Inconsistency"
] = 1


# ============================================================
# 5. DUPLICATE ERROR
# ============================================================

print("Injecting duplicate errors...")

available_indices = np.setdiff1d(
    available_indices,
    inconsistency_indices
)

duplicate_indices = rng.choice(
    available_indices,
    size=error_count,
    replace=False
)

duplicate_rows = df.loc[
    duplicate_indices
].copy()

ground_truth.loc[
    ground_truth["Original_Row_ID"].isin(
        df.loc[duplicate_indices, "Original_Row_ID"]
    ),
    "Duplicate"
] = 1


# ============================================================
# APPEND DUPLICATED RECORDS
# ============================================================

df_corrupted = pd.concat(
    [
        df,
        duplicate_rows
    ],
    ignore_index=True
)


# ============================================================
# SAVE CORRUPTED DATASET
# ============================================================

print("\nSaving corrupted dataset...")

df_corrupted.to_csv(
    CORRUPTED_FILE,
    index=False
)


# ============================================================
# SAVE GROUND TRUTH
# ============================================================

ground_truth.to_csv(
    GROUND_TRUTH_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("       CONTROLLED ERROR INJECTION RESULTS")
print("=" * 60)

print(f"Original rows       : {len(original_df)}")
print(f"Corrupted rows      : {len(df_corrupted)}")

print("\nInjected error counts:")

print(
    f"Missing Value      : "
    f"{ground_truth['Missing_Value'].sum()}"
)

print(
    f"Invalid Value      : "
    f"{ground_truth['Invalid_Value'].sum()}"
)

print(
    f"Outlier            : "
    f"{ground_truth['Outlier'].sum()}"
)

print(
    f"Inconsistency      : "
    f"{ground_truth['Inconsistency'].sum()}"
)

print(
    f"Duplicate           : "
    f"{ground_truth['Duplicate'].sum()}"
)

print(
    f"\nRows containing at least one injected error: "
    f"{(
        ground_truth[
            [
                "Missing_Value",
                "Invalid_Value",
                "Outlier",
                "Inconsistency",
                "Duplicate"
            ]
        ].sum(axis=1) > 0
    ).sum()}"
)

print("\nFiles saved:")
print(CORRUPTED_FILE)
print(GROUND_TRUTH_FILE)

print("=" * 60)