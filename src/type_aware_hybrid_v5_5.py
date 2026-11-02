import os
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# ============================================================
# TYPE-AWARE HYBRID V5.5
# STRATIFIED + LEAKAGE-SAFE RANDOM FOREST
# SPECIALIZED MISSING + DUPLICATE DETECTION
# ============================================================

print("=" * 72)
print("TYPE-AWARE HYBRID V5.5")
print("STRATIFIED + LEAKAGE-SAFE RANDOM FOREST")
print("SPECIALIZED MISSING + DUPLICATE DETECTION")
print("=" * 72)

# ============================================================
# PATHS
# ============================================================

V2_RESULTS = "results/metrics/v2/model_scores_v2.csv"
V2_FEATURES = "results/metrics/v2/ml_features_v2.csv"
GROUND_TRUTH = "results/metrics/ground_truth.csv"
RAW_DATA = "datasets/corrupted/online_retail_corrupted.csv"

OUTPUT_DIR = "results/metrics/v5_5"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading datasets...")

v2_results = pd.read_csv(V2_RESULTS)
v2_features = pd.read_csv(V2_FEATURES)
gt = pd.read_csv(GROUND_TRUTH)
raw = pd.read_csv(RAW_DATA)

print(f"V2 results       : {len(v2_results)}")
print(f"V2 features      : {len(v2_features)}")
print(f"Ground truth     : {len(gt)}")
print(f"Raw data         : {len(raw)}")

# ============================================================
# STABLE ROW ID
# ============================================================

if "Corrupted_Row_ID" not in raw.columns:
    print("\nRaw CSV does not contain Corrupted_Row_ID.")
    print("Creating from raw row order...")
    raw["Corrupted_Row_ID"] = np.arange(
        len(raw),
        dtype=np.int64
    )

if len(raw) != len(v2_results):
    raise ValueError(
        "Raw and V2 results row counts do not match."
    )

if len(raw) != len(v2_features):
    raise ValueError(
        "Raw and V2 features row counts do not match."
    )

# ============================================================
# ORIGINAL ROW ID
# ============================================================

if "Original_Row_ID" not in raw.columns:
    raw["Original_Row_ID"] = np.arange(
        len(raw),
        dtype=np.int64
    )

# ============================================================
# BUILD BASE DATAFRAME
# ============================================================

df = pd.DataFrame({
    "Corrupted_Row_ID": raw["Corrupted_Row_ID"].values,
    "Original_Row_ID": raw["Original_Row_ID"].values
})

for col in v2_results.columns:
    if col not in df.columns:
        df[col] = v2_results[col].values

for col in v2_features.columns:
    if col not in df.columns:
        df[col] = v2_features[col].values

print(f"Unique feature rows: {len(df)}")

# ============================================================
# GROUND TRUTH
# ============================================================

target_columns = [
    "Missing_Value",
    "Invalid_Value",
    "Outlier",
    "Inconsistency",
    "Duplicate"
]

gt_required = [
    "Original_Row_ID"
] + target_columns

missing_gt = [
    c for c in gt_required
    if c not in gt.columns
]

if missing_gt:
    raise ValueError(
        f"Missing ground-truth columns: {missing_gt}"
    )

gt_clean = gt[gt_required].copy()

gt_clean = gt_clean.drop_duplicates(
    subset=["Original_Row_ID"],
    keep="first"
)

df = df.merge(
    gt_clean,
    on="Original_Row_ID",
    how="left",
    validate="many_to_one"
)

print(f"After ground-truth merge: {len(df)}")

# ============================================================
# RAW CONTEXT
# ============================================================

raw_context_cols = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country"
]

for col in raw_context_cols:
    if col in raw.columns and col not in df.columns:
        df[col] = raw[col].values

print("Raw context columns prepared.")
print(f"After raw context merge: {len(df)}")

# ============================================================
# MERGE INTEGRITY
# ============================================================

if len(df) != len(raw):
    raise ValueError(
        "Merge explosion detected."
    )

if df["Corrupted_Row_ID"].duplicated().any():
    raise ValueError(
        "Duplicate Corrupted_Row_ID detected."
    )

df = df.sort_values(
    "Corrupted_Row_ID"
).reset_index(drop=True)

print("Merge integrity verified.")

# ============================================================
# TARGET CLEANUP
# ============================================================

for col in target_columns:
    df[col] = (
        pd.to_numeric(
            df[col],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

# ============================================================
# BASIC NUMERIC CLEANUP
# ============================================================

for col in [
    "Quantity",
    "Price",
    "Customer ID"
]:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

# ============================================================
# DATE FEATURES
# ============================================================

if "InvoiceDate" in df.columns:

    dt = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce"
    )

    df["V55_Hour"] = (
        dt.dt.hour
        .fillna(-1)
    )

    df["V55_DayOfWeek"] = (
        dt.dt.dayofweek
        .fillna(-1)
    )

    df["V55_DayOfMonth"] = (
        dt.dt.day
        .fillna(-1)
    )

    df["V55_Month"] = (
        dt.dt.month
        .fillna(-1)
    )

    df["V55_Year"] = (
        dt.dt.year
        .fillna(-1)
    )

    df["V55_IsWeekend"] = (
        df["V55_DayOfWeek"].isin([5, 6])
    ).astype(int)

    safe_hour = df["V55_Hour"].clip(
        lower=0
    )

    safe_day = df["V55_DayOfWeek"].clip(
        lower=0
    )

    safe_month = df["V55_Month"].clip(
        lower=1
    )

    df["V55_Hour_Sin"] = np.sin(
        2 * np.pi * safe_hour / 24
    )

    df["V55_Hour_Cos"] = np.cos(
        2 * np.pi * safe_hour / 24
    )

    df["V55_Day_Sin"] = np.sin(
        2 * np.pi * safe_day / 7
    )

    df["V55_Day_Cos"] = np.cos(
        2 * np.pi * safe_day / 7
    )

    df["V55_Month_Sin"] = np.sin(
        2 * np.pi * safe_month / 12
    )

    df["V55_Month_Cos"] = np.cos(
        2 * np.pi * safe_month / 12
    )

# ============================================================
# BASIC TRANSACTION FEATURES
# ============================================================

if "Quantity" in df.columns:

    df["V55_Abs_Quantity"] = (
        df["Quantity"].abs()
    )

    df["V55_Log_Quantity"] = np.log1p(
        df["Quantity"].abs()
    )

    df["V55_Negative_Quantity"] = (
        df["Quantity"] < 0
    ).astype(int)

    df["V55_Zero_Quantity"] = (
        df["Quantity"] == 0
    ).astype(int)

if "Price" in df.columns:

    df["V55_Abs_Price"] = (
        df["Price"].abs()
    )

    df["V55_Log_Price"] = np.log1p(
        df["Price"].abs()
    )

    df["V55_Negative_Price"] = (
        df["Price"] < 0
    ).astype(int)

    df["V55_Zero_Price"] = (
        df["Price"] == 0
    ).astype(int)

if (
    "Quantity" in df.columns
    and "Price" in df.columns
):

    df["V55_Total_Amount"] = (
        df["Quantity"].fillna(0)
        *
        df["Price"].fillna(0)
    )

    df["V55_Abs_Total_Amount"] = (
        df["V55_Total_Amount"].abs()
    )

    df["V55_Price_Quantity_Ratio"] = (
        df["Price"]
        /
        df["Quantity"].replace(
            0,
            np.nan
        )
    )

# ============================================================
# MISSINGNESS FEATURES
# ============================================================

missing_columns = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country"
]

missing_feature_names = []

for col in missing_columns:

    if col not in df.columns:
        continue

    feature_name = (
        "V55_Missing_"
        + col.replace(" ", "_")
    )

    df[feature_name] = (
        df[col].isna()
    ).astype(int)

    missing_feature_names.append(
        feature_name
    )

df["V55_Total_Missing_Count"] = (
    df[missing_feature_names]
    .sum(axis=1)
)

df["V55_Any_Missing"] = (
    df["V55_Total_Missing_Count"] > 0
).astype(int)

df["V55_All_Missing"] = (
    df["V55_Total_Missing_Count"]
    ==
    len(missing_feature_names)
).astype(int)

# ============================================================
# SPECIALIZED MISSINGNESS PATTERN
# ============================================================

print("\nBuilding specialized missingness features...")

# Missingness pattern is based only on observable fields.
# It does not use ground truth.

df["_V55_Missing_Pattern"] = (
    df[missing_feature_names]
    .astype(str)
    .agg("".join, axis=1)
)

missing_pattern_frequency = (
    df["_V55_Missing_Pattern"]
    .value_counts(
        dropna=False
    )
)

df["V55_Missing_Pattern_Frequency"] = (
    df["_V55_Missing_Pattern"]
    .map(
        missing_pattern_frequency
    )
    .fillna(1)
    .astype(float)
)

df["V55_Missing_Pattern_Rarity"] = (
    1.0 /
    df["V55_Missing_Pattern_Frequency"]
)

df["V55_Missing_Pattern_Log_Frequency"] = (
    np.log1p(
        df["V55_Missing_Pattern_Frequency"]
    )
)

# Individual missing combinations

df["V55_Core_Missing"] = (
    df.get(
        "V55_Missing_Invoice",
        pd.Series(
            0,
            index=df.index
        )
    )
    +
    df.get(
        "V55_Missing_StockCode",
        pd.Series(
            0,
            index=df.index
        )
    )
    +
    df.get(
        "V55_Missing_Description",
        pd.Series(
            0,
            index=df.index
        )
    )
)

df["V55_Financial_Missing"] = (
    df.get(
        "V55_Missing_Quantity",
        pd.Series(
            0,
            index=df.index
        )
    )
    +
    df.get(
        "V55_Missing_Price",
        pd.Series(
            0,
            index=df.index
        )
    )
)

df["V55_Identity_Missing"] = (
    df.get(
        "V55_Missing_Customer_ID",
        pd.Series(
            0,
            index=df.index
        )
    )
)

df["V55_Context_Missing"] = (
    df.get(
        "V55_Missing_InvoiceDate",
        pd.Series(
            0,
            index=df.index
        )
    )
    +
    df.get(
        "V55_Missing_Country",
        pd.Series(
            0,
            index=df.index
        )
    )
)

# ============================================================
# DESCRIPTION FEATURES
# ============================================================

if "Description" in df.columns:

    desc = (
        df["Description"]
        .fillna("")
        .astype(str)
    )

    df["V55_Description_Length"] = (
        desc.str.len()
    )

    df["V55_Description_Word_Count"] = (
        desc.str.split().str.len()
    )

    df["V55_Description_Is_Empty"] = (
        desc.str.strip() == ""
    ).astype(int)

    df["V55_Description_Has_Question"] = (
        desc.str.contains(
            r"\?",
            regex=True
        )
    ).astype(int)

    df["V55_Description_All_Caps"] = (
        desc.str.isupper()
    ).astype(int)

# ============================================================
# JOINT ERROR-PATTERN STRATIFIED SPLIT
# ============================================================

print("\nCreating STRATIFIED 60/20/20 split...")

pattern_key = (
    df[target_columns]
    .astype(str)
    .agg("_".join, axis=1)
)

rng = np.random.RandomState(42)

train_indices = []
val_indices = []
test_indices = []

for _, group in df.groupby(
    pattern_key,
    sort=True
):

    indices = np.array(
        group.index.tolist(),
        dtype=np.int64
    ).copy()

    rng.shuffle(indices)

    n = len(indices)

    n_train = int(
        round(n * 0.60)
    )

    n_val = int(
        round(n * 0.20)
    )

    if n_train + n_val > n:
        n_val = n - n_train

    train_indices.extend(
        indices[:n_train]
    )

    val_indices.extend(
        indices[
            n_train:
            n_train + n_val
        ]
    )

    test_indices.extend(
        indices[
            n_train + n_val:
        ]
    )

train_idx = np.array(
    train_indices,
    dtype=np.int64
).copy()

val_idx = np.array(
    val_indices,
    dtype=np.int64
).copy()

test_idx = np.array(
    test_indices,
    dtype=np.int64
).copy()

rng.shuffle(train_idx)
rng.shuffle(val_idx)
rng.shuffle(test_idx)

print(f"Train      : {len(train_idx)}")
print(f"Validation : {len(val_idx)}")
print(f"Test       : {len(test_idx)}")

train_set = set(train_idx)
val_set = set(val_idx)
test_set = set(test_idx)

if train_set & val_set:
    raise ValueError(
        "Train/validation overlap detected."
    )

if train_set & test_set:
    raise ValueError(
        "Train/test overlap detected."
    )

if val_set & test_set:
    raise ValueError(
        "Validation/test overlap detected."
    )

print("Split overlap check: PASSED")

# ============================================================
# SPLIT DISTRIBUTION
# ============================================================

print("\nError distribution by split:")

for error_type, col in [
    ("Missing", "Missing_Value"),
    ("Invalid", "Invalid_Value"),
    ("Outlier", "Outlier"),
    ("Inconsistency", "Inconsistency"),
    ("Duplicate", "Duplicate")
]:

    tr = int(
        df.iloc[train_idx][col].sum()
    )

    va = int(
        df.iloc[val_idx][col].sum()
    )

    te = int(
        df.iloc[test_idx][col].sum()
    )

    print(
        f"{error_type:<15}"
        f"Train={tr:5d} "
        f"({tr/len(train_idx)*100:6.3f}%)   "
        f"Val={va:5d} "
        f"({va/len(val_idx)*100:6.3f}%)   "
        f"Test={te:5d} "
        f"({te/len(test_idx)*100:6.3f}%)"
    )

# ============================================================
# TRAIN-ONLY CONTEXT
# ============================================================

print(
    "\nBuilding TRAIN-ONLY contextual statistics..."
)

train_df = df.iloc[
    train_idx
].copy()

# ============================================================
# PRODUCT CONTEXT
# ============================================================

if "StockCode" in df.columns:

    product_stats = (
        train_df
        .groupby(
            "StockCode",
            dropna=False
        )
        .agg(
            Train_Product_Frequency=(
                "StockCode",
                "size"
            ),
            Train_Product_Median_Price=(
                "Price",
                "median"
            ),
            Train_Product_Mean_Price=(
                "Price",
                "mean"
            ),
            Train_Product_Std_Price=(
                "Price",
                "std"
            ),
            Train_Product_Median_Quantity=(
                "Quantity",
                "median"
            ),
            Train_Product_Mean_Quantity=(
                "Quantity",
                "mean"
            ),
            Train_Product_Std_Quantity=(
                "Quantity",
                "std"
            ),
            Train_Product_Q1_Quantity=(
                "Quantity",
                lambda x: x.quantile(0.25)
            ),
            Train_Product_Q3_Quantity=(
                "Quantity",
                lambda x: x.quantile(0.75)
            )
        )
        .reset_index()
    )

    df = df.merge(
        product_stats,
        on="StockCode",
        how="left"
    )

# ============================================================
# CUSTOMER CONTEXT
# ============================================================

if "Customer ID" in df.columns:

    customer_stats = (
        train_df
        .groupby(
            "Customer ID",
            dropna=False
        )
        .size()
        .reset_index(
            name="Train_Customer_Frequency"
        )
    )

    df = df.merge(
        customer_stats,
        on="Customer ID",
        how="left"
    )

# ============================================================
# DESCRIPTION CONTEXT
# ============================================================

if "Description" in df.columns:

    description_stats = (
        train_df
        .groupby(
            "Description",
            dropna=False
        )
        .agg(
            Train_Description_Frequency=(
                "Description",
                "size"
            ),
            Train_Description_Product_Count=(
                "StockCode",
                "nunique"
            )
        )
        .reset_index()
    )

    df = df.merge(
        description_stats,
        on="Description",
        how="left"
    )

# ============================================================
# INVOICE CONTEXT
# ============================================================

if "Invoice" in df.columns:

    invoice_stats = (
        train_df
        .groupby(
            "Invoice",
            dropna=False
        )
        .size()
        .reset_index(
            name="Train_Invoice_Item_Count"
        )
    )

    df = df.merge(
        invoice_stats,
        on="Invoice",
        how="left"
    )

# ============================================================
# SPECIALIZED DUPLICATE EVIDENCE
# ============================================================

print(
    "\nBuilding specialized duplicate evidence..."
)

duplicate_signature_cols = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country"
]

duplicate_signature_cols = [
    c for c in duplicate_signature_cols
    if c in df.columns
]

signature_frame = (
    df[duplicate_signature_cols]
    .copy()
)

for col in signature_frame.columns:

    signature_frame[col] = (
        signature_frame[col]
        .fillna("<MISSING>")
        .astype(str)
    )

df["_V55_Duplicate_Signature"] = (
    signature_frame
    .agg(
        "||".join,
        axis=1
    )
)

signature_counts = (
    df["_V55_Duplicate_Signature"]
    .value_counts(
        dropna=False
    )
)

df["V55_Exact_Duplicate_Group_Size"] = (
    df["_V55_Duplicate_Signature"]
    .map(signature_counts)
    .fillna(1)
    .astype(float)
)

df["V55_Exact_Duplicate_Flag"] = (
    df["V55_Exact_Duplicate_Group_Size"] > 1
).astype(int)

df["V55_Duplicate_Group_Log_Size"] = (
    np.log1p(
        df["V55_Exact_Duplicate_Group_Size"]
    )
)

df["V55_Duplicate_Occurrence"] = (
    df.groupby(
        "_V55_Duplicate_Signature",
        sort=False
    ).cumcount()
)

df["V55_Duplicate_Is_First"] = (
    df["V55_Duplicate_Occurrence"] == 0
).astype(int)

df["V55_Duplicate_Is_Repeated"] = (
    df["V55_Duplicate_Occurrence"] > 0
).astype(int)

# ============================================================
# PARTIAL DUPLICATE SIGNATURE
# ============================================================

partial_duplicate_cols = [
    "StockCode",
    "Description",
    "Quantity",
    "Price",
    "Customer ID",
    "Country"
]

partial_duplicate_cols = [
    c for c in partial_duplicate_cols
    if c in df.columns
]

partial_frame = (
    df[partial_duplicate_cols]
    .copy()
)

for col in partial_frame.columns:

    partial_frame[col] = (
        partial_frame[col]
        .fillna("<MISSING>")
        .astype(str)
    )

df["_V55_Partial_Duplicate_Signature"] = (
    partial_frame
    .agg(
        "||".join,
        axis=1
    )
)

partial_counts = (
    df["_V55_Partial_Duplicate_Signature"]
    .value_counts(
        dropna=False
    )
)

df["V55_Partial_Duplicate_Group_Size"] = (
    df["_V55_Partial_Duplicate_Signature"]
    .map(partial_counts)
    .fillna(1)
    .astype(float)
)

df["V55_Partial_Duplicate_Flag"] = (
    df["V55_Partial_Duplicate_Group_Size"] > 1
).astype(int)

df["V55_Partial_Duplicate_Log_Size"] = (
    np.log1p(
        df["V55_Partial_Duplicate_Group_Size"]
    )
)

# ============================================================
# TRAIN-ONLY DUPLICATE CONTEXT
# ============================================================

train_signature_counts = (
    df.iloc[train_idx]
    .groupby(
        "_V55_Duplicate_Signature",
        dropna=False
    )
    .size()
    .rename(
        "Train_Duplicate_Group_Size"
    )
)

df = df.merge(
    train_signature_counts,
    left_on="_V55_Duplicate_Signature",
    right_index=True,
    how="left"
)

df["Train_Duplicate_Group_Size"] = (
    df["Train_Duplicate_Group_Size"]
    .fillna(1)
)

# ============================================================
# TRAIN-ONLY MISSING PATTERN CONTEXT
# ============================================================

train_missing_pattern_counts = (
    df.iloc[train_idx]
    .groupby(
        "_V55_Missing_Pattern",
        dropna=False
    )
    .size()
    .rename(
        "Train_Missing_Pattern_Frequency"
    )
)

df = df.merge(
    train_missing_pattern_counts,
    left_on="_V55_Missing_Pattern",
    right_index=True,
    how="left"
)

df["Train_Missing_Pattern_Frequency"] = (
    df[
        "Train_Missing_Pattern_Frequency"
    ]
    .fillna(0)
)

df["V55_Train_Missing_Pattern_Rarity"] = (
    1.0 /
    (
        df[
            "Train_Missing_Pattern_Frequency"
        ] + 1.0
    )
)

# ============================================================
# PRODUCT DEVIATION
# ============================================================

if "StockCode" in df.columns:

    df["V55_Product_IQR_Quantity"] = (
        df["Train_Product_Q3_Quantity"]
        -
        df["Train_Product_Q1_Quantity"]
    )

    df["V55_Product_Quantity_Deviation"] = (
        df["Quantity"]
        -
        df["Train_Product_Median_Quantity"]
    ).abs()

    df["V55_Product_Price_Deviation"] = (
        df["Price"]
        -
        df["Train_Product_Median_Price"]
    ).abs()

    df["V55_Product_Price_Ratio"] = (
        df["Price"]
        /
        df["Train_Product_Median_Price"]
    )

# ============================================================
# CLEAN MODEL FEATURE LIST
# ============================================================

identifier_columns = [
    "Corrupted_Row_ID",
    "Original_Row_ID",
    "Customer ID",
    "Original_Row_Occurrence",
    "Original_Row_ID_feature",
    "Original_Row_Occurrence_feature"
]

artifact_columns = [
    "V51_Incorrect_Description_Flag",
    "V52_Incorrect_Description_Flag",
    "V53_Incorrect_Description_Flag",
    "V54_Incorrect_Description_Flag",
    "V55_Incorrect_Description_Flag"
]

non_feature_columns = (
    target_columns
    +
    identifier_columns
    +
    artifact_columns
    +
    [
        "_V55_Duplicate_Signature",
        "_V55_Partial_Duplicate_Signature",
        "_V55_Missing_Pattern",
        "Invoice",
        "StockCode",
        "Description",
        "InvoiceDate",
        "Country"
    ]
)

feature_candidates = []

for col in df.columns:

    if col in non_feature_columns:
        continue

    if pd.api.types.is_numeric_dtype(
        df[col]
    ):
        feature_candidates.append(col)

features = []

for col in feature_candidates:

    if df[col].nunique(
        dropna=False
    ) <= 1:
        continue

    features.append(col)

features = [
    col
    for col in features
    if col not in target_columns
]

print("\n" + "=" * 72)
print("BUILDING CLEAN V5.5 MODEL FEATURES")
print("=" * 72)

print(
    f"Clean model features: {len(features)}"
)

for i, col in enumerate(
    features,
    start=1
):
    print(
        f"{i:3d}. {col}"
    )

# ============================================================
# FEATURE MATRIX
# ============================================================

X = df[features].copy()

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

train_medians = X.iloc[
    train_idx
].median(
    numeric_only=True
)

X = X.fillna(
    train_medians
)

X = X.fillna(0)

if X.isna().any().any():
    raise ValueError(
        "NaN values remain in feature matrix."
    )

# ============================================================
# SPECIALIZED SIGNALS
# ============================================================

# Duplicate signals
exact_duplicate_flag = (
    df[
        "V55_Exact_Duplicate_Flag"
    ]
    .astype(float)
    .values
)

exact_duplicate_size = (
    df[
        "V55_Exact_Duplicate_Group_Size"
    ]
    .astype(float)
    .values
)

partial_duplicate_flag = (
    df[
        "V55_Partial_Duplicate_Flag"
    ]
    .astype(float)
    .values
)

partial_duplicate_size = (
    df[
        "V55_Partial_Duplicate_Group_Size"
    ]
    .astype(float)
    .values
)

# Missing signals
missing_count = (
    df[
        "V55_Total_Missing_Count"
    ]
    .astype(float)
    .values
)

missing_pattern_rarity = (
    df[
        "V55_Train_Missing_Pattern_Rarity"
    ]
    .astype(float)
    .values
)

# ============================================================
# MODEL DEFINITIONS
# ============================================================

error_definitions = {
    "Missing": "Missing_Value",
    "Invalid": "Invalid_Value",
    "Outlier": "Outlier",
    "Inconsistency": "Inconsistency",
    "Duplicate": "Duplicate"
}

results = []
threshold_records = []
feature_importance_records = []

predicted_error_matrix = {}

print("\n")
print("=" * 72)
print("TRAINING TYPE-SPECIFIC RANDOM FORESTS")
print("=" * 72)

# ============================================================
# TRAIN EACH ERROR MODEL
# ============================================================

for error_type, target_col in (
    error_definitions.items()
):

    print("\n" + "-" * 72)
    print(
        f"ERROR TYPE: {error_type}"
    )
    print("-" * 72)

    y = (
        df[target_col]
        .astype(int)
        .values
    )

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    print(
        f"Train positives      : "
        f"{y_train.sum()}"
    )

    print(
        f"Validation positives : "
        f"{y_val.sum()}"
    )

    print(
        f"Test positives       : "
        f"{y_test.sum()}"
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=400,
        max_features="sqrt",
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    print(
        "Training Random Forest..."
    )

    model.fit(
        X.iloc[train_idx],
        y_train
    )

    val_probability = (
        model.predict_proba(
            X.iloc[val_idx]
        )[:, 1]
    )

    test_probability = (
        model.predict_proba(
            X.iloc[test_idx]
        )[:, 1]
    )

    # ========================================================
    # SPECIALIZED MISSING FUSION
    # ========================================================

    if error_type == "Missing":

        print(
            "Applying specialized "
            "missingness evidence fusion..."
        )

        val_missing_count = (
            missing_count[val_idx]
        )

        test_missing_count = (
            missing_count[test_idx]
        )

        val_missing_rarity = (
            missing_pattern_rarity[val_idx]
        )

        test_missing_rarity = (
            missing_pattern_rarity[test_idx]
        )

        # Evidence is bounded.
        #
        # Multiple missing fields provide stronger evidence.
        # Rare missingness patterns receive a small additional
        # boost.
        val_count_strength = np.minimum(
            val_missing_count / 3.0,
            1.0
        )

        test_count_strength = np.minimum(
            test_missing_count / 3.0,
            1.0
        )

        val_rarity_strength = np.minimum(
            val_missing_rarity * 10.0,
            1.0
        )

        test_rarity_strength = np.minimum(
            test_missing_rarity * 10.0,
            1.0
        )

        val_missing_evidence = (
            0.75 *
            val_count_strength
            +
            0.25 *
            val_rarity_strength
        )

        test_missing_evidence = (
            0.75 *
            test_count_strength
            +
            0.25 *
            test_rarity_strength
        )

        val_probability = (
            0.80 * val_probability
            +
            0.20 *
            val_missing_evidence
        )

        test_probability = (
            0.80 * test_probability
            +
            0.20 *
            test_missing_evidence
        )

    # ========================================================
    # SPECIALIZED DUPLICATE FUSION
    # ========================================================

    if error_type == "Duplicate":

        print(
            "Applying specialized "
            "duplicate evidence fusion..."
        )

        val_exact_flag = (
            exact_duplicate_flag[
                val_idx
            ]
        )

        test_exact_flag = (
            exact_duplicate_flag[
                test_idx
            ]
        )

        val_exact_size = (
            exact_duplicate_size[
                val_idx
            ]
        )

        test_exact_size = (
            exact_duplicate_size[
                test_idx
            ]
        )

        val_partial_flag = (
            partial_duplicate_flag[
                val_idx
            ]
        )

        test_partial_flag = (
            partial_duplicate_flag[
                test_idx
            ]
        )

        val_partial_size = (
            partial_duplicate_size[
                val_idx
            ]
        )

        test_partial_size = (
            partial_duplicate_size[
                test_idx
            ]
        )

        val_exact_strength = (
            val_exact_flag
            *
            np.minimum(
                np.log1p(
                    val_exact_size
                )
                /
                np.log(5),
                1.0
            )
        )

        test_exact_strength = (
            test_exact_flag
            *
            np.minimum(
                np.log1p(
                    test_exact_size
                )
                /
                np.log(5),
                1.0
            )
        )

        val_partial_strength = (
            val_partial_flag
            *
            np.minimum(
                np.log1p(
                    val_partial_size
                )
                /
                np.log(5),
                1.0
            )
        )

        test_partial_strength = (
            test_partial_flag
            *
            np.minimum(
                np.log1p(
                    test_partial_size
                )
                /
                np.log(5),
                1.0
            )
        )

        val_duplicate_evidence = (
            0.70 *
            val_exact_strength
            +
            0.30 *
            val_partial_strength
        )

        test_duplicate_evidence = (
            0.70 *
            test_exact_strength
            +
            0.30 *
            test_partial_strength
        )

        val_probability = (
            0.75 * val_probability
            +
            0.25 *
            val_duplicate_evidence
        )

        test_probability = (
            0.75 * test_probability
            +
            0.25 *
            test_duplicate_evidence
        )

    # ========================================================
    # VALIDATION THRESHOLD
    # ========================================================

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in np.arange(
        0.01,
        1.00,
        0.01
    ):

        val_pred = (
            val_probability
            >= threshold
        ).astype(int)

        score = f1_score(
            y_val,
            val_pred,
            zero_division=0
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = threshold

    print(
        f"Validation threshold : "
        f"{best_threshold:.2f}"
    )

    print(
        f"Validation F1        : "
        f"{best_f1 * 100:.2f}%"
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    test_pred = (
        test_probability
        >= best_threshold
    ).astype(int)

    predicted_error_matrix[
        error_type
    ] = test_pred

    accuracy = accuracy_score(
        y_test,
        test_pred
    )

    precision = precision_score(
        y_test,
        test_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        test_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        test_pred,
        zero_division=0
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_test,
            test_pred,
            labels=[0, 1]
        )
        .ravel()
    )

    print(
        f"Test Accuracy  : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Test Precision : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Test Recall    : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"Test F1        : "
        f"{f1 * 100:.2f}%"
    )

    results.append({
        "Error_Type": error_type,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "Threshold": best_threshold,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn
    })

    threshold_records.append({
        "Error_Type": error_type,
        "Best_Threshold": best_threshold,
        "Validation_F1": best_f1
    })

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    for feature, importance in zip(
        features,
        model.feature_importances_
    ):

        feature_importance_records.append({
            "Error_Type": error_type,
            "Feature": feature,
            "Importance": importance
        })

# ============================================================
# GLOBAL ANOMALY
# ============================================================

test_global_prediction = np.zeros(
    len(test_idx),
    dtype=int
)

for error_type in error_definitions:

    test_global_prediction = np.maximum(
        test_global_prediction,
        predicted_error_matrix[
            error_type
        ]
    )

test_global_truth = (
    df.iloc[test_idx][
        target_columns
    ]
    .max(axis=1)
    .astype(int)
    .values
)

overall_accuracy = accuracy_score(
    test_global_truth,
    test_global_prediction
)

overall_precision = precision_score(
    test_global_truth,
    test_global_prediction,
    zero_division=0
)

overall_recall = recall_score(
    test_global_truth,
    test_global_prediction,
    zero_division=0
)

overall_f1 = f1_score(
    test_global_truth,
    test_global_prediction,
    zero_division=0
)

tn, fp, fn, tp = (
    confusion_matrix(
        test_global_truth,
        test_global_prediction,
        labels=[0, 1]
    )
    .ravel()
)

# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 72)
print("V5.5 FINAL TEST RESULTS")
print("=" * 72)

print(
    f"Accuracy  : "
    f"{overall_accuracy * 100:.2f}%"
)

print(
    f"Precision : "
    f"{overall_precision * 100:.2f}%"
)

print(
    f"Recall    : "
    f"{overall_recall * 100:.2f}%"
)

print(
    f"F1 Score  : "
    f"{overall_f1 * 100:.2f}%"
)

print(f"TP        : {tp}")
print(f"FP        : {fp}")
print(f"TN        : {tn}")
print(f"FN        : {fn}")

# ============================================================
# PER-ERROR-TYPE RESULTS
# ============================================================

print("\n")
print("=" * 72)
print("PER-ERROR-TYPE PERFORMANCE")
print("=" * 72)

results_df = pd.DataFrame(
    results
)

display_df = results_df.copy()

for col in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score"
]:

    display_df[col] = (
        display_df[col]
        * 100
    ).map(
        lambda x: f"{x:.2f}%"
    )

print(
    display_df.to_string(
        index=False
    )
)

# ============================================================
# METHODOLOGY AUDIT
# ============================================================

print("\n")
print("=" * 72)
print("METHODOLOGY AUDIT")
print("=" * 72)

print(
    "Split strategy           : "
    "JOINT ERROR-PATTERN STRATIFIED"
)

print(
    "Context statistics       : "
    "TRAIN ONLY"
)

print(
    "Threshold optimization   : "
    "VALIDATION ONLY"
)

print(
    "Test set usage           : "
    "FINAL EVALUATION ONLY"
)

print(
    "Identifier features      : "
    "REMOVED"
)

print(
    "Customer ID feature      : "
    "REMOVED"
)

print(
    "Synthetic artifact flag  : "
    "REMOVED"
)

print(
    "Missingness specialization: "
    "ENABLED"
)

print(
    "Duplicate specialization : "
    "ENABLED"
)

print(
    "Duplicate signature      : "
    "UNSUPERVISED DQ SIGNAL"
)

print(
    f"Clean model features     : "
    f"{len(features)}"
)

# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "type_aware_hybrid_v5_5_results.csv"
    ),
    index=False
)

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "v5_5_error_type_performance.csv"
    ),
    index=False
)

pd.DataFrame(
    threshold_records
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "v5_5_threshold_optimization.csv"
    ),
    index=False
)

pd.DataFrame(
    feature_importance_records
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "v5_5_random_forest_feature_importance.csv"
    ),
    index=False
)

pd.DataFrame([{
    "Accuracy": overall_accuracy,
    "Precision": overall_precision,
    "Recall": overall_recall,
    "F1_Score": overall_f1,
    "TP": tp,
    "FP": fp,
    "TN": tn,
    "FN": fn
}]).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "v5_5_overall_performance.csv"
    ),
    index=False
)

pd.DataFrame({
    "Feature": features
}).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "v5_5_model_features.csv"
    ),
    index=False
)

# ============================================================
# SPLIT DISTRIBUTION FILE
# ============================================================

split_rows = []

for split_name, indices in [
    ("Train", train_idx),
    ("Validation", val_idx),
    ("Test", test_idx)
]:

    subset = df.iloc[indices]

    for error_type, col in (
        error_definitions.items()
    ):

        split_rows.append({
            "Split": split_name,
            "Error_Type": error_type,
            "Rows": len(subset),
            "Positive_Count": int(
                subset[col].sum()
            ),
            "Positive_Percentage": (
                subset[col].mean()
                * 100
            )
        })

pd.DataFrame(
    split_rows
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "v5_5_split_distribution.csv"
    ),
    index=False
)

# ============================================================
# TEST PREDICTIONS
# ============================================================

test_predictions = (
    df.iloc[test_idx][
        [
            "Corrupted_Row_ID",
            "Original_Row_ID"
        ]
    ]
    .copy()
)

for error_type, pred in (
    predicted_error_matrix.items()
):

    test_predictions[
        f"Predicted_{error_type}"
    ] = pred

test_predictions[
    "Predicted_Global_Anomaly"
] = test_global_prediction

test_predictions[
    "Actual_Global_Anomaly"
] = test_global_truth

test_predictions.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "v5_5_test_predictions.csv"
    ),
    index=False
)

# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 72)
print("FILES SAVED")
print("=" * 72)

for filename in [
    "type_aware_hybrid_v5_5_results.csv",
    "v5_5_error_type_performance.csv",
    "v5_5_threshold_optimization.csv",
    "v5_5_random_forest_feature_importance.csv",
    "v5_5_overall_performance.csv",
    "v5_5_model_features.csv",
    "v5_5_split_distribution.csv",
    "v5_5_test_predictions.csv"
]:

    print(
        os.path.join(
            OUTPUT_DIR,
            filename
        )
    )

print("\n")
print("=" * 72)
print("V5.5 COMPLETED SUCCESSFULLY.")
print("=" * 72)