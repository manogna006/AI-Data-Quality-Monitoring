# ============================================================
# TYPE-AWARE HYBRID V5.2
# Leakage-Safe Contextual Random Forest Ensemble
#
# Key methodological improvements over V5.1:
#   1. Contextual statistics computed from TRAIN ONLY
#   2. Validation-only threshold optimization
#   3. Test set used only for final evaluation
#   4. Removes ID/provenance features
#   5. Removes direct synthetic corruption artifact detector
#   6. Keeps legitimate observable data-quality signals
#   7. Separate output directory: results/metrics/v5_2/
# ============================================================

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
# CONFIGURATION
# ============================================================

V2_RESULTS = "results/metrics/v2/model_scores_v2.csv"
V2_FEATURES = "results/metrics/v2/ml_features_v2.csv"
GROUND_TRUTH = "results/metrics/ground_truth.csv"
RAW_DATA = "datasets/corrupted/online_retail_corrupted.csv"

OUTPUT_DIR = "results/metrics/v5_2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20

RF_TREES = 350
MIN_SAMPLES_SPLIT = 4
MIN_SAMPLES_LEAF = 2
MAX_FEATURES = "sqrt"

ERROR_TYPES = [
    "Missing",
    "Invalid",
    "Outlier",
    "Inconsistency",
    "Duplicate"
]

GT_COLUMNS = [
    "Missing_Value",
    "Invalid_Value",
    "Outlier",
    "Inconsistency",
    "Duplicate"
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def safe_divide(a, b):
    return np.where(
        np.abs(b) > 1e-12,
        a / b,
        0.0
    )


def optimize_threshold(y_true, scores):
    """
    Optimize threshold ONLY on validation data.
    Objective: F1.
    """

    thresholds = np.linspace(0.01, 0.99, 99)

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in thresholds:

        pred = (scores >= threshold).astype(int)

        current_f1 = f1_score(
            y_true,
            pred,
            zero_division=0
        )

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    return best_threshold, best_f1


def calculate_metrics(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {
        "Accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "F1_Score": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,

        "Predicted_Anomalies": int(
            np.sum(y_pred)
        ),

        "Actual_Anomalies": int(
            np.sum(y_true)
        )
    }


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("TYPE-AWARE HYBRID V5.2")
print("LEAKAGE-SAFE CONTEXTUAL RANDOM FOREST")
print("=" * 70)

print("\nLoading datasets...")

v2 = pd.read_csv(V2_RESULTS)
features = pd.read_csv(V2_FEATURES)
ground_truth = pd.read_csv(GROUND_TRUTH)
raw = pd.read_csv(RAW_DATA)

print(f"V2 results       : {len(v2)}")
print(f"V2 features      : {len(features)}")
print(f"Ground truth     : {len(ground_truth)}")
print(f"Raw data         : {len(raw)}")


# ============================================================
# VALIDATE REQUIRED KEYS
# ============================================================

if "Corrupted_Row_ID" not in v2.columns:
    raise ValueError(
        "V2 results does not contain Corrupted_Row_ID."
    )

if "Original_Row_ID" not in v2.columns:
    raise ValueError(
        "V2 results does not contain Original_Row_ID."
    )

if "Original_Row_ID" not in ground_truth.columns:
    raise ValueError(
        "Ground truth does not contain Original_Row_ID."
    )


# ============================================================
# PREPARE RAW DATA KEY
# ============================================================

if "Corrupted_Row_ID" not in raw.columns:

    print(
        "\nRaw CSV does not contain Corrupted_Row_ID."
    )

    print(
        "Creating it from raw row order..."
    )

    raw.insert(
        0,
        "Corrupted_Row_ID",
        np.arange(len(raw))
    )

else:

    raw["Corrupted_Row_ID"] = pd.to_numeric(
        raw["Corrupted_Row_ID"],
        errors="coerce"
    )


v2["Corrupted_Row_ID"] = pd.to_numeric(
    v2["Corrupted_Row_ID"],
    errors="coerce"
)

raw["Corrupted_Row_ID"] = pd.to_numeric(
    raw["Corrupted_Row_ID"],
    errors="coerce"
)


# ============================================================
# VERIFY RAW/V2 ALIGNMENT
# ============================================================

if len(raw) != len(v2):

    raise ValueError(
        "Raw data and V2 results have different row counts."
    )

if not np.array_equal(
    raw["Corrupted_Row_ID"].values,
    v2["Corrupted_Row_ID"].values
):

    raise ValueError(
        "Raw data and V2 results are not aligned by "
        "Corrupted_Row_ID."
    )

print(
    "Raw/V2 row-key alignment verified."
)


# ============================================================
# PREPARE V2 RESULTS
# ============================================================

# These columns may already exist inside V2 results.
# Remove them before attaching the authoritative GT.

existing_label_columns = [
    col
    for col in ERROR_TYPES
    if col in v2.columns
]

if existing_label_columns:

    print(
        "\nRemoving existing V2 label columns:"
    )

    print(existing_label_columns)

    v2 = v2.drop(
        columns=existing_label_columns
    )


# Remove possible suffix-created labels if present.

suffix_columns = [
    col
    for col in v2.columns
    if col.endswith("_x") or col.endswith("_y")
]

# Only remove suffix columns corresponding to GT labels.

suffix_label_columns = [
    col
    for col in suffix_columns
    if col.replace("_x", "").replace("_y", "")
    in GT_COLUMNS
]

if suffix_label_columns:

    v2 = v2.drop(
        columns=suffix_label_columns
    )


# ============================================================
# PREPARE FEATURE TABLE
# ============================================================

features = features.copy()

if "Corrupted_Row_ID" not in features.columns:

    raise ValueError(
        "V2 features does not contain Corrupted_Row_ID."
    )

features["Corrupted_Row_ID"] = pd.to_numeric(
    features["Corrupted_Row_ID"],
    errors="coerce"
)

features = features.drop_duplicates(
    subset=["Corrupted_Row_ID"],
    keep="first"
)

print(
    f"\nUnique feature rows: {len(features)}"
)


# ============================================================
# MERGE V2 + FEATURES
# ============================================================

df = v2.merge(
    features,
    on="Corrupted_Row_ID",
    how="left",
    suffixes=("", "_feature")
)

print(
    f"After V2 + feature merge: {len(df)}"
)

if len(df) != len(v2):

    raise ValueError(
        "V2 + feature merge changed row count."
    )


# ============================================================
# PREPARE GROUND TRUTH
# ============================================================

ground_truth = ground_truth.copy()

ground_truth["Original_Row_ID"] = pd.to_numeric(
    ground_truth["Original_Row_ID"],
    errors="coerce"
)

ground_truth = ground_truth.drop_duplicates(
    subset=["Original_Row_ID"],
    keep="first"
)


for col in GT_COLUMNS:

    if col not in ground_truth.columns:

        raise ValueError(
            f"Ground truth missing required column: {col}"
        )

    ground_truth[col] = (
        pd.to_numeric(
            ground_truth[col],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )


# ============================================================
# MERGE GROUND TRUTH
# ============================================================

df = df.merge(
    ground_truth[
        ["Original_Row_ID"] + GT_COLUMNS
    ],
    on="Original_Row_ID",
    how="left"
)

print(
    f"After ground-truth merge: {len(df)}"
)

if len(df) != len(v2):

    raise ValueError(
        "Ground-truth merge changed row count."
    )


# ============================================================
# ADD RAW CONTEXT ONLY WHEN REQUIRED
# ============================================================

raw = raw.copy()

raw_columns_to_add = []

for col in raw.columns:

    if col == "Corrupted_Row_ID":
        continue

    if col not in df.columns:

        raw_columns_to_add.append(col)


if raw_columns_to_add:

    raw_subset = raw[
        ["Corrupted_Row_ID"] + raw_columns_to_add
    ].copy()

    raw_subset = raw_subset.drop_duplicates(
        subset=["Corrupted_Row_ID"],
        keep="first"
    )

    df = df.merge(
        raw_subset,
        on="Corrupted_Row_ID",
        how="left"
    )

    print(
        f"Raw context columns added: "
        f"{len(raw_columns_to_add)}"
    )

else:

    print(
        "All required raw context columns already exist."
    )


print(
    f"After raw context merge: {len(df)}"
)

if len(df) != len(v2):

    raise ValueError(
        "Raw context merge changed row count."
    )

print(
    "Merge integrity verified."
)


# ============================================================
# CLEAN BASIC DATA TYPES
# ============================================================

if "InvoiceDate" in df.columns:

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce"
    )


numeric_source_columns = [
    "Quantity",
    "Price",
    "Customer ID"
]

for col in numeric_source_columns:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


# ============================================================
# CREATE OBSERVABLE FEATURES
# ============================================================

print("\nCreating observable features...")

# ------------------------------------------------------------
# Basic numeric features
# ------------------------------------------------------------

if "Quantity" in df.columns:

    q = safe_numeric(df["Quantity"])

else:

    q = pd.Series(
        np.zeros(len(df)),
        index=df.index
    )


if "Price" in df.columns:

    p = safe_numeric(df["Price"])

else:

    p = pd.Series(
        np.zeros(len(df)),
        index=df.index
    )


df["V52_Abs_Quantity"] = q.abs()

df["V52_Log_Quantity"] = np.log1p(
    q.abs()
)

df["V52_Negative_Quantity"] = (
    q < 0
).astype(int)

df["V52_Zero_Quantity"] = (
    q == 0
).astype(int)


df["V52_Abs_Price"] = p.abs()

df["V52_Log_Price"] = np.log1p(
    p.abs()
)

df["V52_Negative_Price"] = (
    p < 0
).astype(int)

df["V52_Zero_Price"] = (
    p == 0
).astype(int)


df["V52_Total_Amount"] = (
    q * p
)

df["V52_Abs_Total_Amount"] = (
    df["V52_Total_Amount"].abs()
)

df["V52_Price_Quantity_Ratio"] = (
    safe_divide(
        p.values,
        q.abs().values + 1e-6
    )
)


# ------------------------------------------------------------
# Missingness indicators
# ------------------------------------------------------------

observable_missing_columns = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country"
]

missing_feature_columns = []

for col in observable_missing_columns:

    if col in df.columns:

        feature_name = (
            "V52_Missing_"
            + col.replace(" ", "_")
        )

        df[feature_name] = (
            df[col].isna()
            .astype(int)
        )

        missing_feature_columns.append(
            feature_name
        )


if missing_feature_columns:

    df["V52_Total_Missing_Count"] = (
        df[missing_feature_columns]
        .sum(axis=1)
    )

else:

    df["V52_Total_Missing_Count"] = 0


# ------------------------------------------------------------
# Description characteristics
# ------------------------------------------------------------

if "Description" in df.columns:

    description = (
        df["Description"]
        .fillna("")
        .astype(str)
    )

    df["V52_Description_Length"] = (
        description.str.len()
    )

    df["V52_Description_Word_Count"] = (
        description.str.split().str.len()
    )

    df["V52_Description_Is_Empty"] = (
        description.str.strip().eq("")
        .astype(int)
    )

    # Generic suspicious-description features.
    # IMPORTANT:
    # We do NOT check for the exact synthetic corruption
    # string "INCORRECT_PRODUCT_DESCRIPTION".

    df["V52_Description_Has_Question_Mark"] = (
        description.str.contains(
            r"\?",
            regex=True,
            na=False
        )
        .astype(int)
    )

    df["V52_Description_All_Caps"] = (
        description.str.isupper()
        .astype(int)
    )

else:

    df["V52_Description_Length"] = 0
    df["V52_Description_Word_Count"] = 0
    df["V52_Description_Is_Empty"] = 1
    df["V52_Description_Has_Question_Mark"] = 0
    df["V52_Description_All_Caps"] = 0


# ------------------------------------------------------------
# Date/time features
# ------------------------------------------------------------

if "InvoiceDate" in df.columns:

    dt = df["InvoiceDate"]

    df["V52_Hour"] = (
        dt.dt.hour
        .fillna(-1)
    )

    df["V52_DayOfWeek"] = (
        dt.dt.dayofweek
        .fillna(-1)
    )

    df["V52_DayOfMonth"] = (
        dt.dt.day
        .fillna(-1)
    )

    df["V52_Month"] = (
        dt.dt.month
        .fillna(-1)
    )

    df["V52_Year"] = (
        dt.dt.year
        .fillna(-1)
    )

    df["V52_IsWeekend"] = (
        dt.dt.dayofweek
        .isin([5, 6])
        .astype(int)
    )

else:

    for col in [
        "V52_Hour",
        "V52_DayOfWeek",
        "V52_DayOfMonth",
        "V52_Month",
        "V52_Year",
        "V52_IsWeekend"
    ]:

        df[col] = 0


# ============================================================
# DETERMINISTIC TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\nCreating deterministic split...")

df = df.sort_values(
    "Corrupted_Row_ID"
).reset_index(drop=True)

n = len(df)

train_end = int(
    n * TRAIN_RATIO
)

val_end = int(
    n * (TRAIN_RATIO + VAL_RATIO)
)

train_mask = (
    np.arange(n) < train_end
)

val_mask = (
    (np.arange(n) >= train_end)
    &
    (np.arange(n) < val_end)
)

test_mask = (
    np.arange(n) >= val_end
)

train_idx = df.index[train_mask]
val_idx = df.index[val_mask]
test_idx = df.index[test_mask]

print(
    f"Train      : {len(train_idx)}"
)

print(
    f"Validation : {len(val_idx)}"
)

print(
    f"Test       : {len(test_idx)}"
)


# ============================================================
# TRAIN-ONLY CONTEXTUAL STATISTICS
# ============================================================

print(
    "\nBuilding TRAIN-ONLY contextual statistics..."
)

train_df = df.loc[train_idx].copy()


# ------------------------------------------------------------
# Product context
# ------------------------------------------------------------

if "StockCode" in df.columns:

    product_train = (
        train_df
        .groupby("StockCode", dropna=False)
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

else:

    product_train = pd.DataFrame()


# ------------------------------------------------------------
# Customer context
# ------------------------------------------------------------

if "Customer ID" in df.columns:

    customer_train = (
        train_df
        .groupby("Customer ID", dropna=False)
        .size()
        .reset_index(
            name="Train_Customer_Frequency"
        )
    )

else:

    customer_train = pd.DataFrame()


# ------------------------------------------------------------
# Description context
# ------------------------------------------------------------

if (
    "Description" in df.columns
    and "StockCode" in df.columns
):

    description_train = (
        train_df
        .groupby(
            "Description",
            dropna=False
        )
        .size()
        .reset_index(
            name="Train_Description_Frequency"
        )
    )

    description_product_train = (
        train_df
        .groupby(
            "Description",
            dropna=False
        )["StockCode"]
        .nunique()
        .reset_index(
            name="Train_Description_Product_Count"
        )
    )

else:

    description_train = pd.DataFrame()

    description_product_train = (
        pd.DataFrame()
    )


# ------------------------------------------------------------
# Invoice context
# ------------------------------------------------------------

if "Invoice" in df.columns:

    invoice_train = (
        train_df
        .groupby("Invoice", dropna=False)
        .size()
        .reset_index(
            name="Train_Invoice_Item_Count"
        )
    )

else:

    invoice_train = pd.DataFrame()


# ------------------------------------------------------------
# Duplicate-group context
# ------------------------------------------------------------

duplicate_group_columns = [
    col
    for col in [
        "Invoice",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Customer ID",
        "Country"
    ]
    if col in df.columns
]

if duplicate_group_columns:

    duplicate_train = (
        train_df
        .groupby(
            duplicate_group_columns,
            dropna=False
        )
        .size()
        .reset_index(
            name="Train_Duplicate_Group_Size"
        )
    )

else:

    duplicate_train = pd.DataFrame()


# ============================================================
# MAP TRAIN-ONLY CONTEXT TO ALL SPLITS
# ============================================================

def map_product_context(target):

    if product_train.empty:
        return target

    return target.merge(
        product_train,
        on="StockCode",
        how="left"
    )


def map_customer_context(target):

    if customer_train.empty:
        return target

    return target.merge(
        customer_train,
        on="Customer ID",
        how="left"
    )


def map_description_context(target):

    if description_train.empty:
        return target

    target = target.merge(
        description_train,
        on="Description",
        how="left"
    )

    target = target.merge(
        description_product_train,
        on="Description",
        how="left"
    )

    return target


def map_invoice_context(target):

    if invoice_train.empty:
        return target

    return target.merge(
        invoice_train,
        on="Invoice",
        how="left"
    )


def map_duplicate_context(target):

    if duplicate_train.empty:
        return target

    return target.merge(
        duplicate_train,
        on=duplicate_group_columns,
        how="left"
    )


# ------------------------------------------------------------
# Preserve row order through temporary key
# ------------------------------------------------------------

df["_V52_Internal_Row_Index"] = np.arange(
    len(df)
)

df = map_product_context(df)
df = map_customer_context(df)
df = map_description_context(df)
df = map_invoice_context(df)
df = map_duplicate_context(df)

df = df.sort_values(
    "_V52_Internal_Row_Index"
).reset_index(drop=True)

df.drop(
    columns=["_V52_Internal_Row_Index"],
    inplace=True
)


# ============================================================
# DERIVED TRAIN-ONLY CONTEXT FEATURES
# ============================================================

if "Train_Product_Q1_Quantity" in df.columns:

    df["V52_Product_IQR_Quantity"] = (
        df["Train_Product_Q3_Quantity"]
        - df["Train_Product_Q1_Quantity"]
    )

    df["V52_Product_Quantity_Deviation"] = (
        safe_divide(
            (
                q
                - df["Train_Product_Median_Quantity"]
            ).values,
            (
                df["Train_Product_Std_Quantity"]
                .fillna(0)
                .values
                + 1e-6
            )
        )
    )

    df["V52_Product_Price_Deviation"] = (
        safe_divide(
            (
                p
                - df["Train_Product_Median_Price"]
            ).values,
            (
                df["Train_Product_Std_Price"]
                .fillna(0)
                .values
                + 1e-6
            )
        )
    )

    df["V52_Product_Price_Ratio"] = (
        safe_divide(
            p.values,
            (
                df["Train_Product_Median_Price"]
                .fillna(0)
                .values
                + 1e-6
            )
        )
    )

else:

    df["V52_Product_IQR_Quantity"] = 0
    df["V52_Product_Quantity_Deviation"] = 0
    df["V52_Product_Price_Deviation"] = 0
    df["V52_Product_Price_Ratio"] = 0


# ============================================================
# CLEAN CONTEXTUAL FEATURES
# ============================================================

context_columns = [
    "Train_Product_Frequency",
    "Train_Product_Median_Price",
    "Train_Product_Mean_Price",
    "Train_Product_Std_Price",
    "Train_Product_Median_Quantity",
    "Train_Product_Mean_Quantity",
    "Train_Product_Std_Quantity",
    "Train_Customer_Frequency",
    "Train_Description_Frequency",
    "Train_Description_Product_Count",
    "Train_Invoice_Item_Count",
    "Train_Duplicate_Group_Size",
    "Train_Product_Q1_Quantity",
    "Train_Product_Q3_Quantity",
    "V52_Product_IQR_Quantity",
    "V52_Product_Price_Deviation",
    "V52_Product_Price_Ratio",
    "V52_Product_Quantity_Deviation"
]

for col in context_columns:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        df[col] = df[col].replace(
            [np.inf, -np.inf],
            np.nan
        )

        df[col] = df[col].fillna(0)


# ============================================================
# BUILD CLEAN MODEL FEATURE SET
# ============================================================

print(
    "\nBuilding CLEAN V5.2 model feature set..."
)

# ------------------------------------------------------------
# Explicitly exclude:
#
# - Corrupted_Row_ID
# - Original_Row_ID
# - Original_Row_Occurrence
# - any *_feature identifier
# - ground-truth labels
# - synthetic corruption artifact
# - raw categorical/string columns
# ------------------------------------------------------------

EXCLUDED_EXACT = {
    "Corrupted_Row_ID",
    "Original_Row_ID",
    "Original_Row_Occurrence",
    "Original_Row_ID_feature",
    "Original_Row_Occurrence_feature",

    "Missing",
    "Invalid",
    "Outlier",
    "Inconsistency",
    "Duplicate",

    "Missing_Value",
    "Invalid_Value",
    "Outlier",
    "Inconsistency",
    "Duplicate",

    "V51_Incorrect_Description_Flag"
}

EXCLUDED_PREFIXES = [
    "Original_Row_ID",
    "Original_Row_Occurrence"
]

# Observable numeric columns we WANT.

candidate_features = []

for col in df.columns:

    if col in EXCLUDED_EXACT:
        continue

    if any(
        col.startswith(prefix)
        for prefix in EXCLUDED_PREFIXES
    ):
        continue

    # Never allow target labels.

    if col in GT_COLUMNS:
        continue

    # Never allow raw identifier-like columns.

    lower = col.lower()

    if lower in [
        "corrupted_row_id",
        "original_row_id",
        "original_row_occurrence"
    ]:
        continue

    # Keep numeric columns only.

    if pd.api.types.is_numeric_dtype(
        df[col]
    ):

        candidate_features.append(col)


# Remove any accidental duplicate names.

model_features = list(
    dict.fromkeys(candidate_features)
)

print(
    f"Clean model features: "
    f"{len(model_features)}"
)

for i, feature in enumerate(
    model_features,
    start=1
):

    print(
        f"{i:3d}. {feature}"
    )


# ============================================================
# FINAL SAFETY CHECK FOR LEAKAGE
# ============================================================

for feature in model_features:

    if feature in EXCLUDED_EXACT:

        raise ValueError(
            f"Forbidden feature detected: {feature}"
        )

    if "Row_ID" in feature:

        raise ValueError(
            f"Identifier feature detected: {feature}"
        )

    if "Occurrence" in feature:

        raise ValueError(
            f"Occurrence/provenance feature detected: "
            f"{feature}"
        )

    if feature == "V51_Incorrect_Description_Flag":

        raise ValueError(
            "Synthetic corruption artifact detected."
        )


# ============================================================
# PREPARE X
# ============================================================

X = df[model_features].copy()

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)

X = X.astype(float)


# ============================================================
# RESTORE SPLIT INDICES AFTER CONTEXT MERGES
# ============================================================

n = len(df)

train_mask = (
    np.arange(n) < train_end
)

val_mask = (
    (np.arange(n) >= train_end)
    &
    (np.arange(n) < val_end)
)

test_mask = (
    np.arange(n) >= val_end
)

train_indices = np.where(
    train_mask
)[0]

val_indices = np.where(
    val_mask
)[0]

test_indices = np.where(
    test_mask
)[0]


# ============================================================
# TRAIN TYPE-SPECIFIC RANDOM FORESTS
# ============================================================

results = []

performance_rows = []

threshold_rows = []

feature_importance_rows = []

training_rows = []


print("\n")
print("=" * 70)
print("TRAINING TYPE-SPECIFIC RANDOM FORESTS")
print("=" * 70)


for error_type, gt_col in zip(
    ERROR_TYPES,
    GT_COLUMNS
):

    print(
        f"\n{'-' * 70}"
    )

    print(
        f"ERROR TYPE: {error_type}"
    )

    print(
        f"{'-' * 70}"
    )

    y = (
        pd.to_numeric(
            df[gt_col],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
        .values
    )

    y_train = y[train_indices]
    y_val = y[val_indices]
    y_test = y[test_indices]

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

        n_estimators=RF_TREES,

        min_samples_split=MIN_SAMPLES_SPLIT,

        min_samples_leaf=MIN_SAMPLES_LEAF,

        max_features=MAX_FEATURES,

        class_weight="balanced",

        random_state=RANDOM_STATE,

        n_jobs=-1
    )


    print(
        "Training Random Forest..."
    )

    model.fit(
        X.iloc[train_indices],
        y_train
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_scores = model.predict_proba(
        X.iloc[val_indices]
    )[:, 1]

    threshold, val_f1 = (
        optimize_threshold(
            y_val,
            val_scores
        )
    )

    val_pred = (
        val_scores >= threshold
    ).astype(int)


    print(
        f"Validation threshold : "
        f"{threshold:.2f}"
    )

    print(
        f"Validation F1        : "
        f"{val_f1 * 100:.2f}%"
    )


    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    test_scores = model.predict_proba(
        X.iloc[test_indices]
    )[:, 1]

    test_pred = (
        test_scores >= threshold
    ).astype(int)


    metrics = calculate_metrics(
        y_test,
        test_pred
    )

    metrics["Error_Type"] = error_type
    metrics["Threshold"] = threshold
    metrics["Validation_F1"] = val_f1

    performance_rows.append(
        metrics.copy()
    )


    # --------------------------------------------------------
    # SAVE ROW-LEVEL RESULTS
    # --------------------------------------------------------

    test_result = pd.DataFrame({

        "Corrupted_Row_ID":
            df.iloc[test_indices]
            ["Corrupted_Row_ID"]
            .values,

        "Original_Row_ID":
            df.iloc[test_indices]
            ["Original_Row_ID"]
            .values,

        "Error_Type":
            error_type,

        "Ground_Truth":
            y_test,

        "Prediction_Score":
            test_scores,

        "Prediction":
            test_pred,

        "Threshold":
            threshold
    })

    results.append(
        test_result
    )


    # --------------------------------------------------------
    # THRESHOLD RECORD
    # --------------------------------------------------------

    threshold_rows.append({

        "Error_Type":
            error_type,

        "Best_Threshold":
            threshold,

        "Validation_F1":
            val_f1,

        "Validation_Precision":
            precision_score(
                y_val,
                val_pred,
                zero_division=0
            ),

        "Validation_Recall":
            recall_score(
                y_val,
                val_pred,
                zero_division=0
            )
    })


    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    for feature, importance in zip(
        model_features,
        model.feature_importances_
    ):

        feature_importance_rows.append({

            "Error_Type":
                error_type,

            "Feature":
                feature,

            "Importance":
                importance
        })


    # --------------------------------------------------------
    # TRAINING SUMMARY
    # --------------------------------------------------------

    training_rows.append({

        "Error_Type":
            error_type,

        "Training_Rows":
            len(train_indices),

        "Validation_Rows":
            len(val_indices),

        "Test_Rows":
            len(test_indices),

        "Training_Positive":
            int(y_train.sum()),

        "Validation_Positive":
            int(y_val.sum()),

        "Test_Positive":
            int(y_test.sum()),

        "RF_Trees":
            RF_TREES,

        "Min_Samples_Split":
            MIN_SAMPLES_SPLIT,

        "Min_Samples_Leaf":
            MIN_SAMPLES_LEAF,

        "Max_Features":
            MAX_FEATURES,

        "Class_Weight":
            "balanced",

        "Context_Statistics":
            "TRAIN_ONLY",

        "Threshold_Optimization":
            "VALIDATION_ONLY",

        "Test_Usage":
            "FINAL_EVALUATION_ONLY"
    })


    print(
        f"Test Accuracy  : "
        f"{metrics['Accuracy'] * 100:.2f}%"
    )

    print(
        f"Test Precision : "
        f"{metrics['Precision'] * 100:.2f}%"
    )

    print(
        f"Test Recall    : "
        f"{metrics['Recall'] * 100:.2f}%"
    )

    print(
        f"Test F1        : "
        f"{metrics['F1_Score'] * 100:.2f}%"
    )


# ============================================================
# COMBINE TYPE RESULTS
# ============================================================

results_df = pd.concat(
    results,
    ignore_index=True
)

performance_df = pd.DataFrame(
    performance_rows
)

threshold_df = pd.DataFrame(
    threshold_rows
)

feature_importance_df = pd.DataFrame(
    feature_importance_rows
)

training_df = pd.DataFrame(
    training_rows
)


# ============================================================
# OVERALL ROW-LEVEL PREDICTION
# ============================================================

test_ids = df.iloc[
    test_indices
][
    [
        "Corrupted_Row_ID",
        "Original_Row_ID"
    ]
].copy()

overall = test_ids.copy()

overall["GT_Missing"] = (
    df.iloc[test_indices]
    ["Missing_Value"]
    .astype(int)
    .values
)

overall["GT_Invalid"] = (
    df.iloc[test_indices]
    ["Invalid_Value"]
    .astype(int)
    .values
)

overall["GT_Outlier"] = (
    df.iloc[test_indices]
    ["Outlier"]
    .astype(int)
    .values
)

overall["GT_Inconsistency"] = (
    df.iloc[test_indices]
    ["Inconsistency"]
    .astype(int)
    .values
)

overall["GT_Duplicate"] = (
    df.iloc[test_indices]
    ["Duplicate"]
    .astype(int)
    .values
)


for error_type in ERROR_TYPES:

    type_result = results_df[
        results_df["Error_Type"]
        == error_type
    ][
        [
            "Corrupted_Row_ID",
            "Prediction_Score",
            "Prediction"
        ]
    ].copy()

    type_result = type_result.rename(
        columns={
            "Prediction_Score":
                f"{error_type}_Score",

            "Prediction":
                f"{error_type}_Prediction"
        }
    )

    overall = overall.merge(
        type_result,
        on="Corrupted_Row_ID",
        how="left"
    )


# ------------------------------------------------------------
# Overall anomaly = any type predicted
# ------------------------------------------------------------

prediction_columns = [
    f"{error_type}_Prediction"
    for error_type in ERROR_TYPES
]

overall["V52_Anomaly"] = (
    overall[prediction_columns]
    .max(axis=1)
    .astype(int)
)


ground_truth_columns = [
    "GT_Missing",
    "GT_Invalid",
    "GT_Outlier",
    "GT_Inconsistency",
    "GT_Duplicate"
]

overall["Ground_Truth_Anomaly"] = (
    overall[ground_truth_columns]
    .max(axis=1)
    .astype(int)
)


# ============================================================
# OVERALL METRICS
# ============================================================

overall_metrics = calculate_metrics(

    overall["Ground_Truth_Anomaly"],

    overall["V52_Anomaly"]
)

overall_metrics["Model"] = (
    "Type-Aware Hybrid V5.2"
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "type_aware_hybrid_v5_2_results.csv"
)

performance_path = os.path.join(
    OUTPUT_DIR,
    "v5_2_error_type_performance.csv"
)

threshold_path = os.path.join(
    OUTPUT_DIR,
    "v5_2_threshold_optimization.csv"
)

importance_path = os.path.join(
    OUTPUT_DIR,
    "v5_2_random_forest_feature_importance.csv"
)

training_path = os.path.join(
    OUTPUT_DIR,
    "v5_2_training_summary.csv"
)

overall_path = os.path.join(
    OUTPUT_DIR,
    "v5_2_overall_performance.csv"
)

feature_list_path = os.path.join(
    OUTPUT_DIR,
    "v5_2_model_features.csv"
)


results_df.to_csv(
    results_path,
    index=False
)

performance_df.to_csv(
    performance_path,
    index=False
)

threshold_df.to_csv(
    threshold_path,
    index=False
)

feature_importance_df.to_csv(
    importance_path,
    index=False
)

training_df.to_csv(
    training_path,
    index=False
)

pd.DataFrame([
    overall_metrics
]).to_csv(
    overall_path,
    index=False
)

pd.DataFrame({
    "Feature": model_features
}).to_csv(
    feature_list_path,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 70)
print("V5.2 FINAL TEST RESULTS")
print("=" * 70)

print(
    f"Accuracy  : "
    f"{overall_metrics['Accuracy'] * 100:.2f}%"
)

print(
    f"Precision : "
    f"{overall_metrics['Precision'] * 100:.2f}%"
)

print(
    f"Recall    : "
    f"{overall_metrics['Recall'] * 100:.2f}%"
)

print(
    f"F1 Score  : "
    f"{overall_metrics['F1_Score'] * 100:.2f}%"
)

print(
    f"TP        : "
    f"{overall_metrics['TP']}"
)

print(
    f"FP        : "
    f"{overall_metrics['FP']}"
)

print(
    f"TN        : "
    f"{overall_metrics['TN']}"
)

print(
    f"FN        : "
    f"{overall_metrics['FN']}"
)


print("\n")
print("=" * 70)
print("PER-ERROR-TYPE PERFORMANCE")
print("=" * 70)

display_columns = [
    "Error_Type",
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "Threshold",
    "TP",
    "FP",
    "TN",
    "FN"
]

print(
    performance_df[
        display_columns
    ].to_string(
        index=False,
        formatters={
            "Accuracy":
                lambda x: f"{x * 100:.2f}%",
            "Precision":
                lambda x: f"{x * 100:.2f}%",
            "Recall":
                lambda x: f"{x * 100:.2f}%",
            "F1_Score":
                lambda x: f"{x * 100:.2f}%",
            "Threshold":
                lambda x: f"{x:.2f}"
        }
    )
)


print("\n")
print("=" * 70)
print("METHODOLOGY AUDIT")
print("=" * 70)

print(
    "Context statistics       : TRAIN ONLY"
)

print(
    "Threshold optimization   : VALIDATION ONLY"
)

print(
    "Test set usage           : FINAL EVALUATION ONLY"
)

print(
    "Identifier features      : REMOVED"
)

print(
    "Synthetic artifact flag  : REMOVED"
)

print(
    f"Clean model features     : "
    f"{len(model_features)}"
)


print("\n")
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(results_path)
print(performance_path)
print(threshold_path)
print(importance_path)
print(training_path)
print(overall_path)
print(feature_list_path)

print("\nV5.2 completed successfully.")