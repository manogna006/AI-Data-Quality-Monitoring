# ============================================================
# TYPE-AWARE HYBRID V5.3
# Stratified + Leakage-Safe Contextual Random Forest Ensemble
#
# V5.3:
#   1. Joint error-pattern stratified 60/20/20 split
#   2. Prevents duplicate corruption concentration
#   3. Contextual statistics TRAIN ONLY
#   4. Threshold optimization VALIDATION ONLY
#   5. TEST used only for final evaluation
#   6. Removes identifier/provenance features
#   7. Removes Customer ID as a numeric feature
#   8. Removes synthetic corruption artifact detector
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

OUTPUT_DIR = "results/metrics/v5_3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

TRAIN_RATIO = 0.60
VAL_RATIO = 0.20

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

def safe_divide(a, b):
    return np.where(
        np.abs(b) > 1e-12,
        a / b,
        0.0
    )


def optimize_threshold(y_true, scores):

    thresholds = np.linspace(
        0.01,
        0.99,
        99
    )

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in thresholds:

        pred = (
            scores >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            pred,
            zero_division=0
        )

        if score > best_f1:

            best_f1 = score
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

        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),

        "Predicted_Anomalies": int(
            np.sum(y_pred)
        ),

        "Actual_Anomalies": int(
            np.sum(y_true)
        )
    }


# ============================================================
# ROBUST JOINT-STRATIFIED SPLIT
# ============================================================

def create_stratified_split(
    data,
    label_columns,
    random_state=42
):
    """
    Creates a deterministic 60/20/20 split using the complete
    combination of error labels as the stratification pattern.

    Example patterns:

        0_0_0_0_0
        1_0_0_0_0
        0_1_0_0_0
        0_0_1_0_0
        0_0_0_1_0
        0_0_0_0_1

    Every pattern is distributed across train, validation,
    and test instead of relying on row order.
    """

    rng = np.random.RandomState(
        random_state
    )

    working = data[
        label_columns
    ].copy()

    working = working.astype(int)

    # Build joint error pattern.

    pattern = (
        working.astype(str)
        .agg("_".join, axis=1)
    )

    train_indices = []
    val_indices = []
    test_indices = []

    # IMPORTANT:
    # Convert indices to a fresh writable NumPy array.
    # This fixes the read-only array error encountered earlier.

    grouped_indices = {}

    for pattern_name, group in pattern.groupby(
        pattern,
        sort=True
    ):

        idx = np.array(
            group.index.tolist(),
            dtype=np.int64
        )

        # Make absolutely sure it is writable.

        idx = idx.copy()

        rng.shuffle(idx)

        grouped_indices[
            pattern_name
        ] = idx


    # Allocate each pattern.

    for pattern_name, indices in (
        grouped_indices.items()
    ):

        n_group = len(indices)

        n_train = int(
            np.floor(
                n_group * TRAIN_RATIO
            )
        )

        n_val = int(
            np.floor(
                n_group * VAL_RATIO
            )
        )

        # Ensure very small groups still get represented
        # sensibly where possible.

        if n_group >= 5:

            n_train = max(
                n_train,
                1
            )

            n_val = max(
                n_val,
                1
            )

        group_train = indices[
            :n_train
        ]

        group_val = indices[
            n_train:
            n_train + n_val
        ]

        group_test = indices[
            n_train + n_val:
        ]

        train_indices.extend(
            group_train.tolist()
        )

        val_indices.extend(
            group_val.tolist()
        )

        test_indices.extend(
            group_test.tolist()
        )


    # Convert to fresh writable arrays.

    train_indices = np.array(
        train_indices,
        dtype=np.int64
    ).copy()

    val_indices = np.array(
        val_indices,
        dtype=np.int64
    ).copy()

    test_indices = np.array(
        test_indices,
        dtype=np.int64
    ).copy()


    # Final independent shuffle.

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)


    return (
        train_indices,
        val_indices,
        test_indices
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("TYPE-AWARE HYBRID V5.3")
print("STRATIFIED + LEAKAGE-SAFE RANDOM FOREST")
print("=" * 70)

print("\nLoading datasets...")

v2 = pd.read_csv(
    V2_RESULTS
)

features = pd.read_csv(
    V2_FEATURES
)

ground_truth = pd.read_csv(
    GROUND_TRUTH
)

raw = pd.read_csv(
    RAW_DATA
)

print(
    f"V2 results       : {len(v2)}"
)

print(
    f"V2 features      : {len(features)}"
)

print(
    f"Ground truth     : {len(ground_truth)}"
)

print(
    f"Raw data         : {len(raw)}"
)


# ============================================================
# VALIDATE REQUIRED KEYS
# ============================================================

if "Corrupted_Row_ID" not in v2.columns:
    raise ValueError(
        "V2 results missing Corrupted_Row_ID."
    )

if "Original_Row_ID" not in v2.columns:
    raise ValueError(
        "V2 results missing Original_Row_ID."
    )

if "Corrupted_Row_ID" not in features.columns:
    raise ValueError(
        "V2 features missing Corrupted_Row_ID."
    )

if "Original_Row_ID" not in ground_truth.columns:
    raise ValueError(
        "Ground truth missing Original_Row_ID."
    )


# ============================================================
# PREPARE RAW KEY
# ============================================================

if "Corrupted_Row_ID" not in raw.columns:

    print(
        "\nRaw CSV does not contain "
        "Corrupted_Row_ID."
    )

    print(
        "Creating from raw row order..."
    )

    raw.insert(
        0,
        "Corrupted_Row_ID",
        np.arange(
            len(raw),
            dtype=np.int64
        )
    )


v2["Corrupted_Row_ID"] = pd.to_numeric(
    v2["Corrupted_Row_ID"],
    errors="coerce"
)

features["Corrupted_Row_ID"] = pd.to_numeric(
    features["Corrupted_Row_ID"],
    errors="coerce"
)

raw["Corrupted_Row_ID"] = pd.to_numeric(
    raw["Corrupted_Row_ID"],
    errors="coerce"
)


# ============================================================
# RAW / V2 ALIGNMENT
# ============================================================

if len(raw) != len(v2):

    raise ValueError(
        "Raw and V2 row counts differ."
    )

if not np.array_equal(
    raw["Corrupted_Row_ID"].values,
    v2["Corrupted_Row_ID"].values
):

    raise ValueError(
        "Raw/V2 Corrupted_Row_ID alignment failed."
    )

print(
    "Raw/V2 row-key alignment verified."
)


# ============================================================
# REMOVE EXISTING LABELS FROM V2
# ============================================================

existing_labels = [
    col
    for col in ERROR_TYPES
    if col in v2.columns
]

if existing_labels:

    print(
        "\nRemoving existing V2 labels:"
    )

    print(
        existing_labels
    )

    v2 = v2.drop(
        columns=existing_labels
    )


# ============================================================
# DEDUPLICATE FEATURE TABLE
# ============================================================

features = features.drop_duplicates(
    subset=["Corrupted_Row_ID"],
    keep="first"
)

print(
    f"Unique feature rows: {len(features)}"
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
    f"After V2 + feature merge: "
    f"{len(df)}"
)

if len(df) != len(v2):

    raise ValueError(
        "Feature merge changed row count."
    )


# ============================================================
# PREPARE GROUND TRUTH
# ============================================================

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
            f"Ground truth missing: {col}"
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
    f"After ground-truth merge: "
    f"{len(df)}"
)

if len(df) != len(v2):

    raise ValueError(
        "Ground-truth merge changed row count."
    )


# ============================================================
# RAW CONTEXT
# ============================================================

raw_columns_to_add = []

for col in raw.columns:

    if col == "Corrupted_Row_ID":
        continue

    if col not in df.columns:

        raw_columns_to_add.append(col)


if raw_columns_to_add:

    raw_subset = raw[
        ["Corrupted_Row_ID"]
        + raw_columns_to_add
    ].drop_duplicates(
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
        "All raw context columns already exist."
    )


print(
    f"After raw context merge: "
    f"{len(df)}"
)

if len(df) != len(v2):

    raise ValueError(
        "Raw merge changed row count."
    )

print(
    "Merge integrity verified."
)


# ============================================================
# BASIC TYPES
# ============================================================

if "InvoiceDate" in df.columns:

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce"
    )


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
# OBSERVABLE FEATURES
# ============================================================

print(
    "\nCreating observable features..."
)


if "Quantity" in df.columns:

    q = pd.to_numeric(
        df["Quantity"],
        errors="coerce"
    )

else:

    q = pd.Series(
        0.0,
        index=df.index
    )


if "Price" in df.columns:

    p = pd.to_numeric(
        df["Price"],
        errors="coerce"
    )

else:

    p = pd.Series(
        0.0,
        index=df.index
    )


# ------------------------------------------------------------
# Quantity
# ------------------------------------------------------------

df["V53_Abs_Quantity"] = q.abs()

df["V53_Log_Quantity"] = np.log1p(
    q.abs()
)

df["V53_Negative_Quantity"] = (
    q < 0
).astype(int)

df["V53_Zero_Quantity"] = (
    q == 0
).astype(int)


# ------------------------------------------------------------
# Price
# ------------------------------------------------------------

df["V53_Abs_Price"] = p.abs()

df["V53_Log_Price"] = np.log1p(
    p.abs()
)

df["V53_Negative_Price"] = (
    p < 0
).astype(int)

df["V53_Zero_Price"] = (
    p == 0
).astype(int)


# ------------------------------------------------------------
# Amount
# ------------------------------------------------------------

df["V53_Total_Amount"] = (
    q * p
)

df["V53_Abs_Total_Amount"] = (
    df["V53_Total_Amount"].abs()
)

df["V53_Price_Quantity_Ratio"] = (
    safe_divide(
        p.values,
        q.abs().values + 1e-6
    )
)


# ============================================================
# MISSINGNESS FEATURES
# ============================================================

missing_source_columns = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country"
]

missing_features = []

for col in missing_source_columns:

    if col in df.columns:

        feature_name = (
            "V53_Missing_"
            + col.replace(" ", "_")
        )

        df[feature_name] = (
            df[col]
            .isna()
            .astype(int)
        )

        missing_features.append(
            feature_name
        )


if missing_features:

    df["V53_Total_Missing_Count"] = (
        df[missing_features]
        .sum(axis=1)
    )

else:

    df["V53_Total_Missing_Count"] = 0


# ============================================================
# DESCRIPTION FEATURES
# ============================================================

if "Description" in df.columns:

    desc = (
        df["Description"]
        .fillna("")
        .astype(str)
    )

    df["V53_Description_Length"] = (
        desc.str.len()
    )

    df["V53_Description_Word_Count"] = (
        desc.str.split().str.len()
    )

    df["V53_Description_Is_Empty"] = (
        desc.str.strip()
        .eq("")
        .astype(int)
    )

    # Generic only — no exact synthetic corruption detector.

    df["V53_Description_Has_Question"] = (
        desc.str.contains(
            r"\?",
            regex=True,
            na=False
        )
        .astype(int)
    )

    df["V53_Description_All_Caps"] = (
        desc.str.isupper()
        .astype(int)
    )

else:

    df["V53_Description_Length"] = 0
    df["V53_Description_Word_Count"] = 0
    df["V53_Description_Is_Empty"] = 1
    df["V53_Description_Has_Question"] = 0
    df["V53_Description_All_Caps"] = 0


# ============================================================
# DATE FEATURES
# ============================================================

if "InvoiceDate" in df.columns:

    dt = df["InvoiceDate"]

    df["V53_Hour"] = (
        dt.dt.hour
        .fillna(-1)
    )

    df["V53_DayOfWeek"] = (
        dt.dt.dayofweek
        .fillna(-1)
    )

    df["V53_DayOfMonth"] = (
        dt.dt.day
        .fillna(-1)
    )

    df["V53_Month"] = (
        dt.dt.month
        .fillna(-1)
    )

    df["V53_Year"] = (
        dt.dt.year
        .fillna(-1)
    )

    df["V53_IsWeekend"] = (
        dt.dt.dayofweek
        .isin([5, 6])
        .astype(int)
    )

    df["V53_Hour_Sin"] = np.sin(
        2 * np.pi * df["V53_Hour"] / 24
    )

    df["V53_Hour_Cos"] = np.cos(
        2 * np.pi * df["V53_Hour"] / 24
    )

    df["V53_Day_Sin"] = np.sin(
        2 * np.pi * df["V53_DayOfWeek"] / 7
    )

    df["V53_Day_Cos"] = np.cos(
        2 * np.pi * df["V53_DayOfWeek"] / 7
    )

    df["V53_Month_Sin"] = np.sin(
        2 * np.pi * df["V53_Month"] / 12
    )

    df["V53_Month_Cos"] = np.cos(
        2 * np.pi * df["V53_Month"] / 12
    )

else:

    for col in [
        "V53_Hour",
        "V53_DayOfWeek",
        "V53_DayOfMonth",
        "V53_Month",
        "V53_Year",
        "V53_IsWeekend",
        "V53_Hour_Sin",
        "V53_Hour_Cos",
        "V53_Day_Sin",
        "V53_Day_Cos",
        "V53_Month_Sin",
        "V53_Month_Cos"
    ]:

        df[col] = 0


# ============================================================
# STRATIFIED SPLIT
# ============================================================

print(
    "\nCreating STRATIFIED 60/20/20 split..."
)

train_idx, val_idx, test_idx = (
    create_stratified_split(
        df,
        GT_COLUMNS,
        RANDOM_STATE
    )
)


print(
    f"Train      : {len(train_idx)}"
)

print(
    f"Validation : {len(val_idx)}"
)

print(
    f"Test       : {len(test_idx)}"
)


if (
    len(train_idx)
    + len(val_idx)
    + len(test_idx)
    != len(df)
):

    raise ValueError(
        "Split does not cover all rows."
    )


# ============================================================
# ENSURE NO OVERLAP
# ============================================================

if (
    len(
        set(train_idx)
        & set(val_idx)
    )
    > 0
):

    raise ValueError(
        "Train/validation overlap detected."
    )


if (
    len(
        set(train_idx)
        & set(test_idx)
    )
    > 0
):

    raise ValueError(
        "Train/test overlap detected."
    )


if (
    len(
        set(val_idx)
        & set(test_idx)
    )
    > 0
):

    raise ValueError(
        "Validation/test overlap detected."
    )


print(
    "Split overlap check: PASSED"
)


# ============================================================
# VERIFY STRATIFICATION
# ============================================================

print(
    "\nError distribution by split:"
)

split_distribution_rows = []

for error_type, gt_col in zip(
    ERROR_TYPES,
    GT_COLUMNS
):

    train_pos = int(
        df.loc[
            train_idx,
            gt_col
        ].sum()
    )

    val_pos = int(
        df.loc[
            val_idx,
            gt_col
        ].sum()
    )

    test_pos = int(
        df.loc[
            test_idx,
            gt_col
        ].sum()
    )

    train_rate = (
        train_pos
        / len(train_idx)
    )

    val_rate = (
        val_pos
        / len(val_idx)
    )

    test_rate = (
        test_pos
        / len(test_idx)
    )

    print(
        f"{error_type:15s} "
        f"Train={train_pos:5d} "
        f"({train_rate * 100:6.3f}%)   "
        f"Val={val_pos:5d} "
        f"({val_rate * 100:6.3f}%)   "
        f"Test={test_pos:5d} "
        f"({test_rate * 100:6.3f}%)"
    )

    split_distribution_rows.append({

        "Error_Type":
            error_type,

        "Train_Count":
            train_pos,

        "Validation_Count":
            val_pos,

        "Test_Count":
            test_pos,

        "Train_Rate":
            train_rate,

        "Validation_Rate":
            val_rate,

        "Test_Rate":
            test_rate
    })


# ============================================================
# TRAIN-ONLY CONTEXTUAL STATISTICS
# ============================================================

print(
    "\nBuilding TRAIN-ONLY contextual statistics..."
)

train_df = df.loc[
    train_idx
].copy()


# ============================================================
# PRODUCT CONTEXT
# ============================================================

if "StockCode" in df.columns:

    product_train = (
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
                lambda x:
                x.quantile(0.25)
            ),

            Train_Product_Q3_Quantity=(
                "Quantity",
                lambda x:
                x.quantile(0.75)
            )
        )
        .reset_index()
    )

else:

    product_train = pd.DataFrame()


# ============================================================
# CUSTOMER CONTEXT
# ============================================================

if "Customer ID" in df.columns:

    customer_train = (
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

else:

    customer_train = pd.DataFrame()


# ============================================================
# DESCRIPTION CONTEXT
# ============================================================

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


# ============================================================
# INVOICE CONTEXT
# ============================================================

if "Invoice" in df.columns:

    invoice_train = (
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

else:

    invoice_train = pd.DataFrame()


# ============================================================
# DUPLICATE CONTEXT
# ============================================================

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
# APPLY TRAIN-ONLY CONTEXT
# ============================================================

# Preserve original row order.

df["_V53_INTERNAL_INDEX"] = np.arange(
    len(df)
)


if not product_train.empty:

    df = df.merge(
        product_train,
        on="StockCode",
        how="left"
    )


if not customer_train.empty:

    df = df.merge(
        customer_train,
        on="Customer ID",
        how="left"
    )


if not description_train.empty:

    df = df.merge(
        description_train,
        on="Description",
        how="left"
    )

    df = df.merge(
        description_product_train,
        on="Description",
        how="left"
    )


if not invoice_train.empty:

    df = df.merge(
        invoice_train,
        on="Invoice",
        how="left"
    )


if not duplicate_train.empty:

    df = df.merge(
        duplicate_train,
        on=duplicate_group_columns,
        how="left"
    )


df = df.sort_values(
    "_V53_INTERNAL_INDEX"
).reset_index(drop=True)

df.drop(
    columns=["_V53_INTERNAL_INDEX"],
    inplace=True
)


# ============================================================
# DERIVED CONTEXT
# ============================================================

if (
    "Train_Product_Q1_Quantity"
    in df.columns
):

    df["V53_Product_IQR_Quantity"] = (
        df[
            "Train_Product_Q3_Quantity"
        ]
        -
        df[
            "Train_Product_Q1_Quantity"
        ]
    )

    df["V53_Product_Quantity_Deviation"] = (
        safe_divide(
            (
                q
                -
                df[
                    "Train_Product_Median_Quantity"
                ]
            ).values,

            (
                df[
                    "Train_Product_Std_Quantity"
                ]
                .fillna(0)
                .values
                + 1e-6
            )
        )
    )

    df["V53_Product_Price_Deviation"] = (
        safe_divide(
            (
                p
                -
                df[
                    "Train_Product_Median_Price"
                ]
            ).values,

            (
                df[
                    "Train_Product_Std_Price"
                ]
                .fillna(0)
                .values
                + 1e-6
            )
        )
    )

    df["V53_Product_Price_Ratio"] = (
        safe_divide(
            p.values,

            (
                df[
                    "Train_Product_Median_Price"
                ]
                .fillna(0)
                .values
                + 1e-6
            )
        )
    )

else:

    df["V53_Product_IQR_Quantity"] = 0
    df["V53_Product_Quantity_Deviation"] = 0
    df["V53_Product_Price_Deviation"] = 0
    df["V53_Product_Price_Ratio"] = 0


# ============================================================
# CLEAN CONTEXT FEATURES
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

    "V53_Product_IQR_Quantity",
    "V53_Product_Quantity_Deviation",
    "V53_Product_Price_Deviation",
    "V53_Product_Price_Ratio"
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
# CLEAN MODEL FEATURE SET
# ============================================================

print(
    "\nBuilding CLEAN V5.3 model features..."
)

EXCLUDED_EXACT = {

    "Corrupted_Row_ID",

    "Original_Row_ID",
    "Original_Row_Occurrence",

    "Original_Row_ID_feature",
    "Original_Row_Occurrence_feature",

    "Customer ID",

    # Ground truth.

    "Missing_Value",
    "Invalid_Value",
    "Outlier",
    "Inconsistency",
    "Duplicate",

    "Missing",
    "Invalid",
    "Outlier",
    "Inconsistency",
    "Duplicate",

    # Synthetic artifact.

    "V51_Incorrect_Description_Flag",
    "V52_Incorrect_Description_Flag",
    "V53_Incorrect_Description_Flag"
}


EXCLUDED_PREFIXES = [
    "Original_Row_ID",
    "Original_Row_Occurrence"
]


model_features = []

for col in df.columns:

    if col in EXCLUDED_EXACT:
        continue

    if any(
        col.startswith(prefix)
        for prefix in EXCLUDED_PREFIXES
    ):
        continue

    if col in GT_COLUMNS:
        continue

    lower = col.lower()

    if lower in [
        "corrupted_row_id",
        "original_row_id",
        "original_row_occurrence",
        "customer id"
    ]:
        continue

    if pd.api.types.is_numeric_dtype(
        df[col]
    ):

        model_features.append(col)


model_features = list(
    dict.fromkeys(
        model_features
    )
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
# FINAL FEATURE SAFETY CHECK
# ============================================================

for feature in model_features:

    if feature in EXCLUDED_EXACT:

        raise ValueError(
            f"Forbidden feature: {feature}"
        )

    if "Row_ID" in feature:

        raise ValueError(
            f"Identifier feature: {feature}"
        )

    if "Occurrence" in feature:

        raise ValueError(
            f"Occurrence feature: {feature}"
        )

    if feature == "Customer ID":

        raise ValueError(
            "Customer ID is not allowed."
        )


# ============================================================
# X MATRIX
# ============================================================

X = df[
    model_features
].copy()

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)

X = X.astype(float)


# ============================================================
# IMPORTANT:
# Context merges can preserve row count, but verify.
# ============================================================

if len(df) != 530715:

    raise ValueError(
        "Unexpected row count after contextual merges: "
        f"{len(df)}"
    )


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


    y_train = y[
        train_idx
    ]

    y_val = y[
        val_idx
    ]

    y_test = y[
        test_idx
    ]


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


    model = RandomForestClassifier(

        n_estimators=RF_TREES,

        min_samples_split=
            MIN_SAMPLES_SPLIT,

        min_samples_leaf=
            MIN_SAMPLES_LEAF,

        max_features=
            MAX_FEATURES,

        class_weight="balanced",

        random_state=
            RANDOM_STATE,

        n_jobs=-1
    )


    print(
        "Training Random Forest..."
    )


    model.fit(
        X.iloc[train_idx],
        y_train
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_scores = model.predict_proba(
        X.iloc[val_idx]
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
    # TEST
    # --------------------------------------------------------

    test_scores = model.predict_proba(
        X.iloc[test_idx]
    )[:, 1]


    test_pred = (
        test_scores >= threshold
    ).astype(int)


    metrics = calculate_metrics(
        y_test,
        test_pred
    )


    metrics["Error_Type"] = (
        error_type
    )

    metrics["Threshold"] = (
        threshold
    )

    metrics["Validation_F1"] = (
        val_f1
    )


    performance_rows.append(
        metrics.copy()
    )


    # --------------------------------------------------------
    # ROW-LEVEL RESULTS
    # --------------------------------------------------------

    test_result = pd.DataFrame({

        "Corrupted_Row_ID":
            df.iloc[
                test_idx
            ]["Corrupted_Row_ID"].values,

        "Original_Row_ID":
            df.iloc[
                test_idx
            ]["Original_Row_ID"].values,

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
    # THRESHOLD
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
            len(train_idx),

        "Validation_Rows":
            len(val_idx),

        "Test_Rows":
            len(test_idx),

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

        "Split_Strategy":
            "JOINT_ERROR_PATTERN_STRATIFIED",

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
# COMBINE RESULTS
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
# OVERALL TEST RESULTS
# ============================================================

overall = df.iloc[
    test_idx
][
    [
        "Corrupted_Row_ID",
        "Original_Row_ID"
    ]
].copy()


overall["GT_Missing"] = (
    df.iloc[
        test_idx
    ]["Missing_Value"]
    .astype(int)
    .values
)

overall["GT_Invalid"] = (
    df.iloc[
        test_idx
    ]["Invalid_Value"]
    .astype(int)
    .values
)

overall["GT_Outlier"] = (
    df.iloc[
        test_idx
    ]["Outlier"]
    .astype(int)
    .values
)

overall["GT_Inconsistency"] = (
    df.iloc[
        test_idx
    ]["Inconsistency"]
    .astype(int)
    .values
)

overall["GT_Duplicate"] = (
    df.iloc[
        test_idx
    ]["Duplicate"]
    .astype(int)
    .values
)


# ============================================================
# MERGE PREDICTIONS
# ============================================================

for error_type in ERROR_TYPES:

    temp = results_df[
        results_df["Error_Type"]
        == error_type
    ][
        [
            "Corrupted_Row_ID",
            "Prediction_Score",
            "Prediction"
        ]
    ].copy()


    temp = temp.rename(
        columns={

            "Prediction_Score":
                f"{error_type}_Score",

            "Prediction":
                f"{error_type}_Prediction"
        }
    )


    overall = overall.merge(
        temp,
        on="Corrupted_Row_ID",
        how="left"
    )


# ============================================================
# OVERALL ANOMALY
# ============================================================

prediction_columns = [
    f"{error_type}_Prediction"
    for error_type in ERROR_TYPES
]

ground_truth_columns = [
    "GT_Missing",
    "GT_Invalid",
    "GT_Outlier",
    "GT_Inconsistency",
    "GT_Duplicate"
]


overall["V53_Anomaly"] = (
    overall[
        prediction_columns
    ]
    .max(axis=1)
    .astype(int)
)


overall["Ground_Truth_Anomaly"] = (
    overall[
        ground_truth_columns
    ]
    .max(axis=1)
    .astype(int)
)


# ============================================================
# OVERALL METRICS
# ============================================================

overall_metrics = calculate_metrics(

    overall[
        "Ground_Truth_Anomaly"
    ],

    overall[
        "V53_Anomaly"
    ]
)

overall_metrics["Model"] = (
    "Type-Aware Hybrid V5.3"
)

overall_metrics["Split_Strategy"] = (
    "Joint Error Pattern Stratified"
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "type_aware_hybrid_v5_3_results.csv"
)

performance_path = os.path.join(
    OUTPUT_DIR,
    "v5_3_error_type_performance.csv"
)

threshold_path = os.path.join(
    OUTPUT_DIR,
    "v5_3_threshold_optimization.csv"
)

importance_path = os.path.join(
    OUTPUT_DIR,
    "v5_3_random_forest_feature_importance.csv"
)

training_path = os.path.join(
    OUTPUT_DIR,
    "v5_3_training_summary.csv"
)

overall_path = os.path.join(
    OUTPUT_DIR,
    "v5_3_overall_performance.csv"
)

feature_path = os.path.join(
    OUTPUT_DIR,
    "v5_3_model_features.csv"
)

split_path = os.path.join(
    OUTPUT_DIR,
    "v5_3_split_distribution.csv"


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
    feature_path,
    index=False
)

pd.DataFrame(
    split_distribution_rows
).to_csv(
    split_path,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 70)
print("V5.3 FINAL TEST RESULTS")
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
                lambda x:
                f"{x * 100:.2f}%",

            "Precision":
                lambda x:
                f"{x * 100:.2f}%",

            "Recall":
                lambda x:
                f"{x * 100:.2f}%",

            "F1_Score":
                lambda x:
                f"{x * 100:.2f}%",

            "Threshold":
                lambda x:
                f"{x:.2f}"
        }
    )
)


print("\n")
print("=" * 70)
print("METHODOLOGY AUDIT")
print("=" * 70)

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
print(feature_path)
print(split_path)

print("\nV5.3 completed successfully.")