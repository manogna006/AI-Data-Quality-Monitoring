# ============================================================
# TYPE-AWARE HYBRID V5.1
# LEAKAGE-SAFE CONTEXT-AWARE RANDOM FOREST
#
# V5.1 improvements:
#   1. Contextual statistics are learned from TRAIN ONLY
#   2. Validation/Test receive TRAIN-derived statistics
#   3. Thresholds are optimized on VALIDATION ONLY
#   4. TEST is used only for final evaluation
#   5. Robust handling when raw CSV does not contain
#      Corrupted_Row_ID
#
# V5 remains untouched as the benchmark.
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

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

V2_RESULTS = os.path.join(
    BASE_DIR,
    "results",
    "metrics",
    "v2",
    "model_scores_v2.csv"
)

V2_FEATURES = os.path.join(
    BASE_DIR,
    "results",
    "metrics",
    "v2",
    "ml_features_v2.csv"
)

GROUND_TRUTH = os.path.join(
    BASE_DIR,
    "results",
    "metrics",
    "ground_truth.csv"
)

RAW_DATA = os.path.join(
    BASE_DIR,
    "datasets",
    "corrupted",
    "online_retail_corrupted.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "metrics",
    "v5_1"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

RANDOM_STATE = 42

TRAIN_RATIO = 0.60
VAL_RATIO = 0.20

N_ESTIMATORS = 350
MIN_SAMPLES_SPLIT = 4
MIN_SAMPLES_LEAF = 2
MAX_FEATURES = "sqrt"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("TYPE-AWARE HYBRID V5.1")
print("LEAKAGE-SAFE CONTEXT-AWARE RANDOM FOREST")
print("=" * 70)

print("\nLoading V2 results...")
v2 = pd.read_csv(V2_RESULTS)

print("V2 rows:", len(v2))

print("\nLoading V2 features...")
features = pd.read_csv(V2_FEATURES)

print("V2 feature rows:", len(features))

print("\nLoading ground truth...")
gt = pd.read_csv(GROUND_TRUTH)

print("Ground truth rows:", len(gt))

print("\nLoading raw corrupted dataset...")
raw = pd.read_csv(RAW_DATA)

print("Raw rows:", len(raw))


# ============================================================
# BASIC VALIDATION
# ============================================================

if len(v2) != len(features):
    raise ValueError(
        f"V2 results ({len(v2)}) and V2 features "
        f"({len(features)}) have different row counts."
    )

if len(raw) != len(v2):
    raise ValueError(
        f"Raw dataset ({len(raw)}) and V2 results "
        f"({len(v2)}) have different row counts. "
        "Cannot safely align raw context."
    )


# ============================================================
# IDENTIFY MERGE KEY
# ============================================================

if "Corrupted_Row_ID" in v2.columns:

    MERGE_KEY = "Corrupted_Row_ID"

else:

    raise ValueError(
        "Corrupted_Row_ID not found in V2 results."
    )

print("\nMerge key:", MERGE_KEY)


# ============================================================
# FIX RAW DATA KEY
#
# The raw corrupted CSV may not contain Corrupted_Row_ID.
#
# Since the raw file and V2 files have identical row counts,
# and the corruption pipeline preserves row order, create
# Corrupted_Row_ID from raw row order if necessary.
# ============================================================

if MERGE_KEY not in raw.columns:

    print(
        "\nWARNING: Corrupted_Row_ID not found in raw CSV."
    )

    print(
        "Creating Corrupted_Row_ID from raw row order..."
    )

    raw.insert(
        0,
        MERGE_KEY,
        np.arange(
            len(raw),
            dtype=np.int64
        )
    )

else:

    print(
        "\nRaw dataset already contains Corrupted_Row_ID."
    )


# ============================================================
# VERIFY KEY ALIGNMENT
# ============================================================

v2_key = pd.to_numeric(
    v2[MERGE_KEY],
    errors="coerce"
)

raw_key = pd.to_numeric(
    raw[MERGE_KEY],
    errors="coerce"
)

if v2_key.isna().any():
    raise ValueError(
        "V2 contains invalid/null Corrupted_Row_ID values."
    )

if raw_key.isna().any():
    raise ValueError(
        "Raw dataset contains invalid/null Corrupted_Row_ID values."
    )

if not np.array_equal(
    v2_key.to_numpy(),
    raw_key.to_numpy()
):

    print(
        "\nV2 and raw key ordering differs."
    )

    print(
        "Attempting safe key-based alignment..."
    )

    raw = (
        raw
        .set_index(MERGE_KEY)
        .reindex(v2[MERGE_KEY])
        .reset_index()
    )

    if not np.array_equal(
        pd.to_numeric(
            raw[MERGE_KEY],
            errors="coerce"
        ).to_numpy(),
        v2_key.to_numpy()
    ):

        raise ValueError(
            "Could not safely align raw data with V2 results."
        )

else:

    print(
        "Raw/V2 row-key alignment verified."
    )


# ============================================================
# REMOVE CONFLICTING V2 LABEL COLUMNS
# ============================================================

error_columns = [
    "Missing",
    "Invalid",
    "Outlier",
    "Inconsistency",
    "Duplicate"
]

for col in error_columns:

    if col in v2.columns:

        print(
            "Removing existing V2 label column:",
            col
        )

        v2 = v2.drop(
            columns=[col]
        )


# ============================================================
# CLEAN FEATURE TABLE
# ============================================================

features = features.drop_duplicates(
    subset=[MERGE_KEY],
    keep="first"
)

print(
    "\nUnique feature rows:",
    len(features)
)


# ============================================================
# MERGE V2 + FEATURES
# ============================================================

df = v2.merge(
    features,
    on=MERGE_KEY,
    how="left",
    suffixes=(
        "",
        "_feature"
    )
)

print(
    "After V2 + feature merge:",
    len(df)
)


# ============================================================
# GROUND TRUTH MERGE
# ============================================================

required_gt_columns = [
    "Original_Row_ID",
    "Missing_Value",
    "Invalid_Value",
    "Outlier",
    "Inconsistency",
    "Duplicate"
]

missing_gt_columns = [
    c
    for c in required_gt_columns
    if c not in gt.columns
]

if missing_gt_columns:

    raise ValueError(
        "Ground truth is missing columns: "
        + str(missing_gt_columns)
    )

gt_clean = (
    gt[required_gt_columns]
    .drop_duplicates(
        subset=["Original_Row_ID"],
        keep="first"
    )
)

if "Original_Row_ID" not in df.columns:

    raise ValueError(
        "Original_Row_ID missing from V2 results."
    )

df = df.merge(
    gt_clean,
    on="Original_Row_ID",
    how="left"
)

print(
    "After ground-truth merge:",
    len(df)
)


# ============================================================
# RAW CONTEXT MERGE
# ============================================================

raw_context_columns = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country"
]

raw_context_columns = [
    c
    for c in raw_context_columns
    if c in raw.columns
]

# Only merge columns that are not already in df.
new_raw_columns = [
    c
    for c in raw_context_columns
    if c not in df.columns
]

if new_raw_columns:

    print(
        "\nAdding raw context columns:",
        new_raw_columns
    )

    raw_context = (
        raw[
            [MERGE_KEY] + new_raw_columns
        ]
        .drop_duplicates(
            subset=[MERGE_KEY],
            keep="first"
        )
    )

    df = df.merge(
        raw_context,
        on=MERGE_KEY,
        how="left"
    )

else:

    print(
        "\nAll raw context columns already exist."
    )

print(
    "After raw context merge:",
    len(df)
)


# ============================================================
# FINAL MERGE VALIDATION
# ============================================================

if len(df) != len(v2):

    raise ValueError(
        "Row count changed after merging raw context. "
        f"Expected {len(v2)}, got {len(df)}."
    )

print(
    "Merge integrity verified:",
    len(df),
    "rows"
)


# ============================================================
# NORMALIZE BASIC DATA TYPES
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


if "InvoiceDate" in df.columns:

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce"
    )


for col in [
    "StockCode",
    "Description",
    "Invoice",
    "Country"
]:

    if col in df.columns:

        df[col] = (
            df[col]
            .fillna("")
            .astype(str)
        )


# ============================================================
# CREATE TARGET LABELS
# ============================================================

label_map = {
    "Missing_Value": "Missing",
    "Invalid_Value": "Invalid",
    "Outlier": "Outlier",
    "Inconsistency": "Inconsistency",
    "Duplicate": "Duplicate"
}

for source, target in label_map.items():

    df[target] = (
        pd.to_numeric(
            df[source],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )


# ============================================================
# DETERMINISTIC SPLIT
# ============================================================

def get_split_ids(data):

    ids = (
        pd.to_numeric(
            data["Corrupted_Row_ID"],
            errors="coerce"
        )
        .astype(np.int64)
    )

    unique_ids = np.sort(
        ids.unique()
    )

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    shuffled = unique_ids.copy()

    rng.shuffle(
        shuffled
    )

    n = len(shuffled)

    train_end = int(
        n * TRAIN_RATIO
    )

    val_end = int(
        n * (TRAIN_RATIO + VAL_RATIO)
    )

    train_ids = set(
        shuffled[:train_end]
    )

    val_ids = set(
        shuffled[
            train_end:val_end
        ]
    )

    test_ids = set(
        shuffled[val_end:]
    )

    return (
        ids.isin(train_ids),
        ids.isin(val_ids),
        ids.isin(test_ids)
    )


print(
    "\nCreating deterministic "
    "train/validation/test split..."
)

train_mask, val_mask, test_mask = (
    get_split_ids(df)
)

train_df = df.loc[
    train_mask
].copy()

val_df = df.loc[
    val_mask
].copy()

test_df = df.loc[
    test_mask
].copy()

print(
    "Train rows:",
    len(train_df)
)

print(
    "Validation rows:",
    len(val_df)
)

print(
    "Test rows:",
    len(test_df)
)


# ============================================================
# BASIC OBSERVABLE FEATURES
# ============================================================

def add_basic_features(data):

    data = data.copy()

    if "Quantity" in data.columns:

        data["V51_Abs_Quantity"] = (
            data["Quantity"].abs()
        )

        data["V51_Log_Quantity"] = np.log1p(
            data["Quantity"].abs()
        )

        data["V51_Negative_Quantity"] = (
            data["Quantity"] < 0
        ).astype(int)

        data["V51_Zero_Quantity"] = (
            data["Quantity"] == 0
        ).astype(int)

    if "Price" in data.columns:

        data["V51_Abs_Price"] = (
            data["Price"].abs()
        )

        data["V51_Log_Price"] = np.log1p(
            data["Price"].abs()
        )

        data["V51_Negative_Price"] = (
            data["Price"] < 0
        ).astype(int)

        data["V51_Zero_Price"] = (
            data["Price"] == 0
        ).astype(int)

    if (
        "Quantity" in data.columns
        and "Price" in data.columns
    ):

        data["V51_Total_Amount"] = (
            data["Quantity"]
            * data["Price"]
        )

        data["V51_Abs_Total_Amount"] = (
            data["V51_Total_Amount"].abs()
        )

        data["V51_Price_Quantity_Ratio"] = (
            data["Price"]
            /
            (
                data["Quantity"].abs()
                + 1e-6
            )
        )

    if "Description" in data.columns:

        data["V51_Description_Length"] = (
            data["Description"]
            .astype(str)
            .str.len()
        )

        data["V51_Description_Word_Count"] = (
            data["Description"]
            .astype(str)
            .str.split()
            .str.len()
        )

        data["V51_Description_Missing"] = (
            data["Description"]
            .astype(str)
            .str.strip()
            .eq("")
            .astype(int)
        )

        data[
            "V51_Incorrect_Description_Flag"
        ] = (
            data["Description"]
            .astype(str)
            .eq(
                "INCORRECT_PRODUCT_DESCRIPTION"
            )
            .astype(int)
        )

    if "Customer ID" in data.columns:

        data["V51_Customer_Missing"] = (
            data["Customer ID"]
            .isna()
            .astype(int)
        )

    if "Invoice" in data.columns:

        data["V51_Invoice_Missing"] = (
            data["Invoice"]
            .astype(str)
            .str.strip()
            .eq("")
            .astype(int)
        )

    if "Country" in data.columns:

        data["V51_Country_Missing"] = (
            data["Country"]
            .astype(str)
            .str.strip()
            .eq("")
            .astype(int)
        )

    return data


train_df = add_basic_features(
    train_df
)

val_df = add_basic_features(
    val_df
)

test_df = add_basic_features(
    test_df
)


# ============================================================
# TRAIN-ONLY PRODUCT STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("BUILDING TRAIN-ONLY CONTEXTUAL FEATURES")
print("=" * 70)


if "StockCode" in train_df.columns:

    print(
        "\nProduct-level statistics..."
    )

    product_stats = (
        train_df
        .groupby(
            "StockCode"
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
            )
        )
        .reset_index()
    )

    product_stats = (
        product_stats
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    train_df = train_df.merge(
        product_stats,
        on="StockCode",
        how="left"
    )

    val_df = val_df.merge(
        product_stats,
        on="StockCode",
        how="left"
    )

    test_df = test_df.merge(
        product_stats,
        on="StockCode",
        how="left"
    )


# ============================================================
# TRAIN-ONLY CUSTOMER STATISTICS
# ============================================================

if "Customer ID" in train_df.columns:

    print(
        "Customer-level statistics..."
    )

    customer_stats = (
        train_df
        .groupby(
            "Customer ID"
        )
        .size()
        .rename(
            "Train_Customer_Frequency"
        )
        .reset_index()
    )

    train_df = train_df.merge(
        customer_stats,
        on="Customer ID",
        how="left"
    )

    val_df = val_df.merge(
        customer_stats,
        on="Customer ID",
        how="left"
    )

    test_df = test_df.merge(
        customer_stats,
        on="Customer ID",
        how="left"
    )


# ============================================================
# TRAIN-ONLY DESCRIPTION STATISTICS
# ============================================================

if "Description" in train_df.columns:

    print(
        "Description-level statistics..."
    )

    description_stats = (
        train_df
        .groupby(
            "Description"
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

    train_df = train_df.merge(
        description_stats,
        on="Description",
        how="left"
    )

    val_df = val_df.merge(
        description_stats,
        on="Description",
        how="left"
    )

    test_df = test_df.merge(
        description_stats,
        on="Description",
        how="left"
    )


# ============================================================
# TRAIN-ONLY INVOICE STATISTICS
# ============================================================

if "Invoice" in train_df.columns:

    print(
        "Invoice-level statistics..."
    )

    invoice_stats = (
        train_df
        .groupby(
            "Invoice"
        )
        .size()
        .rename(
            "Train_Invoice_Item_Count"
        )
        .reset_index()
    )

    train_df = train_df.merge(
        invoice_stats,
        on="Invoice",
        how="left"
    )

    val_df = val_df.merge(
        invoice_stats,
        on="Invoice",
        how="left"
    )

    test_df = test_df.merge(
        invoice_stats,
        on="Invoice",
        how="left"
    )


# ============================================================
# TRAIN-ONLY DUPLICATE GROUP SIZE
# ============================================================

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

duplicate_columns = [
    c
    for c in duplicate_columns
    if c in train_df.columns
]

if duplicate_columns:

    print(
        "Duplicate-group statistics..."
    )

    duplicate_stats = (
        train_df
        .groupby(
            duplicate_columns,
            dropna=False
        )
        .size()
        .rename(
            "Train_Duplicate_Group_Size"
        )
        .reset_index()
    )

    train_df = train_df.merge(
        duplicate_stats,
        on=duplicate_columns,
        how="left"
    )

    val_df = val_df.merge(
        duplicate_stats,
        on=duplicate_columns,
        how="left"
    )

    test_df = test_df.merge(
        duplicate_stats,
        on=duplicate_columns,
        how="left"
    )


# ============================================================
# TRAIN-ONLY PRODUCT QUANTITY DISTRIBUTION
# ============================================================

if (
    "StockCode" in train_df.columns
    and "Quantity" in train_df.columns
):

    print(
        "Product quantity distribution..."
    )

    quantity_stats = (
        train_df
        .groupby(
            "StockCode"
        )["Quantity"]
        .quantile(
            [0.25, 0.75]
        )
        .unstack()
        .reset_index()
    )

    quantity_stats.columns = [
        "StockCode",
        "Train_Product_Q1_Quantity",
        "Train_Product_Q3_Quantity"
    ]

    train_df = train_df.merge(
        quantity_stats,
        on="StockCode",
        how="left"
    )

    val_df = val_df.merge(
        quantity_stats,
        on="StockCode",
        how="left"
    )

    test_df = test_df.merge(
        quantity_stats,
        on="StockCode",
        how="left"
    )

    for data in [
        train_df,
        val_df,
        test_df
    ]:

        data[
            "Train_Product_Q1_Quantity"
        ] = (
            data[
                "Train_Product_Q1_Quantity"
            ]
            .fillna(0)
        )

        data[
            "Train_Product_Q3_Quantity"
        ] = (
            data[
                "Train_Product_Q3_Quantity"
            ]
            .fillna(0)
        )

        data[
            "V51_Product_IQR_Quantity"
        ] = (
            data[
                "Train_Product_Q3_Quantity"
            ]
            -
            data[
                "Train_Product_Q1_Quantity"
            ]
        )


# ============================================================
# TRAIN-ONLY PRICE / QUANTITY DEVIATION
# ============================================================

for data in [
    train_df,
    val_df,
    test_df
]:

    if (
        "Price" in data.columns
        and
        "Train_Product_Median_Price"
        in data.columns
    ):

        data[
            "V51_Product_Price_Deviation"
        ] = (
            data["Price"]
            -
            data[
                "Train_Product_Median_Price"
            ]
        ).abs()

        data[
            "V51_Product_Price_Ratio"
        ] = (
            data["Price"]
            /
            (
                data[
                    "Train_Product_Median_Price"
                ].abs()
                + 1e-6
            )
        )

    if (
        "Quantity" in data.columns
        and
        "Train_Product_Median_Quantity"
        in data.columns
    ):

        data[
            "V51_Product_Quantity_Deviation"
        ] = (
            data["Quantity"]
            -
            data[
                "Train_Product_Median_Quantity"
            ]
        ).abs()


# ============================================================
# FILL NUMERIC VALUES
# ============================================================

for data in [
    train_df,
    val_df,
    test_df
]:

    numeric_columns = (
        data
        .select_dtypes(
            include=[np.number]
        )
        .columns
    )

    data[numeric_columns] = (
        data[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )


# ============================================================
# MODEL FEATURE SELECTION
# ============================================================

exclude_columns = {
    "Corrupted_Row_ID",
    "Original_Row_ID",

    "Missing_Value",
    "Invalid_Value",
    "Outlier",
    "Inconsistency",
    "Duplicate",

    "Missing",
    "Invalid",

    "InvoiceDate",

    "Invoice",
    "StockCode",
    "Description",
    "Country",

    "Customer ID"
}


def get_numeric_features(data):

    numeric_columns = (
        data
        .select_dtypes(
            include=[np.number]
        )
        .columns
        .tolist()
    )

    return [
        c
        for c in numeric_columns
        if c not in exclude_columns
    ]


feature_columns = get_numeric_features(
    train_df
)

print(
    "\nNumber of V5.1 model features:",
    len(feature_columns)
)

print(
    "\nV5.1 model features:"
)

for i, feature in enumerate(
    feature_columns,
    1
):

    print(
        f"{i:3d}. {feature}"
    )


# ============================================================
# ALIGN FEATURES
# ============================================================

for data in [
    train_df,
    val_df,
    test_df
]:

    for feature in feature_columns:

        if feature not in data.columns:

            data[feature] = 0

    data[feature_columns] = (
        data[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )


X_train = train_df[
    feature_columns
].copy()

X_val = val_df[
    feature_columns
].copy()

X_test = test_df[
    feature_columns
].copy()


# ============================================================
# TARGETS
# ============================================================

target_types = [
    "Missing",
    "Invalid",
    "Outlier",
    "Inconsistency",
    "Duplicate"
]

y_train = {
    target:
    train_df[target].astype(int)
    for target in target_types
}

y_val = {
    target:
    val_df[target].astype(int)
    for target in target_types
}

y_test = {
    target:
    test_df[target].astype(int)
    for target in target_types
}


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

def optimize_threshold(
    y_true,
    probabilities
):

    best_threshold = 0.50
    best_f1 = -1

    thresholds = np.arange(
        0.01,
        1.00,
        0.01
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = threshold

    return (
        best_threshold,
        best_f1
    )


# ============================================================
# TRAIN RANDOM FORESTS
# ============================================================

models = {}
thresholds = {}

performance_rows = []
importance_rows = []
threshold_rows = []

test_predictions = {}
test_probabilities = {}

print("\n" + "=" * 70)
print(
    "TRAINING TYPE-SPECIFIC RANDOM FORESTS"
)
print("=" * 70)


for target in target_types:

    print(
        "\n" + "-" * 70
    )

    print(
        "ERROR TYPE:",
        target
    )

    print(
        "-" * 70
    )

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train[target]
    )

    models[target] = model

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_probability = (
        model
        .predict_proba(X_val)[:, 1]
    )

    threshold, val_f1 = (
        optimize_threshold(
            y_val[target],
            val_probability
        )
    )

    thresholds[target] = threshold

    val_prediction = (
        val_probability >= threshold
    ).astype(int)

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_probability = (
        model
        .predict_proba(X_test)[:, 1]
    )

    test_prediction = (
        test_probability >= threshold
    ).astype(int)

    test_predictions[target] = (
        test_prediction
    )

    test_probabilities[target] = (
        test_probability
    )

    test_accuracy = accuracy_score(
        y_test[target],
        test_prediction
    )

    test_precision = precision_score(
        y_test[target],
        test_prediction,
        zero_division=0
    )

    test_recall = recall_score(
        y_test[target],
        test_prediction,
        zero_division=0
    )

    test_f1 = f1_score(
        y_test[target],
        test_prediction,
        zero_division=0
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_test[target],
            test_prediction,
            labels=[0, 1]
        ).ravel()
    )

    print(
        f"Threshold: {threshold:.2f}"
    )

    print(
        f"Validation F1: "
        f"{val_f1 * 100:.2f}%"
    )

    print(
        f"Test Accuracy : "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        f"Test Precision: "
        f"{test_precision * 100:.2f}%"
    )

    print(
        f"Test Recall   : "
        f"{test_recall * 100:.2f}%"
    )

    print(
        f"Test F1       : "
        f"{test_f1 * 100:.2f}%"
    )

    print(
        f"TP={tp} "
        f"FP={fp} "
        f"TN={tn} "
        f"FN={fn}"
    )

    performance_rows.append({

        "Error_Type": target,

        "Threshold": threshold,

        "Validation_F1": val_f1,

        "Test_Accuracy": test_accuracy,

        "Test_Precision": test_precision,

        "Test_Recall": test_recall,

        "Test_F1": test_f1,

        "TP": tp,

        "FP": fp,

        "TN": tn,

        "FN": fn
    })

    threshold_rows.append({

        "Error_Type": target,

        "Threshold": threshold,

        "Validation_F1": val_f1
    })

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    for feature, importance in zip(
        feature_columns,
        model.feature_importances_
    ):

        importance_rows.append({

            "Error_Type": target,

            "Feature": feature,

            "Importance": importance
        })


# ============================================================
# OVERALL ANOMALY DECISION
# ============================================================

print("\n" + "=" * 70)
print(
    "CALCULATING OVERALL ANOMALY PERFORMANCE"
)
print("=" * 70)


overall_prediction = np.zeros(
    len(test_df),
    dtype=int
)

for target in target_types:

    overall_prediction = np.maximum(
        overall_prediction,
        test_predictions[target]
    )


overall_truth = np.zeros(
    len(test_df),
    dtype=int
)

for target in target_types:

    overall_truth = np.maximum(
        overall_truth,
        y_test[target].values
    )


overall_accuracy = accuracy_score(
    overall_truth,
    overall_prediction
)

overall_precision = precision_score(
    overall_truth,
    overall_prediction,
    zero_division=0
)

overall_recall = recall_score(
    overall_truth,
    overall_prediction,
    zero_division=0
)

overall_f1 = f1_score(
    overall_truth,
    overall_prediction,
    zero_division=0
)

tn, fp, fn, tp = (
    confusion_matrix(
        overall_truth,
        overall_prediction,
        labels=[0, 1]
    ).ravel()
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print(
    "V5.1 FINAL TEST RESULTS"
)
print("=" * 70)

print(
    f"Accuracy : "
    f"{overall_accuracy * 100:.2f}%"
)

print(
    f"Precision: "
    f"{overall_precision * 100:.2f}%"
)

print(
    f"Recall   : "
    f"{overall_recall * 100:.2f}%"
)

print(
    f"F1 Score : "
    f"{overall_f1 * 100:.2f}%"
)

print(
    f"TP={tp} "
    f"FP={fp} "
    f"TN={tn} "
    f"FN={fn}"
)

print(
    "\nPredicted anomalies:",
    int(overall_prediction.sum())
)

print(
    "Actual anomalies:",
    int(overall_truth.sum())
)


# ============================================================
# SAVE ROW-LEVEL RESULTS
# ============================================================

result_df = test_df[
    [
        "Corrupted_Row_ID",
        "Original_Row_ID"
    ]
].copy()

for target in target_types:

    result_df[
        f"V5_1_{target}_Prediction"
    ] = test_predictions[target]

    result_df[
        f"V5_1_{target}_Truth"
    ] = y_test[target].values

    result_df[
        f"V5_1_{target}_Probability"
    ] = test_probabilities[target]


result_df[
    "V5_1_Overall_Anomaly"
] = overall_prediction

result_df[
    "V5_1_Overall_Truth"
] = overall_truth


results_path = os.path.join(
    OUTPUT_DIR,
    "type_aware_hybrid_v5_1_results.csv"
)

result_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# SAVE ERROR-TYPE PERFORMANCE
# ============================================================

performance_df = pd.DataFrame(
    performance_rows
)

performance_path = os.path.join(
    OUTPUT_DIR,
    "v5_1_error_type_performance.csv"
)

performance_df.to_csv(
    performance_path,
    index=False
)


# ============================================================
# SAVE THRESHOLD OPTIMIZATION
# ============================================================

threshold_df = pd.DataFrame(
    threshold_rows
)

threshold_path = os.path.join(
    OUTPUT_DIR,
    "v5_1_threshold_optimization.csv"
)

threshold_df.to_csv(
    threshold_path,
    index=False
)


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame(
    importance_rows
)

importance_path = os.path.join(
    OUTPUT_DIR,
    "v5_1_random_forest_feature_importance.csv"
)

importance_df.to_csv(
    importance_path,
    index=False
)


# ============================================================
# SAVE OVERALL PERFORMANCE
# ============================================================

overall_df = pd.DataFrame([
    {
        "Model": "V5.1 Leakage-Safe",

        "Accuracy": overall_accuracy,

        "Precision": overall_precision,

        "Recall": overall_recall,

        "F1_Score": overall_f1,

        "TP": tp,

        "FP": fp,

        "TN": tn,

        "FN": fn,

        "Predicted_Anomalies":
            int(
                overall_prediction.sum()
            ),

        "Actual_Anomalies":
            int(
                overall_truth.sum()
            )
    }
])

overall_path = os.path.join(
    OUTPUT_DIR,
    "v5_1_overall_performance.csv"
)

overall_df.to_csv(
    overall_path,
    index=False
)


# ============================================================
# SAVE TRAINING SUMMARY
# ============================================================

summary_df = pd.DataFrame([
    {

        "Model":
            "V5.1 Leakage-Safe",

        "Train_Rows":
            len(train_df),

        "Validation_Rows":
            len(val_df),

        "Test_Rows":
            len(test_df),

        "Number_of_Features":
            len(feature_columns),

        "Random_Forest_Trees":
            N_ESTIMATORS,

        "Min_Samples_Split":
            MIN_SAMPLES_SPLIT,

        "Min_Samples_Leaf":
            MIN_SAMPLES_LEAF,

        "Max_Features":
            MAX_FEATURES,

        "Random_State":
            RANDOM_STATE,

        "Context_Statistics":
            "TRAIN_ONLY",

        "Threshold_Tuning":
            "VALIDATION_ONLY",

        "Test_Used_For_Tuning":
            False
    }
])

summary_path = os.path.join(
    OUTPUT_DIR,
    "v5_1_training_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("V5.1 COMPLETE")
print("=" * 70)

print(
    "\nFiles saved to:"
)

print(
    OUTPUT_DIR
)

print(
    "\nSaved files:"
)

print(
    "1.",
    results_path
)

print(
    "2.",
    performance_path
)

print(
    "3.",
    threshold_path
)

print(
    "4.",
    importance_path
)

print(
    "5.",
    overall_path
)

print(
    "6.",
    summary_path
)

print("\n" + "=" * 70)
print("METHODOLOGY CHECK")
print("=" * 70)

print(
    "Contextual statistics : TRAIN ONLY"
)

print(
    "Threshold optimization: VALIDATION ONLY"
)

print(
    "Test set              : FINAL EVALUATION ONLY"
)

print(
    "V5 benchmark          : UNCHANGED"
)

print("=" * 70)