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
# CONFIG
# ============================================================

V2_RESULTS = "results/metrics/v2/smart_hybrid_v2_results.csv"
V2_FEATURES = "results/metrics/v2/ml_features_v2.csv"
GROUND_TRUTH = "results/metrics/ground_truth.csv"
RAW_DATA = "datasets/corrupted/online_retail_corrupted.csv"

OUTPUT_DIR = "results/metrics/v5"

os.makedirs(OUTPUT_DIR, exist_ok=True)

ERROR_TYPES = [
    "Missing",
    "Invalid",
    "Outlier",
    "Inconsistency",
    "Duplicate"
]

RANDOM_STATE = 42


# ============================================================
# HELPER
# ============================================================

def get_metrics(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {
        "Accuracy": accuracy_score(
            y_true, y_pred
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
        "F1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn)
    }


def find_best_threshold(y_true, probabilities):

    best_threshold = 0.50
    best_f1 = -1

    for threshold in np.arange(
        0.05,
        0.991,
        0.01
    ):

        prediction = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            prediction,
            zero_division=0
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = threshold

    return best_threshold, best_f1


# ============================================================
# START
# ============================================================

print("=" * 70)
print("TYPE-AWARE HYBRID V5 - ROBUST VERSION")
print("=" * 70)


# ============================================================
# LOAD
# ============================================================

print("\nLoading V2 results...")
v2 = pd.read_csv(V2_RESULTS)

print("V2 rows:", len(v2))


print("\nLoading V2 features...")
features = pd.read_csv(V2_FEATURES)

print("Feature rows:", len(features))


print("\nLoading ground truth...")
gt = pd.read_csv(GROUND_TRUTH)

print("Ground truth rows:", len(gt))


print("\nLoading raw corrupted data...")
raw = pd.read_csv(RAW_DATA)

print("Raw rows:", len(raw))


EXPECTED_ROWS = len(v2)


# ============================================================
# CHECK ROW COUNTS
# ============================================================

if len(features) != EXPECTED_ROWS:

    raise RuntimeError(
        f"V2 feature mismatch: "
        f"{len(features)} vs {EXPECTED_ROWS}"
    )


if len(raw) != EXPECTED_ROWS:

    raise RuntimeError(
        f"Raw data mismatch: "
        f"{len(raw)} vs {EXPECTED_ROWS}"
    )


# ============================================================
# SAFE MERGE KEY
# ============================================================

if "Corrupted_Row_ID" in v2.columns:

    MERGE_KEY = ["Corrupted_Row_ID"]

elif (
    "Original_Row_ID" in v2.columns
    and
    "Original_Row_Occurrence" in v2.columns
):

    MERGE_KEY = [
        "Original_Row_ID",
        "Original_Row_Occurrence"
    ]

else:

    raise RuntimeError(
        "No safe merge key found."
    )


print(
    "\nUsing merge key:",
    MERGE_KEY
)


# ============================================================
# V2
# ============================================================

v2_use = v2.drop_duplicates(
    subset=MERGE_KEY,
    keep="first"
).copy()


features_use = features.drop_duplicates(
    subset=MERGE_KEY,
    keep="first"
).copy()


print(
    "V2 rows after deduplication:",
    len(v2_use)
)

print(
    "Feature rows after deduplication:",
    len(features_use)
)


# ============================================================
# MERGE V2 + FEATURES
# ============================================================

df = v2_use.merge(
    features_use,
    on=MERGE_KEY,
    how="left",
    suffixes=("", "_FEATURE"),
    validate="one_to_one"
)


print(
    "\nAfter V2 + feature merge:",
    len(df)
)


if len(df) != EXPECTED_ROWS:

    raise RuntimeError(
        "Row count changed during V2 + feature merge."
    )


# ============================================================
# GROUND TRUTH
# ============================================================

print("\nGround truth columns:")
print(gt.columns.tolist())


required_gt = [
    "Original_Row_ID",
    "Missing_Value",
    "Invalid_Value",
    "Outlier",
    "Inconsistency",
    "Duplicate"
]


missing_gt = [
    col
    for col in required_gt
    if col not in gt.columns
]


if missing_gt:

    raise RuntimeError(
        f"Ground truth columns missing: {missing_gt}"
    )


# ------------------------------------------------------------
# IMPORTANT FIX
# ------------------------------------------------------------
#
# V2 results may already contain columns called:
#
# Missing
# Invalid
# Outlier
# Inconsistency
# Duplicate
#
# REMOVE THEM before adding the actual ground truth.
#
# This prevents pandas from creating:
#
# Outlier_x / Outlier_y
#
# ------------------------------------------------------------

print("\nRemoving possible existing ground-truth label columns...")


for error_type in ERROR_TYPES:

    if error_type in df.columns:

        print(
            f"Removing existing V2 column: {error_type}"
        )

        df.drop(
            columns=[error_type],
            inplace=True
        )


# Also remove accidental merge suffixes if a previous file
# contains them.

suffix_columns = []

for error_type in ERROR_TYPES:

    suffix_columns.extend([
        f"{error_type}_x",
        f"{error_type}_y"
    ])


for col in suffix_columns:

    if col in df.columns:

        print(
            f"Removing conflicting column: {col}"
        )

        df.drop(
            columns=[col],
            inplace=True
        )


# ------------------------------------------------------------
# PREPARE GT
# ------------------------------------------------------------

gt_small = gt[
    required_gt
].copy()


gt_small = gt_small.drop_duplicates(
    subset=["Original_Row_ID"],
    keep="first"
)


gt_small = gt_small.rename(
    columns={
        "Missing_Value": "Missing",
        "Invalid_Value": "Invalid"
    }
)


for error_type in ERROR_TYPES:

    gt_small[error_type] = pd.to_numeric(
        gt_small[error_type],
        errors="coerce"
    ).fillna(0).astype(int)


print("\nGround truth standardized:")

for error_type in ERROR_TYPES:

    print(
        f"{error_type}:",
        error_type in gt_small.columns
    )


# ============================================================
# MERGE GROUND TRUTH
# ============================================================

if "Original_Row_ID" not in df.columns:

    raise RuntimeError(
        "Original_Row_ID is required for ground truth."
    )


df = df.merge(
    gt_small,
    on="Original_Row_ID",
    how="left",
    validate="many_to_one"
)


print(
    "\nAfter ground-truth merge:",
    len(df)
)


if len(df) != EXPECTED_ROWS:

    raise RuntimeError(
        "Row count changed during ground-truth merge."
    )


# ------------------------------------------------------------
# FINAL GUARANTEE
# ------------------------------------------------------------

for error_type in ERROR_TYPES:

    if error_type not in df.columns:

        raise RuntimeError(
            f"Ground truth column '{error_type}' "
            f"does not exist after merge."
        )

    df[error_type] = (
        pd.to_numeric(
            df[error_type],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )


print("\nGround truth columns successfully attached.")


df["Ground_Truth_Anomaly"] = (
    df[ERROR_TYPES]
    .max(axis=1)
    .astype(int)
)


print(
    "Ground-truth anomaly rows:",
    df["Ground_Truth_Anomaly"].sum()
)


# ============================================================
# RAW CONTEXT
# ============================================================

print("\nPreparing raw contextual information...")


if "Original_Row_ID" not in raw.columns:

    raise RuntimeError(
        "Original_Row_ID missing from raw dataset."
    )


# ------------------------------------------------------------
# Instead of merging ALL raw columns,
# only merge columns that are not already present.
#
# This prevents collisions such as:
#
# Quantity_x / Quantity_y
# Price_x / Price_y
# Description_x / Description_y
# ------------------------------------------------------------

raw_context_columns = [
    "Original_Row_ID",
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
    col
    for col in raw_context_columns
    if col in raw.columns
]


raw_context = raw[
    raw_context_columns
].copy()


raw_context = raw_context.drop_duplicates(
    subset=["Original_Row_ID"],
    keep="first"
)


# Only add columns that do not already exist

safe_context_columns = [
    "Original_Row_ID"
]


for col in raw_context.columns:

    if col == "Original_Row_ID":
        continue

    if col not in df.columns:

        safe_context_columns.append(col)


raw_context = raw_context[
    safe_context_columns
]


df = df.merge(
    raw_context,
    on="Original_Row_ID",
    how="left",
    validate="many_to_one"
)


print(
    "After raw context merge:",
    len(df)
)


if len(df) != EXPECTED_ROWS:

    raise RuntimeError(
        "Row count changed during raw context merge."
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\nCreating V5 contextual features...")


# ============================================================
# BASIC NUMERIC
# ============================================================

if "Quantity" in df.columns:

    df["V5_Abs_Quantity"] = (
        df["Quantity"].abs()
    )

    df["V5_Log_Quantity"] = np.log1p(
        df["Quantity"].abs()
    )

    df["V5_Quantity_Zero"] = (
        df["Quantity"] == 0
    ).astype(int)

    df["V5_Quantity_Negative"] = (
        df["Quantity"] < 0
    ).astype(int)


if "Price" in df.columns:

    df["V5_Abs_Price"] = (
        df["Price"].abs()
    )

    df["V5_Log_Price"] = np.log1p(
        df["Price"].abs()
    )

    df["V5_Price_Zero"] = (
        df["Price"] == 0
    ).astype(int)

    df["V5_Price_Negative"] = (
        df["Price"] < 0
    ).astype(int)


# ============================================================
# TOTAL AMOUNT
# ============================================================

if (
    "Quantity" in df.columns
    and
    "Price" in df.columns
):

    df["V5_Total_Amount"] = (
        df["Quantity"] *
        df["Price"]
    )

    df["V5_Abs_Total_Amount"] = (
        df["V5_Total_Amount"].abs()
    )

    df["V5_Log_Total_Amount"] = np.log1p(
        df["V5_Total_Amount"].abs()
    )


# ============================================================
# PRODUCT CONTEXT
# ============================================================

if "StockCode" in df.columns:

    print("Creating product statistics...")


    product_group = df.groupby(
        "StockCode",
        dropna=False
    )


    df["V5_Product_Frequency"] = (
        product_group["StockCode"]
        .transform("size")
    )


    if "Price" in df.columns:

        median_price = (
            product_group["Price"]
            .transform("median")
        )

        std_price = (
            product_group["Price"]
            .transform("std")
            .fillna(0)
        )

        df["V5_Product_Median_Price"] = (
            median_price
        )

        df["V5_Product_Price_Deviation"] = (
            (
                df["Price"]
                - median_price
            ).abs()
            /
            (std_price + 1e-6)
        )


    if "Quantity" in df.columns:

        median_quantity = (
            product_group["Quantity"]
            .transform("median")
        )

        q1 = (
            product_group["Quantity"]
            .transform(
                lambda x: x.quantile(0.25)
            )
        )

        q3 = (
            product_group["Quantity"]
            .transform(
                lambda x: x.quantile(0.75)
            )
        )

        iqr = q3 - q1


        df["V5_Product_Median_Quantity"] = (
            median_quantity
        )

        df["V5_Product_Q1"] = q1

        df["V5_Product_Q3"] = q3

        df["V5_Product_IQR"] = iqr

        df["V5_Product_Quantity_Deviation"] = (
            (
                df["Quantity"]
                - median_quantity
            ).abs()
            /
            (iqr + 1e-6)
        )


# ============================================================
# CUSTOMER CONTEXT
# ============================================================

if "Customer ID" in df.columns:

    print("Creating customer statistics...")


    customer_group = df.groupby(
        "Customer ID",
        dropna=False
    )


    df["V5_Customer_Frequency"] = (
        customer_group["Customer ID"]
        .transform("size")
    )


    df["V5_Customer_Missing"] = (
        df["Customer ID"]
        .isna()
        .astype(int)
    )


# ============================================================
# INVOICE CONTEXT
# ============================================================

if "Invoice" in df.columns:

    print("Creating invoice statistics...")


    invoice_group = df.groupby(
        "Invoice",
        dropna=False
    )


    df["V5_Invoice_Item_Count"] = (
        invoice_group["Invoice"]
        .transform("size")
    )


    if "Customer ID" in df.columns:

        missing_customer_count = (
            invoice_group["Customer ID"]
            .transform(
                lambda x: x.isna().sum()
            )
        )


        df[
            "V5_Invoice_Missing_Customer_Count"
        ] = missing_customer_count


        df[
            "V5_Invoice_Missing_Customer_Rate"
        ] = (
            missing_customer_count
            /
            (
                df["V5_Invoice_Item_Count"]
                + 1e-6
            )
        )


# ============================================================
# DESCRIPTION CONTEXT
# ============================================================

if "Description" in df.columns:

    print("Creating description statistics...")


    df["V5_Description_Missing"] = (
        df["Description"]
        .isna()
        .astype(int)
    )


    df["V5_Description_Length"] = (
        df["Description"]
        .fillna("")
        .astype(str)
        .str.len()
    )


    description_frequency = (
        df["Description"]
        .fillna("__MISSING__")
        .value_counts()
    )


    df["V5_Description_Frequency"] = (
        df["Description"]
        .fillna("__MISSING__")
        .map(description_frequency)
        .fillna(0)
    )


    if "StockCode" in df.columns:

        df[
            "V5_Product_Description_Count"
        ] = (
            df.groupby(
                "StockCode"
            )["Description"]
            .transform("nunique")
        )


# ============================================================
# DATETIME
# ============================================================

if "InvoiceDate" in df.columns:

    print("Creating datetime statistics...")


    dt = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce"
    )


    df["V5_Hour"] = (
        dt.dt.hour
        .fillna(-1)
    )


    df["V5_DayOfWeek"] = (
        dt.dt.dayofweek
        .fillna(-1)
    )


    df["V5_Month"] = (
        dt.dt.month
        .fillna(-1)
    )


    df["V5_IsWeekend"] = (
        dt.dt.dayofweek
        .isin([5, 6])
        .astype(int)
    )


# ============================================================
# DUPLICATE CONTEXT
# ============================================================

if (
    "Invoice" in df.columns
    and
    "StockCode" in df.columns
):

    df[
        "V5_Invoice_Product_Group_Size"
    ] = (
        df.groupby(
            ["Invoice", "StockCode"],
            dropna=False
        )["Invoice"]
        .transform("size")
    )


# ============================================================
# MODEL FEATURES
# ============================================================

print("\nPreparing model features...")


exclude_columns = {

    "Original_Row_ID",
    "Corrupted_Row_ID",
    "Original_Row_Occurrence",

    "Missing",
    "Invalid",
    "Outlier",
    "Inconsistency",
    "Duplicate",

    "Ground_Truth_Anomaly",

    "Invoice",
    "StockCode",
    "Description",
    "InvoiceDate",
    "Customer ID",
    "Country"
}


feature_columns = []


for col in df.columns:

    if col in exclude_columns:
        continue

    if pd.api.types.is_numeric_dtype(
        df[col]
    ):

        feature_columns.append(col)


feature_columns = list(
    dict.fromkeys(feature_columns)
)


print(
    "Total model features:",
    len(feature_columns)
)


# ------------------------------------------------------------
# REMOVE ANY ACCIDENTAL LABEL LEAKAGE
# ------------------------------------------------------------

for error_type in ERROR_TYPES:

    if error_type in feature_columns:

        feature_columns.remove(
            error_type
        )


X = df[
    feature_columns
].copy()


X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


X = X.fillna(0)


print(
    "Final model feature count:",
    len(feature_columns)
)


# ============================================================
# SPLIT
# ============================================================

print("\nCreating train/validation/test split...")


if "Corrupted_Row_ID" in df.columns:

    unique_ids = (
        df["Corrupted_Row_ID"]
        .dropna()
        .unique()
    )

    split_column = (
        df["Corrupted_Row_ID"]
    )

else:

    unique_ids = (
        df["Original_Row_ID"]
        .dropna()
        .unique()
    )

    split_column = (
        df["Original_Row_ID"]
    )


rng = np.random.RandomState(
    RANDOM_STATE
)

rng.shuffle(unique_ids)


n_ids = len(unique_ids)


train_end = int(
    n_ids * 0.60
)

validation_end = int(
    n_ids * 0.80
)


train_ids = set(
    unique_ids[:train_end]
)

validation_ids = set(
    unique_ids[
        train_end:validation_end
    ]
)

test_ids = set(
    unique_ids[
        validation_end:
    ]
)


train_mask = split_column.isin(
    train_ids
)

validation_mask = split_column.isin(
    validation_ids
)

test_mask = split_column.isin(
    test_ids
)


print(
    "Train rows:",
    train_mask.sum()
)

print(
    "Validation rows:",
    validation_mask.sum()
)

print(
    "Test rows:",
    test_mask.sum()
)


# ============================================================
# TRAIN MODELS
# ============================================================

performance = []
threshold_records = []
importance_records = []

prediction_data = {}


for error_type in ERROR_TYPES:

    print("\n")
    print("=" * 70)
    print(
        f"TRAINING {error_type.upper()} MODEL"
    )
    print("=" * 70)


    y = df[
        error_type
    ].astype(int)


    y_train = y[
        train_mask
    ]

    y_validation = y[
        validation_mask
    ]

    y_test = y[
        test_mask
    ]


    X_train = X[
        train_mask
    ]

    X_validation = X[
        validation_mask
    ]

    X_test = X[
        test_mask
    ]


    print(
        "Train positives:",
        y_train.sum()
    )

    print(
        "Validation positives:",
        y_validation.sum()
    )

    print(
        "Test positives:",
        y_test.sum()
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = RandomForestClassifier(

        n_estimators=350,

        max_depth=None,

        min_samples_split=4,

        min_samples_leaf=2,

        max_features="sqrt",

        class_weight="balanced",

        bootstrap=True,

        random_state=RANDOM_STATE,

        n_jobs=-1
    )


    print("Training...")


    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validation_probability = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )


    best_threshold, validation_best_f1 = (
        find_best_threshold(
            y_validation,
            validation_probability
        )
    )


    validation_prediction = (
        validation_probability
        >= best_threshold
    ).astype(int)


    validation_metrics = get_metrics(
        y_validation,
        validation_prediction
    )


    print(
        f"Best threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Validation F1: "
        f"{validation_metrics['F1'] * 100:.2f}%"
    )


    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_probability = (
        model.predict_proba(
            X_test
        )[:, 1]
    )


    test_prediction = (
        test_probability
        >= best_threshold
    ).astype(int)


    test_metrics = get_metrics(
        y_test,
        test_prediction
    )


    print("\nTEST:")

    print(
        f"Accuracy : "
        f"{test_metrics['Accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{test_metrics['Precision'] * 100:.2f}%"
    )

    print(
        f"Recall   : "
        f"{test_metrics['Recall'] * 100:.2f}%"
    )

    print(
        f"F1 Score : "
        f"{test_metrics['F1'] * 100:.2f}%"
    )


    print(
        f"TP={test_metrics['TP']} "
        f"FP={test_metrics['FP']} "
        f"TN={test_metrics['TN']} "
        f"FN={test_metrics['FN']}"
    )


    # --------------------------------------------------------
    # STORE FULL PREDICTIONS
    # --------------------------------------------------------

    full_probability = np.zeros(
        len(df)
    )

    full_prediction = np.zeros(
        len(df),
        dtype=int
    )


    full_probability[
        validation_mask
    ] = validation_probability

    full_probability[
        test_mask
    ] = test_probability


    full_prediction[
        validation_mask
    ] = validation_prediction

    full_prediction[
        test_mask
    ] = test_prediction


    prediction_data[
        f"V5_{error_type}_Probability"
    ] = full_probability


    prediction_data[
        f"V5_{error_type}_Prediction"
    ] = full_prediction


    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    performance.append({

        "Error_Type":
            error_type,

        "Threshold":
            best_threshold,

        "Validation_Accuracy":
            validation_metrics["Accuracy"],

        "Validation_Precision":
            validation_metrics["Precision"],

        "Validation_Recall":
            validation_metrics["Recall"],

        "Validation_F1":
            validation_metrics["F1"],

        "Test_Accuracy":
            test_metrics["Accuracy"],

        "Test_Precision":
            test_metrics["Precision"],

        "Test_Recall":
            test_metrics["Recall"],

        "Test_F1":
            test_metrics["F1"],

        "TP":
            test_metrics["TP"],

        "FP":
            test_metrics["FP"],

        "TN":
            test_metrics["TN"],

        "FN":
            test_metrics["FN"]
    })


    threshold_records.append({

        "Error_Type":
            error_type,

        "Best_Threshold":
            best_threshold,

        "Validation_F1":
            validation_best_f1
    })


    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    for feature, importance in zip(
        feature_columns,
        model.feature_importances_
    ):

        importance_records.append({

            "Error_Type":
                error_type,

            "Feature":
                feature,

            "Importance":
                importance
        })


# ============================================================
# ADD PREDICTIONS
# ============================================================

print("\nCombining type-aware predictions...")


for column, values in prediction_data.items():

    df[column] = values


prediction_columns = [
    f"V5_{error_type}_Prediction"
    for error_type in ERROR_TYPES
]


probability_columns = [
    f"V5_{error_type}_Probability"
    for error_type in ERROR_TYPES
]


df["V5_Anomaly"] = (
    df[prediction_columns]
    .max(axis=1)
    .astype(int)
)


df["V5_Confidence"] = (
    df[probability_columns]
    .max(axis=1)
)


def primary_type(row):

    scores = {
        error_type:
            row[
                f"V5_{error_type}_Probability"
            ]
        for error_type in ERROR_TYPES
    }

    return max(
        scores,
        key=scores.get
    )


df["V5_Predicted_Error_Type"] = (
    df.apply(
        primary_type,
        axis=1
    )
)


# ============================================================
# OVERALL TEST
# ============================================================

print("\n")
print("=" * 70)
print("V5 OVERALL TEST PERFORMANCE")
print("=" * 70)


y_true = (
    df.loc[
        test_mask,
        "Ground_Truth_Anomaly"
    ]
    .astype(int)
)


y_pred = (
    df.loc[
        test_mask,
        "V5_Anomaly"
    ]
    .astype(int)
)


overall = get_metrics(
    y_true,
    y_pred
)


print(
    f"Accuracy : "
    f"{overall['Accuracy'] * 100:.2f}%"
)

print(
    f"Precision: "
    f"{overall['Precision'] * 100:.2f}%"
)

print(
    f"Recall   : "
    f"{overall['Recall'] * 100:.2f}%"
)

print(
    f"F1 Score : "
    f"{overall['F1'] * 100:.2f}%"
)

print(
    f"TP={overall['TP']} "
    f"FP={overall['FP']} "
    f"TN={overall['TN']} "
    f"FN={overall['FN']}"
)


# ============================================================
# SAVE PERFORMANCE
# ============================================================

performance_df = pd.DataFrame(
    performance
)


threshold_df = pd.DataFrame(
    threshold_records
)


importance_df = pd.DataFrame(
    importance_records
)


overall_df = pd.DataFrame([{

    "Model":
        "Type-Aware Hybrid V5",

    "Accuracy":
        overall["Accuracy"],

    "Precision":
        overall["Precision"],

    "Recall":
        overall["Recall"],

    "F1_Score":
        overall["F1"],

    "TP":
        overall["TP"],

    "FP":
        overall["FP"],

    "TN":
        overall["TN"],

    "FN":
        overall["FN"],

    "Predicted_Anomalies":
        int(y_pred.sum()),

    "Actual_Anomalies":
        int(y_true.sum())
}])


performance_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "v5_error_type_performance.csv"
    ),
    index=False
)


threshold_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "v5_threshold_optimization.csv"
    ),
    index=False
)


importance_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "v5_random_forest_feature_importance.csv"
    ),
    index=False
)


overall_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "v5_overall_performance.csv"
    ),
    index=False
)


# ============================================================
# SAVE RESULT DATASET
# ============================================================

result_columns = [
    "Original_Row_ID"
]


if "Corrupted_Row_ID" in df.columns:

    result_columns.append(
        "Corrupted_Row_ID"
    )


result_columns += [
    "Missing",
    "Invalid",
    "Outlier",
    "Inconsistency",
    "Duplicate",
    "Ground_Truth_Anomaly"
]


result_columns += prediction_columns
result_columns += probability_columns


result_columns += [
    "V5_Anomaly",
    "V5_Confidence",
    "V5_Predicted_Error_Type"
]


result_columns = [
    col
    for col in result_columns
    if col in df.columns
]


df[
    result_columns
].to_csv(
    os.path.join(
        OUTPUT_DIR,
        "type_aware_hybrid_v5_results.csv"
    ),
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 70)
print("V5 COMPLETE")
print("=" * 70)

print(
    f"FINAL F1: "
    f"{overall['F1'] * 100:.2f}%"
)

print(
    f"FINAL Accuracy: "
    f"{overall['Accuracy'] * 100:.2f}%"
)

print(
    f"FINAL Precision: "
    f"{overall['Precision'] * 100:.2f}%"
)

print(
    f"FINAL Recall: "
    f"{overall['Recall'] * 100:.2f}%"
)

print("\nSaved to:")
print(OUTPUT_DIR)

print("\n")
print("=" * 70)