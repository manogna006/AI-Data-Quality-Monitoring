# ============================================================
# TYPE-AWARE HYBRID V3
# AI-Based Data Quality Monitoring
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

V2_RESULTS = os.path.join(
    BASE_DIR,
    "results",
    "metrics",
    "v2",
    "smart_hybrid_v2_results.csv"
)

V2_FEATURES = os.path.join(
    BASE_DIR,
    "results",
    "metrics",
    "v2",
    "ml_features_v2.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "metrics",
    "v3"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

ERROR_TYPES = {
    "Missing": {
        "gt": "Missing_Value",
        "dq": "DQ_Missing_Value",
        "prediction": "V3_Missing_Prediction",
        "score": "V3_Missing_Score"
    },
    "Invalid": {
        "gt": "Invalid_Value",
        "dq": "DQ_Invalid_Value",
        "prediction": "V3_Invalid_Prediction",
        "score": "V3_Invalid_Score"
    },
    "Outlier": {
        "gt": "Outlier",
        "dq": "DQ_Outlier",
        "prediction": "V3_Outlier_Prediction",
        "score": "V3_Outlier_Score"
    },
    "Inconsistency": {
        "gt": "Inconsistency",
        "dq": "DQ_Inconsistency",
        "prediction": "V3_Inconsistency_Prediction",
        "score": "V3_Inconsistency_Score"
    },
    "Duplicate": {
        "gt": "Duplicate",
        "dq": "DQ_Duplicate",
        "prediction": "V3_Duplicate_Prediction",
        "score": "V3_Duplicate_Score"
    }
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_score(series, train_series):
    """
    Robust min-max normalization using train 1st and 99th percentiles.
    """

    low = train_series.quantile(0.01)
    high = train_series.quantile(0.99)

    if high <= low:
        return pd.Series(
            np.zeros(len(series)),
            index=series.index
        )

    result = (series - low) / (high - low)

    return result.clip(0, 1)


def calculate_metrics(y_true, y_pred):

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
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
        )
    }


def find_best_threshold(y_true, scores):

    best_threshold = 0.50
    best_f1 = -1

    for threshold in np.arange(0.05, 0.96, 0.01):

        predictions = (
            scores >= threshold
        ).astype(int)

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("TYPE-AWARE HYBRID V3")
print("=" * 60)

print("\nLoading V2 results...")

results = pd.read_csv(V2_RESULTS)

print(f"V2 result rows: {len(results):,}")

print("\nLoading V2 features...")

features = pd.read_csv(V2_FEATURES)

print(f"Feature rows: {len(features):,}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Original_Row_ID",
    "Data_Split",

    "IF_Normalized_Score",
    "LOF_Normalized_Score",
    "SVM_Normalized_Score",
    "ML_Combined_Score",

    "DQ_Missing_Value",
    "DQ_Invalid_Value",
    "DQ_Outlier",
    "DQ_Inconsistency",
    "DQ_Duplicate",

    "Missing_Value",
    "Invalid_Value",
    "Outlier",
    "Inconsistency",
    "Duplicate",
    "Ground_Truth_Anomaly"
]

missing_columns = [
    c for c in required_columns
    if c not in results.columns
]

if missing_columns:

    raise ValueError(
        f"Missing columns in V2 results:\n{missing_columns}"
    )


# ============================================================
# PREPARE CONTEXTUAL FEATURES
# ============================================================

print("\nPreparing contextual features...")

# Only one row per Original_Row_ID.
# This prevents many-to-many merge problems caused by duplicates.

features_unique = (
    features
    .drop_duplicates(
        subset=["Original_Row_ID"],
        keep="first"
    )
    .copy()
)

print(
    f"Unique contextual source rows: "
    f"{len(features_unique):,}"
)


# ------------------------------------------------------------
# Contextual features
# ------------------------------------------------------------

context_features = pd.DataFrame()

context_features["Original_Row_ID"] = (
    features_unique["Original_Row_ID"]
)

# ------------------------------------------------------------
# Product context
# ------------------------------------------------------------

if "StockCode" in features_unique.columns:

    product_group = features_unique.groupby(
        "StockCode",
        dropna=False
    )

    if "Abs_Price" in features_unique.columns:

        context_features["Product_Median_Price"] = (
            features_unique["StockCode"]
            .map(
                product_group["Abs_Price"]
                .median()
            )
        )

    if "Abs_Quantity" in features_unique.columns:

        context_features["Product_Median_Quantity"] = (
            features_unique["StockCode"]
            .map(
                product_group["Abs_Quantity"]
                .median()
            )
        )

    context_features["Product_Frequency"] = (
        features_unique["StockCode"]
        .map(
            product_group.size()
        )
    )


# ------------------------------------------------------------
# Customer context
# ------------------------------------------------------------

if "Customer ID" in features_unique.columns:

    customer_group = features_unique.groupby(
        "Customer ID",
        dropna=False
    )

    context_features["Customer_Frequency"] = (
        features_unique["Customer ID"]
        .map(
            customer_group.size()
        )
    )


# ------------------------------------------------------------
# Invoice context
# ------------------------------------------------------------

if "Invoice" in features_unique.columns:

    invoice_group = features_unique.groupby(
        "Invoice",
        dropna=False
    )

    context_features["Invoice_Item_Count"] = (
        features_unique["Invoice"]
        .map(
            invoice_group.size()
        )
    )


# ------------------------------------------------------------
# Duplicate context
# ------------------------------------------------------------

duplicate_counts = (
    results
    .groupby("Original_Row_ID")
    .size()
)

context_features["Duplicate_Group_Size"] = (
    context_features["Original_Row_ID"]
    .map(duplicate_counts)
    .fillna(1)
)

context_features["Duplicate_Group_Size"] = (
    context_features["Duplicate_Group_Size"]
    .astype(float)
)

context_features["Duplicate_Group_Evidence"] = (
    context_features["Duplicate_Group_Size"] > 1
).astype(int)


# ============================================================
# MERGE CONTEXT
# ============================================================

results = results.merge(
    context_features,
    on="Original_Row_ID",
    how="left",
    validate="many_to_one"
)

print(
    f"Merged dataset rows: {len(results):,}"
)

# Critical validation
if len(results) != 530715:

    raise ValueError(
        "ERROR: Row count changed after merge. "
        f"Expected 530715 but got {len(results)}."
    )


# ============================================================
# CLEAN CONTEXT FEATURES
# ============================================================

context_columns = [
    c for c in context_features.columns
    if c != "Original_Row_ID"
]

for column in context_columns:

    results[column] = pd.to_numeric(
        results[column],
        errors="coerce"
    )

    results[column] = (
        results[column]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )


# ============================================================
# NORMALIZE CONTEXT FEATURES
# ============================================================

for column in context_columns:

    values = results[column]

    low = values.quantile(0.01)
    high = values.quantile(0.99)

    if high > low:

        results[column] = (
            (values - low) /
            (high - low)
        ).clip(0, 1)

    else:

        results[column] = 0.0


# ============================================================
# DATA SPLIT
# ============================================================

print("\nChecking data split...")

print(
    results["Data_Split"]
    .value_counts()
    .sort_index()
)


train = results[
    results["Data_Split"] == "Train"
].copy()

validation = results[
    results["Data_Split"] == "Validation"
].copy()

test = results[
    results["Data_Split"] == "Test"
].copy()


print("\nData split:")
print(
    f"Train      : {len(train):,}"
)
print(
    f"Validation : {len(validation):,}"
)
print(
    f"Test       : {len(test):,}"
)


# ============================================================
# BASE FEATURES
# ============================================================

base_features = [
    "IF_Normalized_Score",
    "LOF_Normalized_Score",
    "SVM_Normalized_Score",
    "ML_Combined_Score",

    "DQ_Missing_Value",
    "DQ_Invalid_Value",
    "DQ_Outlier",
    "DQ_Inconsistency",
    "DQ_Duplicate"
]


# Add contextual features

model_features = (
    base_features +
    context_columns
)


print(
    f"\nTotal V3 model features: "
    f"{len(model_features)}"
)

print("\nFeatures:")
for feature in model_features:
    print(f"  - {feature}")


# ============================================================
# PREPARE MATRICES
# ============================================================

X_train = train[model_features].copy()
X_validation = validation[model_features].copy()
X_test = test[model_features].copy()


# ------------------------------------------------------------
# Replace invalid values
# ------------------------------------------------------------

X_train = (
    X_train
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)

X_validation = (
    X_validation
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)

X_test = (
    X_test
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)


# ============================================================
# STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_validation_scaled = scaler.transform(
    X_validation
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# OUTPUT DATAFRAME
# ============================================================

v3_test = test[
    [
        "Original_Row_ID",
        "Corrupted_Row_ID",
        "Data_Split"
    ]
].copy()


# ============================================================
# TRAIN TYPE-SPECIFIC MODELS
# ============================================================

performance_rows = []
coefficient_rows = []
threshold_rows = []

print("\n")
print("=" * 60)
print("TRAINING TYPE-SPECIFIC V3 MODELS")
print("=" * 60)


for error_name, cfg in ERROR_TYPES.items():

    print("\n" + "-" * 60)
    print(f"Training model for: {error_name}")
    print("-" * 60)

    target_column = cfg["gt"]

    y_train = (
        train[target_column]
        .astype(int)
        .values
    )

    y_validation = (
        validation[target_column]
        .astype(int)
        .values
    )

    y_test = (
        test[target_column]
        .astype(int)
        .values
    )

    print(
        f"Training positives   : "
        f"{y_train.sum():,}"
    )

    print(
        f"Validation positives : "
        f"{y_validation.sum():,}"
    )

    print(
        f"Test positives       : "
        f"{y_test.sum():,}"
    )

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    # --------------------------------------------------------
    # Validation probability
    # --------------------------------------------------------

    validation_scores = model.predict_proba(
        X_validation_scaled
    )[:, 1]

    # --------------------------------------------------------
    # Threshold optimization
    # --------------------------------------------------------

    best_threshold, validation_f1 = (
        find_best_threshold(
            y_validation,
            validation_scores
        )
    )

    print(
        f"Best threshold      : "
        f"{best_threshold:.2f}"
    )

    print(
        f"Validation F1       : "
        f"{validation_f1:.4f}"
    )

    # --------------------------------------------------------
    # Test scores
    # --------------------------------------------------------

    test_scores = model.predict_proba(
        X_test_scaled
    )[:, 1]

    test_predictions = (
        test_scores >= best_threshold
    ).astype(int)

    metrics = calculate_metrics(
        y_test,
        test_predictions
    )

    print(
        f"Test Accuracy       : "
        f"{metrics['Accuracy']:.4f}"
    )

    print(
        f"Test Precision      : "
        f"{metrics['Precision']:.4f}"
    )

    print(
        f"Test Recall         : "
        f"{metrics['Recall']:.4f}"
    )

    print(
        f"Test F1             : "
        f"{metrics['F1_Score']:.4f}"
    )

    # --------------------------------------------------------
    # Store predictions
    # --------------------------------------------------------

    v3_test[
        cfg["score"]
    ] = test_scores

    v3_test[
        cfg["prediction"]
    ] = test_predictions

    # --------------------------------------------------------
    # Performance
    # --------------------------------------------------------

    performance_rows.append({
        "Error_Type": error_name,
        "Threshold": best_threshold,
        "Accuracy": metrics["Accuracy"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1_Score": metrics["F1_Score"],
        "Test_Positive_Count": int(y_test.sum()),
        "Predicted_Positive_Count": int(
            test_predictions.sum()
        )
    })

    # --------------------------------------------------------
    # Threshold information
    # --------------------------------------------------------

    threshold_rows.append({
        "Error_Type": error_name,
        "Best_Validation_Threshold": best_threshold,
        "Validation_F1": validation_f1
    })

    # --------------------------------------------------------
    # Model coefficients
    # --------------------------------------------------------

    for feature, coefficient in zip(
        model_features,
        model.coef_[0]
    ):

        coefficient_rows.append({
            "Error_Type": error_name,
            "Feature": feature,
            "Coefficient": coefficient,
            "Absolute_Coefficient": abs(coefficient)
        })


# ============================================================
# SAVE ERROR TYPE RESULTS
# ============================================================

performance_df = pd.DataFrame(
    performance_rows
)

performance_path = os.path.join(
    OUTPUT_DIR,
    "v3_error_type_performance.csv"
)

performance_df.to_csv(
    performance_path,
    index=False
)


threshold_df = pd.DataFrame(
    threshold_rows
)

threshold_path = os.path.join(
    OUTPUT_DIR,
    "v3_thresholds.csv"
)

threshold_df.to_csv(
    threshold_path,
    index=False
)


coefficient_df = pd.DataFrame(
    coefficient_rows
)

coefficient_path = os.path.join(
    OUTPUT_DIR,
    "v3_fusion_model_coefficients.csv"
)

coefficient_df.to_csv(
    coefficient_path,
    index=False
)


# ============================================================
# ADD GROUND TRUTH TO TEST RESULTS
# ============================================================

for error_name, cfg in ERROR_TYPES.items():

    v3_test[
        cfg["gt"]
    ] = test[
        cfg["gt"]
    ].astype(int).values


v3_test["Ground_Truth_Anomaly"] = (
    test["Ground_Truth_Anomaly"]
    .astype(int)
    .values
)


# ============================================================
# TYPE-AWARE FUSION
# ============================================================

score_columns = [
    ERROR_TYPES[name]["score"]
    for name in ERROR_TYPES
]

prediction_columns = [
    ERROR_TYPES[name]["prediction"]
    for name in ERROR_TYPES
]


# ------------------------------------------------------------
# Maximum type-specific probability
# ------------------------------------------------------------

v3_test["V3_Max_Type_Score"] = (
    v3_test[score_columns]
    .max(axis=1)
)


# ------------------------------------------------------------
# Final anomaly
# ------------------------------------------------------------

v3_test["V3_Anomaly"] = (
    v3_test[prediction_columns]
    .max(axis=1)
)


# ------------------------------------------------------------
# Predicted error type
# ------------------------------------------------------------

def determine_error_type(row):

    active_types = []

    for error_name, cfg in ERROR_TYPES.items():

        if row[cfg["prediction"]] == 1:
            active_types.append(
                (
                    error_name,
                    row[cfg["score"]]
                )
            )

    if not active_types:
        return "Normal"

    active_types.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return active_types[0][0]


v3_test["V3_Error_Type"] = (
    v3_test.apply(
        determine_error_type,
        axis=1
    )
)


# ============================================================
# OVERALL TEST PERFORMANCE
# ============================================================

y_true = (
    v3_test["Ground_Truth_Anomaly"]
    .astype(int)
)

y_pred = (
    v3_test["V3_Anomaly"]
    .astype(int)
)


overall_metrics = calculate_metrics(
    y_true,
    y_pred
)

tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
).ravel()


print("\n")
print("=" * 60)
print("V3 OVERALL TEST PERFORMANCE")
print("=" * 60)

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
    f"TP        : {tp:,}"
)

print(
    f"FP        : {fp:,}"
)

print(
    f"TN        : {tn:,}"
)

print(
    f"FN        : {fn:,}"
)

print(
    f"Predicted anomalies : "
    f"{y_pred.sum():,}"
)

print(
    f"Actual anomalies    : "
    f"{y_true.sum():,}"
)


# ============================================================
# COMPLETE ERROR TYPE PERFORMANCE
# ============================================================

print("\n")
print("=" * 60)
print("V3 ERROR-TYPE PERFORMANCE")
print("=" * 60)

for error_name, cfg in ERROR_TYPES.items():

    y_type_true = (
        v3_test[cfg["gt"]]
        .astype(int)
    )

    y_type_pred = (
        v3_test[cfg["prediction"]]
        .astype(int)
    )

    metrics = calculate_metrics(
        y_type_true,
        y_type_pred
    )

    print(f"\n{error_name}")

    print(
        f"  Precision : "
        f"{metrics['Precision'] * 100:.2f}%"
    )

    print(
        f"  Recall    : "
        f"{metrics['Recall'] * 100:.2f}%"
    )

    print(
        f"  F1 Score  : "
        f"{metrics['F1_Score'] * 100:.2f}%"
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame([
    {
        "Model": "DQ Only V1",
        "Accuracy": 0.6950,
        "Precision": 0.1408,
        "Recall": 0.9999,
        "F1_Score": 0.2469
    },
    {
        "Model": "Smart Hybrid V1",
        "Accuracy": 0.9356,
        "Precision": 0.3468,
        "Recall": 0.3260,
        "F1_Score": 0.3361
    },
    {
        "Model": "Smart Hybrid V2",
        "Accuracy": 0.9025,
        "Precision": 0.2921,
        "Recall": 0.4832,
        "F1_Score": 0.3641
    },
    {
        "Model": "Type-Aware Hybrid V3",
        "Accuracy": overall_metrics["Accuracy"],
        "Precision": overall_metrics["Precision"],
        "Recall": overall_metrics["Recall"],
        "F1_Score": overall_metrics["F1_Score"]
    }
])

comparison_path = os.path.join(
    OUTPUT_DIR,
    "v3_model_comparison.csv"
)

comparison.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# SAVE FINAL V3 RESULTS
# ============================================================

final_path = os.path.join(
    OUTPUT_DIR,
    "type_aware_hybrid_v3_results.csv"
)

v3_test.to_csv(
    final_path,
    index=False
)


# ============================================================
# TRAINING SUMMARY
# ============================================================

training_summary = pd.DataFrame({

    "Metric": [
        "Train Rows",
        "Validation Rows",
        "Test Rows",
        "Model Features",
        "V3 Test Accuracy",
        "V3 Test Precision",
        "V3 Test Recall",
        "V3 Test F1",
        "True Positives",
        "False Positives",
        "True Negatives",
        "False Negatives"
    ],

    "Value": [
        len(train),
        len(validation),
        len(test),
        len(model_features),
        overall_metrics["Accuracy"],
        overall_metrics["Precision"],
        overall_metrics["Recall"],
        overall_metrics["F1_Score"],
        tp,
        fp,
        tn,
        fn
    ]
})

summary_path = os.path.join(
    OUTPUT_DIR,
    "v3_training_validation_summary.csv"
)

training_summary.to_csv(
    summary_path,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 60)
print("V3 COMPLETE")
print("=" * 60)

print(
    "\nSaved files:"
)

print(
    f"1. {final_path}"
)

print(
    f"2. {performance_path}"
)

print(
    f"3. {threshold_path}"
)

print(
    f"4. {coefficient_path}"
)

print(
    f"5. {comparison_path}"
)

print(
    f"6. {summary_path}"
)

print("\nV3 execution completed successfully.")