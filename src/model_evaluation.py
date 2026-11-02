import pandas as pd
import numpy as np
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# FILE PATHS
# ============================================================

GROUND_TRUTH_FILE = "results/metrics/ground_truth.csv"

ISOLATION_FILE = "results/metrics/isolation_forest_results.csv"
LOF_FILE = "results/metrics/lof_results.csv"
SVM_FILE = "results/metrics/one_class_svm_results.csv"

MODEL_COMPARISON_FILE = "results/metrics/model_comparison.csv"
ERROR_TYPE_FILE = "results/metrics/error_type_detection.csv"

os.makedirs("results/metrics", exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading ground truth...")
ground_truth = pd.read_csv(GROUND_TRUTH_FILE)

print("Loading model predictions...")
isolation = pd.read_csv(ISOLATION_FILE)
lof = pd.read_csv(LOF_FILE)
svm = pd.read_csv(SVM_FILE)


# ============================================================
# GROUND TRUTH
# ============================================================

error_columns = [
    "Missing_Value",
    "Invalid_Value",
    "Outlier",
    "Inconsistency",
    "Duplicate"
]

print("\nCreating overall ground truth...")

# A transaction is considered anomalous if it contains
# at least one injected error.

ground_truth["Ground_Truth_Anomaly"] = (
    ground_truth[error_columns].sum(axis=1) > 0
).astype(int)

print(
    "Original transactions with injected errors:",
    ground_truth["Ground_Truth_Anomaly"].sum()
)

print(
    "Total original transactions:",
    len(ground_truth)
)


# ============================================================
# FUNCTION: AGGREGATE MODEL PREDICTIONS
# ============================================================

def aggregate_predictions(df, prediction_column):

    # Multiple physical records can have the same
    # Original_Row_ID because of duplicate injection.
    #
    # If either physical record is detected as anomalous,
    # mark the original transaction as anomalous.

    aggregated = (
        df.groupby("Original_Row_ID")[prediction_column]
        .max()
        .reset_index()
    )

    aggregated["Predicted_Anomaly"] = (
        aggregated[prediction_column] == -1
    ).astype(int)

    return aggregated[
        [
            "Original_Row_ID",
            "Predicted_Anomaly"
        ]
    ]


# ============================================================
# AGGREGATE EACH MODEL
# ============================================================

print("\nAggregating predictions by Original_Row_ID...")

isolation_predictions = aggregate_predictions(
    isolation,
    "IF_Prediction"
)

lof_predictions = aggregate_predictions(
    lof,
    "LOF_Prediction"
)

svm_predictions = aggregate_predictions(
    svm,
    "SVM_Prediction"
)


# ============================================================
# MERGE WITH GROUND TRUTH
# ============================================================

evaluation_data = ground_truth[
    [
        "Original_Row_ID",
        "Ground_Truth_Anomaly"
    ] + error_columns
].copy()

evaluation_data = evaluation_data.merge(
    isolation_predictions,
    on="Original_Row_ID",
    how="left"
)

evaluation_data = evaluation_data.rename(
    columns={
        "Predicted_Anomaly": "IF_Anomaly"
    }
)

evaluation_data = evaluation_data.merge(
    lof_predictions,
    on="Original_Row_ID",
    how="left"
)

evaluation_data = evaluation_data.rename(
    columns={
        "Predicted_Anomaly": "LOF_Anomaly"
    }
)

evaluation_data = evaluation_data.merge(
    svm_predictions,
    on="Original_Row_ID",
    how="left"
)

evaluation_data = evaluation_data.rename(
    columns={
        "Predicted_Anomaly": "SVM_Anomaly"
    }
)

evaluation_data = evaluation_data.fillna(0)


# ============================================================
# MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    prediction_column
):

    y_true = evaluation_data["Ground_Truth_Anomaly"]
    y_pred = evaluation_data[prediction_column]

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "True_Positive": tp,
        "False_Positive": fp,
        "True_Negative": tn,
        "False_Negative": fn
    }


# ============================================================
# EVALUATE ALL MODELS
# ============================================================

print("\nEvaluating models...")

model_results = []

model_results.append(
    evaluate_model(
        "Isolation Forest",
        "IF_Anomaly"
    )
)

model_results.append(
    evaluate_model(
        "LOF",
        "LOF_Anomaly"
    )
)

model_results.append(
    evaluate_model(
        "One-Class SVM",
        "SVM_Anomaly"
    )
)


# ============================================================
# MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame(model_results)

comparison = comparison.sort_values(
    by="F1_Score",
    ascending=False
)

comparison.to_csv(
    MODEL_COMPARISON_FILE,
    index=False
)


# ============================================================
# ERROR-TYPE DETECTION
# ============================================================

print("\nCalculating error-type detection rates...")

model_prediction_columns = {
    "Isolation Forest": "IF_Anomaly",
    "LOF": "LOF_Anomaly",
    "One-Class SVM": "SVM_Anomaly"
}

error_type_names = {
    "Missing_Value": "Missing Value",
    "Invalid_Value": "Invalid Value",
    "Outlier": "Outlier",
    "Inconsistency": "Inconsistency",
    "Duplicate": "Duplicate"
}

error_type_results = []

for model_name, prediction_column in model_prediction_columns.items():

    for error_column, error_name in error_type_names.items():

        error_rows = evaluation_data[
            evaluation_data[error_column] == 1
        ]

        total_errors = len(error_rows)

        detected_errors = (
            error_rows[prediction_column] == 1
        ).sum()

        if total_errors > 0:
            detection_rate = (
                detected_errors / total_errors
            )
        else:
            detection_rate = 0

        error_type_results.append(
            {
                "Model": model_name,
                "Error_Type": error_name,
                "Total_Errors": total_errors,
                "Detected_Errors": detected_errors,
                "Detection_Rate": detection_rate
            }
        )


error_type_comparison = pd.DataFrame(
    error_type_results
)

error_type_comparison.to_csv(
    ERROR_TYPE_FILE,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 75)
print("                 MODEL EVALUATION RESULTS")
print("=" * 75)

print(
    comparison.to_string(index=False)
)

print("=" * 75)

print("\n")
print("=" * 75)
print("                 ERROR-TYPE DETECTION")
print("=" * 75)

display_table = error_type_comparison.copy()

display_table["Detection_Rate"] = (
    display_table["Detection_Rate"] * 100
).round(2)

print(
    display_table.to_string(index=False)
)

print("=" * 75)


# ============================================================
# SAVE EVALUATION DATA
# ============================================================

evaluation_data.to_csv(
    "results/metrics/evaluation_data.csv",
    index=False
)


print("\nResults saved to:")
print(MODEL_COMPARISON_FILE)
print(ERROR_TYPE_FILE)
print("results/metrics/evaluation_data.csv")