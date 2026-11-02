import pandas as pd
import os


# ============================================================
# FILE PATHS
# ============================================================

DQ_FILE = "results/metrics/dq_rule_results.csv"
IF_FILE = "results/metrics/isolation_forest_results.csv"
LOF_FILE = "results/metrics/lof_results.csv"
SVM_FILE = "results/metrics/one_class_svm_results.csv"
GROUND_TRUTH_FILE = "results/metrics/ground_truth.csv"

OUTPUT_COMPARISON = "results/metrics/hybrid_model_comparison.csv"
OUTPUT_DETAILS = "results/metrics/hybrid_evaluation_data.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading DQ results...")
dq = pd.read_csv(DQ_FILE)

print("Loading Isolation Forest results...")
isolation_forest = pd.read_csv(IF_FILE)

print("Loading LOF results...")
lof = pd.read_csv(LOF_FILE)

print("Loading One-Class SVM results...")
svm = pd.read_csv(SVM_FILE)

print("Loading ground truth...")
ground_truth = pd.read_csv(GROUND_TRUTH_FILE)


# ============================================================
# PREPARE GROUND TRUTH
# ============================================================

error_columns = [
    "Missing_Value",
    "Invalid_Value",
    "Outlier",
    "Inconsistency",
    "Duplicate"
]

ground_truth["Ground_Truth_Anomaly"] = (
    ground_truth[error_columns].sum(axis=1) > 0
).astype(int)

ground_truth = ground_truth[
    [
        "Original_Row_ID",
        "Ground_Truth_Anomaly"
    ]
]


# ============================================================
# PREPARE DQ PREDICTIONS
# ============================================================

dq_predictions = dq[
    [
        "Original_Row_ID",
        "DQ_Anomaly"
    ]
].copy()

dq_predictions["DQ_Anomaly"] = (
    dq_predictions["DQ_Anomaly"] > 0
).astype(int)


# ============================================================
# PREPARE ML PREDICTIONS
# ============================================================

def prepare_ml_predictions(
    df,
    prediction_column,
    output_name
):
    """
    Convert ML predictions to binary anomaly flags.

    ML convention:
        -1 = anomaly
         1 = normal

    Duplicate physical rows are aggregated using
    Original_Row_ID.
    """

    temp = df[
        [
            "Original_Row_ID",
            prediction_column
        ]
    ].copy()

    temp["ML_Anomaly"] = (
        temp[prediction_column] == -1
    ).astype(int)

    result = (
        temp
        .groupby("Original_Row_ID")["ML_Anomaly"]
        .max()
        .reset_index()
    )

    result.rename(
        columns={
            "ML_Anomaly": output_name
        },
        inplace=True
    )

    return result


if_predictions = prepare_ml_predictions(
    isolation_forest,
    "IF_Prediction",
    "Isolation_Forest"
)

lof_predictions = prepare_ml_predictions(
    lof,
    "LOF_Prediction",
    "LOF"
)

svm_predictions = prepare_ml_predictions(
    svm,
    "SVM_Prediction",
    "One_Class_SVM"
)


# ============================================================
# MERGE ALL RESULTS
# ============================================================

print("\nMerging predictions...")

evaluation = ground_truth.merge(
    dq_predictions,
    on="Original_Row_ID",
    how="left"
)

evaluation = evaluation.merge(
    if_predictions,
    on="Original_Row_ID",
    how="left"
)

evaluation = evaluation.merge(
    lof_predictions,
    on="Original_Row_ID",
    how="left"
)

evaluation = evaluation.merge(
    svm_predictions,
    on="Original_Row_ID",
    how="left"
)


# ============================================================
# CLEAN PREDICTIONS
# ============================================================

prediction_columns = [
    "DQ_Anomaly",
    "Isolation_Forest",
    "LOF",
    "One_Class_SVM"
]

evaluation[prediction_columns] = (
    evaluation[prediction_columns]
    .fillna(0)
    .astype(int)
)


# ============================================================
# HYBRID DETECTION
# ============================================================
#
# Hybrid = DQ rule OR any ML detector
#
# If at least one component identifies the transaction
# as anomalous, the hybrid system flags it.

evaluation["Hybrid_DQ_ML"] = (
    (
        evaluation["DQ_Anomaly"] == 1
    )
    |
    (
        evaluation[
            [
                "Isolation_Forest",
                "LOF",
                "One_Class_SVM"
            ]
        ].max(axis=1) == 1
    )
).astype(int)


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(y_true, y_pred):

    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    accuracy = (
        (tp + tn) / len(y_true)
        if len(y_true) > 0
        else 0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
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
# MODEL COMPARISON
# ============================================================

models = {
    "DQ Only": "DQ_Anomaly",
    "Isolation Forest": "Isolation_Forest",
    "LOF": "LOF",
    "One-Class SVM": "One_Class_SVM",
    "Hybrid DQ + ML": "Hybrid_DQ_ML"
}


results = []

y_true = evaluation["Ground_Truth_Anomaly"]

for model_name, prediction_column in models.items():

    metrics = calculate_metrics(
        y_true,
        evaluation[prediction_column]
    )

    metrics["Model"] = model_name

    results.append(metrics)


comparison = pd.DataFrame(results)

comparison = comparison[
    [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score",
        "True_Positive",
        "False_Positive",
        "True_Negative",
        "False_Negative"
    ]
]


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("       DQ + ML MODEL COMPARISON")
print("========================================")

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("========================================")


# ============================================================
# BEST MODEL
# ============================================================

best_model = comparison.loc[
    comparison["F1_Score"].idxmax()
]

print("\nBest model based on F1-score:")
print(f"Model     : {best_model['Model']}")
print(f"F1-score  : {best_model['F1_Score']:.4f}")
print(f"Precision : {best_model['Precision']:.4f}")
print(f"Recall    : {best_model['Recall']:.4f}")


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    "results/metrics",
    exist_ok=True
)

comparison.to_csv(
    OUTPUT_COMPARISON,
    index=False
)

evaluation.to_csv(
    OUTPUT_DETAILS,
    index=False
)

print("\nSaved model comparison to:")
print(OUTPUT_COMPARISON)

print("\nSaved detailed evaluation data to:")
print(OUTPUT_DETAILS)