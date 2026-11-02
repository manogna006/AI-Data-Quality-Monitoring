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

OUTPUT_FILE = "results/metrics/smart_hybrid_results.csv"
COMPARISON_FILE = "results/metrics/smart_hybrid_comparison.csv"


# ============================================================
# LOAD RESULTS
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
# GROUND TRUTH
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
    ["Original_Row_ID", "Ground_Truth_Anomaly"]
]


# ============================================================
# DQ INFORMATION
# ============================================================

dq_columns = [
    "Original_Row_ID",
    "DQ_Missing_Value",
    "DQ_Invalid_Value",
    "DQ_Outlier",
    "DQ_Inconsistency",
    "DQ_Duplicate",
    "DQ_Error_Count",
    "DQ_Anomaly",
    "DQ_Row_Score"
]

dq_selected = dq[dq_columns].copy()


# ============================================================
# PREPARE ML PREDICTIONS
# ============================================================

def prepare_ml(df, prediction_column, output_name):

    temp = df[
        [
            "Original_Row_ID",
            prediction_column
        ]
    ].copy()

    temp[output_name] = (
        temp[prediction_column] == -1
    ).astype(int)

    # Handle duplicate physical rows
    result = (
        temp
        .groupby("Original_Row_ID")[output_name]
        .max()
        .reset_index()
    )

    return result


if_result = prepare_ml(
    isolation_forest,
    "IF_Prediction",
    "IF_Anomaly"
)

lof_result = prepare_ml(
    lof,
    "LOF_Prediction",
    "LOF_Anomaly"
)

svm_result = prepare_ml(
    svm,
    "SVM_Prediction",
    "SVM_Anomaly"
)


# ============================================================
# MERGE EVERYTHING
# ============================================================

print("\nCombining DQ and ML evidence...")

evaluation = ground_truth.merge(
    dq_selected,
    on="Original_Row_ID",
    how="left"
)

evaluation = evaluation.merge(
    if_result,
    on="Original_Row_ID",
    how="left"
)

evaluation = evaluation.merge(
    lof_result,
    on="Original_Row_ID",
    how="left"
)

evaluation = evaluation.merge(
    svm_result,
    on="Original_Row_ID",
    how="left"
)
 # ============================================================
# FINAL ROW-LEVEL AGGREGATION
# ============================================================

# Multiple physical rows may refer to the same
# Original_Row_ID. The evaluation must operate at
# the original-record level.

print("\nChecking duplicate Original_Row_ID values...")

duplicate_count = (
    evaluation["Original_Row_ID"]
    .duplicated()
    .sum()
)

print(
    f"Duplicate physical rows found: {duplicate_count}"
)


if duplicate_count > 0:

    print(
        "Aggregating duplicate Original_Row_ID values..."
    )

    aggregation_rules = {

        # Ground truth:
        # if any representation is anomalous,
        # retain anomaly = 1
        "Ground_Truth_Anomaly": "max",

        # DQ evidence:
        "DQ_Missing_Value": "max",
        "DQ_Invalid_Value": "max",
        "DQ_Outlier": "max",
        "DQ_Inconsistency": "max",
        "DQ_Duplicate": "max",

        # DQ summary values:
        "DQ_Error_Count": "max",
        "DQ_Anomaly": "max",
        "DQ_Row_Score": "max",

        # ML predictions:
        "IF_Anomaly": "max",
        "LOF_Anomaly": "max",
        "SVM_Anomaly": "max"
    }

    evaluation = (
        evaluation
        .groupby(
            "Original_Row_ID",
            as_index=False
        )
        .agg(aggregation_rules)
    )

    print(
        f"Rows after aggregation: "
        f"{len(evaluation)}"
    )

else:

    print(
        "No duplicate Original_Row_ID values found."
    )

# ============================================================
# CLEAN MISSING VALUES
# ============================================================

numeric_columns = [
    "DQ_Missing_Value",
    "DQ_Invalid_Value",
    "DQ_Outlier",
    "DQ_Inconsistency",
    "DQ_Duplicate",
    "DQ_Error_Count",
    "DQ_Anomaly",
    "DQ_Row_Score",
    "IF_Anomaly",
    "LOF_Anomaly",
    "SVM_Anomaly"
]

evaluation[numeric_columns] = (
    evaluation[numeric_columns]
    .fillna(0)
)


# ============================================================
# ML CONSENSUS
# ============================================================

evaluation["ML_Agreement"] = (
    evaluation[
        [
            "IF_Anomaly",
            "LOF_Anomaly",
            "SVM_Anomaly"
        ]
    ].sum(axis=1)
)


# ============================================================
# DQ EVIDENCE SCORE
# ============================================================

# Each independent DQ violation contributes evidence.

evaluation["DQ_Evidence"] = (
    evaluation[
        [
            "DQ_Missing_Value",
            "DQ_Invalid_Value",
            "DQ_Outlier",
            "DQ_Inconsistency",
            "DQ_Duplicate"
        ]
    ].sum(axis=1)
)


# ============================================================
# HYBRID CONFIDENCE SCORE
# ============================================================

# DQ evidence:
#   0 violations = 0
#   1 violation  = 1
#   2+ violations = stronger evidence
#
# ML evidence:
#   1 model agrees  = moderate
#   2 models agree  = stronger
#   3 models agree  = strongest
#
# Maximum score = 8

evaluation["Hybrid_Confidence"] = (
    evaluation["DQ_Evidence"]
    +
    evaluation["ML_Agreement"]
)


# ============================================================
# SMART HYBRID DECISION
# ============================================================

# Decision logic:
#
# 0 = Normal
# 1 = DQ-only issue
# 2 = ML-supported / stronger DQ evidence
# 3+ = High-confidence anomaly
#
# We use a threshold of 2 for the primary model.

evaluation["Smart_Hybrid_Anomaly"] = (
    evaluation["Hybrid_Confidence"] >= 2
).astype(int)


# ============================================================
# CONFIDENCE LEVEL
# ============================================================

def confidence_level(score):

    if score == 0:
        return "Normal"

    elif score == 1:
        return "Low"

    elif score == 2:
        return "Medium"

    else:
        return "High"


evaluation["Confidence_Level"] = (
    evaluation["Hybrid_Confidence"]
    .apply(confidence_level)
)


# ============================================================
# ERROR CLASSIFICATION
# ============================================================

def classify_error(row):

    # Deterministic DQ classifications take priority

    if row["DQ_Missing_Value"] == 1:
        return "Missing Value"

    if row["DQ_Invalid_Value"] == 1:
        return "Invalid Value"

    if row["DQ_Outlier"] == 1:
        return "Outlier"

    if row["DQ_Inconsistency"] == 1:
        return "Inconsistency"

    if row["DQ_Duplicate"] == 1:
        return "Duplicate"

    # If no DQ rule triggered but ML models agree
    if row["ML_Agreement"] >= 2:
        return "ML Anomaly"

    if row["ML_Agreement"] == 1:
        return "Potential Anomaly"

    return "Normal"


evaluation["Detected_Error_Type"] = (
    evaluation.apply(
        classify_error,
        axis=1
    )
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    accuracy = (
        (tp + tn) / len(y_true)
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0
        else 0
    )

    return [
        accuracy,
        precision,
        recall,
        f1,
        tp,
        fp,
        tn,
        fn
    ]


# ============================================================
# MODEL COMPARISON
# ============================================================

models = {
    "DQ Only": "DQ_Anomaly",
    "Isolation Forest": "IF_Anomaly",
    "LOF": "LOF_Anomaly",
    "One-Class SVM": "SVM_Anomaly",
    "Smart Hybrid": "Smart_Hybrid_Anomaly"
}


results = []

y_true = evaluation["Ground_Truth_Anomaly"]

for model_name, column in models.items():

    metrics = calculate_metrics(
        y_true,
        evaluation[column]
    )

    results.append(
        [
            model_name,
            *metrics
        ]
    )


comparison = pd.DataFrame(
    results,
    columns=[
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
)


# ============================================================
# DISPLAY
# ============================================================

print("\n========================================")
print("       SMART HYBRID MODEL RESULTS")
print("========================================")

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("========================================")


# ============================================================
# CONFIDENCE DISTRIBUTION
# ============================================================

print("\nConfidence distribution:")

print(
    evaluation[
        "Confidence_Level"
    ].value_counts()
)


# ============================================================
# ERROR TYPE DISTRIBUTION
# ============================================================

print("\nDetected error types:")

print(
    evaluation[
        "Detected_Error_Type"
    ].value_counts()
)


# ============================================================
# BEST MODEL
# ============================================================

best = comparison.loc[
    comparison["F1_Score"].idxmax()
]

print("\nBest model based on F1-score:")

print(
    f"Model     : {best['Model']}"
)

print(
    f"F1-score  : {best['F1_Score']:.4f}"
)

print(
    f"Precision : {best['Precision']:.4f}"
)

print(
    f"Recall    : {best['Recall']:.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    "results/metrics",
    exist_ok=True
)

evaluation.to_csv(
    OUTPUT_FILE,
    index=False
)

comparison.to_csv(
    COMPARISON_FILE,
    index=False
)


print("\nSaved detailed smart hybrid results to:")
print(OUTPUT_FILE)

print("\nSaved model comparison to:")
print(COMPARISON_FILE)