import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ============================================================
# PATHS
# ============================================================

SMART_HYBRID_FILE = "results/metrics/smart_hybrid_results.csv"
OUTPUT_FILE = "results/metrics/final_model_comparison.csv"


# ============================================================
# LOAD RESULTS
# ============================================================

print("Loading Smart Hybrid results...")

df = pd.read_csv(SMART_HYBRID_FILE)

print(f"Rows loaded: {len(df):,}")


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "DQ_Anomaly",
    "IF_Anomaly",
    "LOF_Anomaly",
    "SVM_Anomaly",
    "Smart_Hybrid_Anomaly",
    "Ground_Truth_Anomaly"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# CONVERT TO BINARY
# ============================================================

for col in required_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    ).fillna(0).astype(int)


# ============================================================
# GROUND TRUTH
# ============================================================

y_true = df["Ground_Truth_Anomaly"]


# ============================================================
# MODEL PREDICTIONS
# ============================================================

predictions = {

    "DQ Only":
        df["DQ_Anomaly"],

    "Isolation Forest":
        df["IF_Anomaly"],

    "LOF":
        df["LOF_Anomaly"],

    "One-Class SVM":
        df["SVM_Anomaly"],

    # Simple OR-based hybrid baseline
    "OR Hybrid":
        (
            (df["DQ_Anomaly"] == 1) |
            (df["IF_Anomaly"] == 1) |
            (df["LOF_Anomaly"] == 1) |
            (df["SVM_Anomaly"] == 1)
        ).astype(int),

    # Proposed method
    "Smart Hybrid":
        df["Smart_Hybrid_Anomaly"]
}


# ============================================================
# CALCULATE METRICS
# ============================================================

results = []

print("\nCalculating model performance...\n")

for model_name, y_pred in predictions.items():

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

    true_positive = (
        (y_true == 1) &
        (y_pred == 1)
    ).sum()

    false_positive = (
        (y_true == 0) &
        (y_pred == 1)
    ).sum()

    true_negative = (
        (y_true == 0) &
        (y_pred == 0)
    ).sum()

    false_negative = (
        (y_true == 1) &
        (y_pred == 0)
    ).sum()

    results.append({
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1_Score": round(f1, 4),
        "True_Positive": int(true_positive),
        "False_Positive": int(false_positive),
        "True_Negative": int(true_negative),
        "False_Negative": int(false_negative)
    })


# ============================================================
# CREATE COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame(results)


# ============================================================
# RANK MODELS BY F1
# ============================================================

comparison = comparison.sort_values(
    by="F1_Score",
    ascending=False
).reset_index(drop=True)

comparison.insert(
    0,
    "Rank",
    range(1, len(comparison) + 1)
)


# ============================================================
# SAVE RESULTS
# ============================================================

comparison.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("=" * 80)
print("             FINAL MODEL COMPARISON")
print("=" * 80)

print(
    comparison[
        [
            "Rank",
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1_Score"
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 80)

best_model = comparison.iloc[0]

print(
    f"Best Model by F1: {best_model['Model']}"
)

print(
    f"Best F1 Score: {best_model['F1_Score']:.4f}"
)

print("=" * 80)

print(
    f"\nResults saved to:\n{OUTPUT_FILE}"
)