import pandas as pd
import os


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = "results/metrics/smart_hybrid_results.csv"

OUTPUT_FILE = "results/metrics/threshold_optimization.csv"


# ============================================================
# LOAD SMART HYBRID RESULTS
# ============================================================

print("Loading Smart Hybrid results...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df)}")


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Ground_Truth_Anomaly",
    "Hybrid_Confidence"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Required column '{column}' "
            f"not found in {INPUT_FILE}"
        )


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
# THRESHOLD RANGE
# ============================================================

# Determine the maximum confidence present in the dataset.

max_confidence = int(
    df["Hybrid_Confidence"].max()
)


print(
    f"Maximum Hybrid Confidence: "
    f"{max_confidence}"
)


# ============================================================
# TEST EVERY POSSIBLE THRESHOLD
# ============================================================

results = []


y_true = df["Ground_Truth_Anomaly"]


for threshold in range(
    1,
    max_confidence + 1
):

    # --------------------------------------------------------
    # Smart Hybrid decision
    # --------------------------------------------------------

    y_pred = (
        df["Hybrid_Confidence"] >= threshold
    ).astype(int)


    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        y_true,
        y_pred
    )


    results.append(
        [
            threshold,
            *metrics
        ]
    )


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

comparison = pd.DataFrame(

    results,

    columns=[
        "Threshold",
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
# SORT BY F1 SCORE
# ============================================================

comparison_sorted = comparison.sort_values(
    by="F1_Score",
    ascending=False
).reset_index(
    drop=True
)


# ============================================================
# BEST THRESHOLD
# ============================================================

best = comparison_sorted.iloc[0]


# ============================================================
# DISPLAY ALL RESULTS
# ============================================================

print("\n==============================================")
print("       SMART HYBRID THRESHOLD ANALYSIS")
print("==============================================")

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("==============================================")


# ============================================================
# BEST THRESHOLD
# ============================================================

print("\nBest threshold based on F1-score:")

print(
    f"Threshold : "
    f"{int(best['Threshold'])}"
)

print(
    f"Accuracy  : "
    f"{best['Accuracy']:.4f}"
)

print(
    f"Precision : "
    f"{best['Precision']:.4f}"
)

print(
    f"Recall    : "
    f"{best['Recall']:.4f}"
)

print(
    f"F1-score  : "
    f"{best['F1_Score']:.4f}"
)

print(
    f"TP        : "
    f"{int(best['True_Positive'])}"
)

print(
    f"FP        : "
    f"{int(best['False_Positive'])}"
)

print(
    f"TN        : "
    f"{int(best['True_Negative'])}"
)

print(
    f"FN        : "
    f"{int(best['False_Negative'])}"
)


# ============================================================
# TOP 5 THRESHOLDS
# ============================================================

print("\nTop 5 thresholds by F1-score:")

top5 = comparison_sorted.head(5)

print(
    top5[
        [
            "Threshold",
            "Accuracy",
            "Precision",
            "Recall",
            "F1_Score"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    "results/metrics",
    exist_ok=True
)

comparison.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\nThreshold analysis saved to:"
)

print(
    OUTPUT_FILE
)