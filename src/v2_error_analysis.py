import os
import pandas as pd
import numpy as np

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

INPUT_FILE = "results/metrics/v2/smart_hybrid_v2_results.csv"
OUTPUT_DIR = "results/metrics/v2"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def binary_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "TP": int(((y_true == 1) & (y_pred == 1)).sum()),
        "FP": int(((y_true == 0) & (y_pred == 1)).sum()),
        "TN": int(((y_true == 0) & (y_pred == 0)).sum()),
        "FN": int(((y_true == 1) & (y_pred == 0)).sum()),
    }


def print_metrics(name, y_true, y_pred):
    m = binary_metrics(y_true, y_pred)

    print(f"\n{name}")
    print("-" * 60)
    print(f"Accuracy : {m['Accuracy']:.4f}")
    print(f"Precision: {m['Precision']:.4f}")
    print(f"Recall   : {m['Recall']:.4f}")
    print(f"F1       : {m['F1']:.4f}")
    print(f"TP       : {m['TP']}")
    print(f"FP       : {m['FP']}")
    print(f"TN       : {m['TN']}")
    print(f"FN       : {m['FN']}")

    return m


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("V2 ERROR AND EVIDENCE ANALYSIS")
print("=" * 70)

print("\nLoading V2 results...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df):,}")
print(f"Columns loaded: {len(df.columns)}")


# ============================================================
# USE TEST SET ONLY FOR FINAL ERROR ANALYSIS
# ============================================================

test = df[df["Data_Split"] == "Test"].copy()

print(f"\nTest rows: {len(test):,}")

if len(test) == 0:
    raise ValueError("No Test rows found.")


# ============================================================
# BASIC GROUND-TRUTH DISTRIBUTION
# ============================================================

gt_cols = {
    "Missing": "Missing_Value",
    "Invalid": "Invalid_Value",
    "Outlier": "Outlier",
    "Inconsistency": "Inconsistency",
    "Duplicate": "Duplicate"
}

print("\n" + "=" * 70)
print("GROUND TRUTH ERROR DISTRIBUTION")
print("=" * 70)

gt_distribution = []

for error_name, col in gt_cols.items():
    count = int(test[col].sum())
    rate = count / len(test)

    gt_distribution.append({
        "Error_Type": error_name,
        "Count": count,
        "Rate": rate
    })

    print(
        f"{error_name:15s}: "
        f"{count:6,d} "
        f"({rate:.4%})"
    )

gt_anomaly_count = int(test["Ground_Truth_Anomaly"].sum())

print(
    f"\nAny injected anomaly: "
    f"{gt_anomaly_count:,} "
    f"({gt_anomaly_count / len(test):.4%})"
)


# ============================================================
# MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON ON TEST SET")
print("=" * 70)

model_results = []

models = {
    "Isolation Forest": "IF_Prediction",
    "LOF": "LOF_Prediction",
    "One-Class SVM": "SVM_Prediction",
    "Smart Hybrid V2": "Smart_Hybrid_V2_Anomaly"
}

for name, pred_col in models.items():

    # sklearn models use -1 = anomaly, 1 = normal
    if pred_col in [
        "IF_Prediction",
        "LOF_Prediction",
        "SVM_Prediction"
    ]:
        y_pred = (test[pred_col] == -1).astype(int)
    else:
        y_pred = test[pred_col].astype(int)

    y_true = test["Ground_Truth_Anomaly"].astype(int)

    metrics = binary_metrics(y_true, y_pred)

    model_results.append({
        "Model": name,
        **metrics
    })

    print_metrics(name, y_true, y_pred)


model_comparison = pd.DataFrame(model_results)

model_comparison.to_csv(
    os.path.join(OUTPUT_DIR, "v2_error_analysis_model_comparison.csv"),
    index=False
)


# ============================================================
# ERROR-TYPE PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("ERROR-TYPE PERFORMANCE")
print("=" * 70)

type_results = []

# DQ flags
dq_predictions = {
    "DQ Missing": "DQ_Missing_Value",
    "DQ Invalid": "DQ_Invalid_Value",
    "DQ Outlier": "DQ_Outlier",
    "DQ Inconsistency": "DQ_Inconsistency",
    "DQ Duplicate": "DQ_Duplicate"
}

# Ground truth
for error_name, gt_col in gt_cols.items():

    print(f"\n--- {error_name} ---")

    # Matching DQ rule
    dq_name = f"DQ {error_name}"
    dq_col = dq_predictions[dq_name]

    y_true = test[gt_col].astype(int)
    y_pred = test[dq_col].astype(int)

    metrics = binary_metrics(y_true, y_pred)

    print(
        f"DQ -> "
        f"Precision={metrics['Precision']:.4f}, "
        f"Recall={metrics['Recall']:.4f}, "
        f"F1={metrics['F1']:.4f}"
    )

    type_results.append({
        "Error_Type": error_name,
        "Method": "DQ Rule",
        **metrics
    })

    # Hybrid prediction against this particular error type
    hybrid_pred = test["Smart_Hybrid_V2_Anomaly"].astype(int)

    hybrid_metrics = binary_metrics(y_true, hybrid_pred)

    print(
        f"Hybrid -> "
        f"Precision={hybrid_metrics['Precision']:.4f}, "
        f"Recall={hybrid_metrics['Recall']:.4f}, "
        f"F1={hybrid_metrics['F1']:.4f}"
    )

    type_results.append({
        "Error_Type": error_name,
        "Method": "Smart Hybrid V2",
        **hybrid_metrics
    })


type_results_df = pd.DataFrame(type_results)

type_results_df.to_csv(
    os.path.join(OUTPUT_DIR, "v2_error_type_analysis.csv"),
    index=False
)


# ============================================================
# DQ EVIDENCE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("DQ EVIDENCE ANALYSIS")
print("=" * 70)

evidence_analysis = []

for evidence_level in sorted(test["DQ_Evidence"].unique()):

    subset = test[test["DQ_Evidence"] == evidence_level]

    anomaly_rate = subset["Ground_Truth_Anomaly"].mean()

    evidence_analysis.append({
        "DQ_Evidence": evidence_level,
        "Rows": len(subset),
        "Ground_Truth_Anomalies": int(
            subset["Ground_Truth_Anomaly"].sum()
        ),
        "Ground_Truth_Anomaly_Rate": anomaly_rate,
        "Hybrid_Anomaly_Rate": subset[
            "Smart_Hybrid_V2_Anomaly"
        ].mean()
    })

    print(
        f"DQ Evidence {evidence_level}: "
        f"{len(subset):,} rows | "
        f"GT anomaly rate = {anomaly_rate:.4%}"
    )


evidence_df = pd.DataFrame(evidence_analysis)

evidence_df.to_csv(
    os.path.join(OUTPUT_DIR, "v2_dq_evidence_analysis.csv"),
    index=False
)


# ============================================================
# ML SCORE DECILE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("ML SCORE ANALYSIS")
print("=" * 70)

score_cols = [
    "IF_Normalized_Score",
    "LOF_Normalized_Score",
    "SVM_Normalized_Score",
    "ML_Combined_Score",
    "Hybrid_Score"
]

score_results = []

for score_col in score_cols:

    temp = test[
        [
            score_col,
            "Ground_Truth_Anomaly"
        ]
    ].copy()

    # Handle duplicate values safely
    try:
        temp["Score_Decile"] = pd.qcut(
            temp[score_col],
            q=10,
            labels=False,
            duplicates="drop"
        )
    except ValueError:
        continue

    grouped = temp.groupby("Score_Decile", observed=True).agg(
        Rows=("Ground_Truth_Anomaly", "size"),
        Ground_Truth_Anomalies=("Ground_Truth_Anomaly", "sum"),
        Mean_Score=(score_col, "mean"),
        Anomaly_Rate=("Ground_Truth_Anomaly", "mean")
    ).reset_index()

    grouped["Score"] = score_col

    score_results.append(grouped)

    print(f"\n{score_col}")
    print(grouped.to_string(index=False))


if score_results:
    score_analysis_df = pd.concat(
        score_results,
        ignore_index=True
    )

    score_analysis_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "v2_ml_score_decile_analysis.csv"
        ),
        index=False
    )


# ============================================================
# FALSE POSITIVE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("FALSE POSITIVE ANALYSIS")
print("=" * 70)

hybrid_pred = test["Smart_Hybrid_V2_Anomaly"].astype(int)
gt = test["Ground_Truth_Anomaly"].astype(int)

fp = test[
    (hybrid_pred == 1) &
    (gt == 0)
].copy()

fn = test[
    (hybrid_pred == 0) &
    (gt == 1)
].copy()

print(f"False positives: {len(fp):,}")
print(f"False negatives: {len(fn):,}")


# DQ evidence distribution among false positives
if len(fp) > 0:

    fp_dq = (
        fp["DQ_Evidence"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    fp_dq.columns = [
        "DQ_Evidence",
        "False_Positive_Count"
    ]

    fp_dq["Percentage"] = (
        fp_dq["False_Positive_Count"] /
        len(fp)
    )

    fp_dq.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "v2_false_positive_dq_evidence.csv"
        ),
        index=False
    )

    print("\nFalse positives by DQ evidence:")
    print(fp_dq.to_string(index=False))


# ============================================================
# FALSE NEGATIVE ANALYSIS
# ============================================================

if len(fn) > 0:

    fn_types = []

    for error_name, gt_col in gt_cols.items():

        count = int(fn[gt_col].sum())

        fn_types.append({
            "Error_Type": error_name,
            "False_Negative_Rows": count
        })

    fn_types_df = pd.DataFrame(fn_types)

    fn_types_df["Percentage"] = (
        fn_types_df["False_Negative_Rows"] /
        len(fn)
    )

    fn_types_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "v2_false_negative_error_types.csv"
        ),
        index=False
    )

    print("\nFalse negatives by ground-truth error type:")
    print(fn_types_df.to_string(index=False))


# ============================================================
# ERROR CO-OCCURRENCE
# ============================================================

print("\n" + "=" * 70)
print("GROUND-TRUTH ERROR CO-OCCURRENCE")
print("=" * 70)

gt_matrix = test[
    list(gt_cols.values())
].astype(int)

cooccurrence = pd.DataFrame(
    index=gt_cols.keys(),
    columns=gt_cols.keys(),
    dtype=int
)

for row_name, row_col in gt_cols.items():

    for col_name, col_col in gt_cols.items():

        cooccurrence.loc[row_name, col_name] = int(
            (
                (test[row_col] == 1) &
                (test[col_col] == 1)
            ).sum()
        )

print(cooccurrence)

cooccurrence.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "v2_ground_truth_error_cooccurrence.csv"
    )
)


# ============================================================
# DUPLICATE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE ANALYSIS")
print("=" * 70)

duplicate_counts = (
    test.groupby("Original_Row_ID")
    .agg(
        Occurrences=("Original_Row_ID", "size"),
        GT_Duplicate=("Duplicate", "max"),
        Hybrid_Anomaly=("Smart_Hybrid_V2_Anomaly", "max"),
        DQ_Duplicate=("DQ_Duplicate", "max")
    )
    .reset_index()
)

duplicate_counts["Is_Actual_Duplicate"] = (
    duplicate_counts["Occurrences"] > 1
).astype(int)

print(
    f"Unique Original_Row_IDs in test: "
    f"{len(duplicate_counts):,}"
)

print(
    f"IDs with multiple occurrences: "
    f"{int(duplicate_counts['Is_Actual_Duplicate'].sum()):,}"
)

duplicate_counts.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "v2_duplicate_group_analysis.csv"
    ),
    index=False
)


# ============================================================
# SAVE GROUND TRUTH DISTRIBUTION
# ============================================================

pd.DataFrame(gt_distribution).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "v2_ground_truth_distribution.csv"
    ),
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print("\nGenerated files:")

output_files = [
    "v2_error_analysis_model_comparison.csv",
    "v2_error_type_analysis.csv",
    "v2_dq_evidence_analysis.csv",
    "v2_ml_score_decile_analysis.csv",
    "v2_false_positive_dq_evidence.csv",
    "v2_false_negative_error_types.csv",
    "v2_ground_truth_error_cooccurrence.csv",
    "v2_duplicate_group_analysis.csv",
    "v2_ground_truth_distribution.csv"
]

for filename in output_files:
    path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(path):
        print(f"  [OK] {filename}")

print("\nUse these results to identify the actual V2 bottleneck")
print("before changing the hybrid model.")