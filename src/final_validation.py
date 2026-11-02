import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ============================================================
# PATHS
# ============================================================

BASE_DIR = "results/metrics"

FILES = {
    "DQ Rules": "dq_rule_results.csv",
    "Isolation Forest": "isolation_forest_results.csv",
    "LOF": "lof_results.csv",
    "One-Class SVM": "one_class_svm_results.csv",
    "Ground Truth": "ground_truth.csv",
    "Smart Hybrid": "smart_hybrid_results.csv",
    "Final Comparison": "final_model_comparison.csv",
    "Threshold Optimization": "threshold_optimization.csv",
    "Evidence Analysis": "evidence_analysis.csv",
    "Confidence Analysis": "confidence_analysis.csv",
    "Explanation Results": "explanation_results.csv",
    "Data Quality Score": "data_quality_score.csv",
    "Error Type Evaluation": "error_type_evaluation.csv"
}


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("                 FINAL PROJECT VALIDATION")
print("=" * 75)


# ============================================================
# 1. CHECK FILES
# ============================================================

print("\n[1] Checking result files...\n")

all_files_exist = True

for name, filename in FILES.items():

    path = os.path.join(BASE_DIR, filename)

    if os.path.exists(path):
        print(f"PASS  - {name}: {filename}")
    else:
        print(f"FAIL  - {name}: {filename}")
        all_files_exist = False


# ============================================================
# 2. LOAD IMPORTANT FILES
# ============================================================

print("\n[2] Loading important result files...\n")

ground_truth = pd.read_csv(
    os.path.join(BASE_DIR, "ground_truth.csv")
)

smart_hybrid = pd.read_csv(
    os.path.join(BASE_DIR, "smart_hybrid_results.csv")
)

final_comparison = pd.read_csv(
    os.path.join(BASE_DIR, "final_model_comparison.csv")
)

threshold = pd.read_csv(
    os.path.join(BASE_DIR, "threshold_optimization.csv")
)

data_quality = pd.read_csv(
    os.path.join(BASE_DIR, "data_quality_score.csv")
)

print(f"Ground Truth rows : {len(ground_truth):,}")
print(f"Smart Hybrid rows : {len(smart_hybrid):,}")


# ============================================================
# 3. ROW COUNT VALIDATION
# ============================================================

print("\n[3] Checking row counts...\n")

EXPECTED_ROWS = 525461

checks = []

checks.append(
    ("Ground Truth row count", len(ground_truth) == EXPECTED_ROWS)
)

checks.append(
    ("Smart Hybrid row count", len(smart_hybrid) == EXPECTED_ROWS)
)

for name, result in checks:

    if result:
        print(f"PASS  - {name}")
    else:
        print(f"FAIL  - {name}")


# ============================================================
# 4. DUPLICATE ID CHECK
# ============================================================

print("\n[4] Checking Original_Row_ID uniqueness...\n")

for name, df in [
    ("Ground Truth", ground_truth),
    ("Smart Hybrid", smart_hybrid)
]:

    if "Original_Row_ID" in df.columns:

        duplicates = df["Original_Row_ID"].duplicated().sum()

        if duplicates == 0:
            print(
                f"PASS  - {name}: no duplicate Original_Row_ID"
            )
        else:
            print(
                f"FAIL  - {name}: {duplicates:,} duplicate IDs"
            )

    else:
        print(
            f"FAIL  - {name}: Original_Row_ID column missing"
        )


# ============================================================
# 5. GROUND TRUTH CHECK
# ============================================================

print("\n[5] Checking ground truth...\n")

if "Ground_Truth_Anomaly" in smart_hybrid.columns:

    anomaly_count = smart_hybrid[
        "Ground_Truth_Anomaly"
    ].sum()

    anomaly_rate = (
        anomaly_count /
        len(smart_hybrid)
    )

    print(
        f"Ground Truth anomalies : {anomaly_count:,}"
    )

    print(
        f"Ground Truth anomaly rate : {anomaly_rate:.4%}"
    )

    if anomaly_count == 26270:
        print("PASS  - Ground truth anomaly count matches expected result")
    else:
        print("WARNING - Ground truth anomaly count differs from expected")


# ============================================================
# 6. SMART HYBRID CHECK
# ============================================================

print("\n[6] Checking Smart Hybrid results...\n")

required_columns = [
    "DQ_Evidence",
    "ML_Agreement",
    "Hybrid_Confidence",
    "Smart_Hybrid_Anomaly",
    "Ground_Truth_Anomaly"
]

missing = [
    column
    for column in required_columns
    if column not in smart_hybrid.columns
]

if not missing:

    print("PASS  - Required Smart Hybrid columns are present")

    expected_confidence = (
        smart_hybrid["DQ_Evidence"] +
        smart_hybrid["ML_Agreement"]
    )

    confidence_match = (
        smart_hybrid["Hybrid_Confidence"]
        == expected_confidence
    ).all()

    if confidence_match:
        print(
            "PASS  - Hybrid confidence calculation is consistent"
        )
    else:
        print(
            "FAIL  - Hybrid confidence calculation mismatch"
        )

    expected_prediction = (
        smart_hybrid["Hybrid_Confidence"] >= 2
    ).astype(int)

    prediction_match = (
        smart_hybrid["Smart_Hybrid_Anomaly"]
        == expected_prediction
    ).all()

    if prediction_match:
        print(
            "PASS  - Smart Hybrid threshold >= 2 is consistent"
        )
    else:
        print(
            "FAIL  - Smart Hybrid threshold calculation mismatch"
        )

else:

    print(
        f"FAIL  - Missing Smart Hybrid columns: {missing}"
    )


# ============================================================
# 7. FINAL MODEL COMPARISON CHECK
# ============================================================

print("\n[7] Checking final model comparison...\n")

expected_models = {
    "DQ Only",
    "Isolation Forest",
    "LOF",
    "One-Class SVM",
    "OR Hybrid",
    "Smart Hybrid"
}

actual_models = set(
    final_comparison["Model"]
)

missing_models = expected_models - actual_models

if not missing_models:

    print("PASS  - All six models are present")

else:

    print(
        f"FAIL  - Missing models: {missing_models}"
    )


# ============================================================
# 8. BEST MODEL CHECK
# ============================================================

print("\n[8] Checking best model...\n")

best_row = final_comparison.loc[
    final_comparison["F1_Score"].idxmax()
]

print(
    f"Best model : {best_row['Model']}"
)

print(
    f"Best F1    : {best_row['F1_Score']:.4f}"
)

if best_row["Model"] == "Smart Hybrid":

    print(
        "PASS  - Smart Hybrid is the best model by F1-score"
    )

else:

    print(
        "WARNING - Smart Hybrid is not the best model by F1-score"
    )


# ============================================================
# 9. THRESHOLD CHECK
# ============================================================

print("\n[9] Checking optimized threshold...\n")

best_threshold_row = threshold.loc[
    threshold["F1_Score"].idxmax()
]

print(
    f"Best threshold : {best_threshold_row['Threshold']}"
)

print(
    f"Best F1        : {best_threshold_row['F1_Score']:.4f}"
)

if best_threshold_row["Threshold"] == 2:

    print(
        "PASS  - Optimal Smart Hybrid threshold is 2"
    )

else:

    print(
        "WARNING - Optimal threshold differs from 2"
    )


# ============================================================
# 10. DATA QUALITY SCORE CHECK
# ============================================================

print("\n[10] Checking data quality score...\n")

overall_score = None

if "Overall DQ Score" in data_quality["Quality_Dimension"].values:

    row = data_quality[
        data_quality["Quality_Dimension"]
        == "Overall DQ Score"
    ]

    overall_score = row["Quality_Score"].iloc[0]

elif "Overall" in data_quality["Quality_Dimension"].values:

    row = data_quality[
        data_quality["Quality_Dimension"]
        == "Overall"
    ]

    overall_score = row["Quality_Score"].iloc[0]

if overall_score is not None:

    print(
        f"Overall Data Quality Score : {overall_score:.2f}"
    )

else:

    print(
        "INFO - Overall DQ score stored in a different format"
    )


# ============================================================
# 11. SMART HYBRID PERFORMANCE
# ============================================================

print("\n[11] Final Smart Hybrid performance...\n")

smart_row = final_comparison[
    final_comparison["Model"]
    == "Smart Hybrid"
].iloc[0]

print(
    f"Accuracy  : {smart_row['Accuracy']:.4f}"
)

print(
    f"Precision : {smart_row['Precision']:.4f}"
)

print(
    f"Recall    : {smart_row['Recall']:.4f}"
)

print(
    f"F1 Score  : {smart_row['F1_Score']:.4f}"
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 75)
print("                    VALIDATION SUMMARY")
print("=" * 75)

if (
    all_files_exist
    and len(ground_truth) == EXPECTED_ROWS
    and len(smart_hybrid) == EXPECTED_ROWS
    and not missing_models
    and best_row["Model"] == "Smart Hybrid"
    and best_threshold_row["Threshold"] == 2
    and confidence_match
    and prediction_match
):

    print("\nSUCCESS!")
    print(
        "The core AI Data Quality Monitoring pipeline "
        "has passed final validation."
    )

else:

    print("\nWARNING!")
    print(
        "One or more validation checks require attention."
    )

print("=" * 75)