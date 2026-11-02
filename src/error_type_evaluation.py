import pandas as pd


# ============================================
# FILES
# ============================================

SMART_HYBRID_FILE = "results/metrics/smart_hybrid_results.csv"
GROUND_TRUTH_FILE = "results/metrics/ground_truth.csv"

OUTPUT_FILE = "results/metrics/error_type_evaluation.csv"


# ============================================
# LOAD FILES
# ============================================

print("Loading Smart Hybrid results...")

hybrid = pd.read_csv(SMART_HYBRID_FILE)

print(f"Smart Hybrid rows loaded: {len(hybrid)}")


print("\nLoading ground truth...")

truth = pd.read_csv(GROUND_TRUTH_FILE)

print(f"Ground truth rows loaded: {len(truth)}")


# ============================================
# SHOW COLUMNS
# ============================================

print("\nGround truth columns:")

print(truth.columns.tolist())


# ============================================
# KEEP REQUIRED COLUMNS
# ============================================

truth = truth[
    [
        "Original_Row_ID",
        "Missing_Value",
        "Invalid_Value",
        "Outlier",
        "Inconsistency",
        "Duplicate"
    ]
]


hybrid = hybrid[
    [
        "Original_Row_ID",
        "Smart_Hybrid_Anomaly"
    ]
]


# ============================================
# REMOVE DUPLICATE IDs
# ============================================

truth = (
    truth
    .groupby("Original_Row_ID", as_index=False)
    .max()
)


hybrid = (
    hybrid
    .groupby("Original_Row_ID", as_index=False)
    .max()
)


# ============================================
# MERGE
# ============================================

df = pd.merge(
    truth,
    hybrid,
    on="Original_Row_ID",
    how="inner"
)


print(
    f"\nMerged records: {len(df)}"
)


# ============================================
# ERROR TYPES
# ============================================

error_types = {
    "Missing Value": "Missing_Value",
    "Invalid Value": "Invalid_Value",
    "Outlier": "Outlier",
    "Inconsistency": "Inconsistency",
    "Duplicate": "Duplicate"
}


# ============================================
# EVALUATION
# ============================================

results = []


for error_name, column in error_types.items():

    actual = df[column].astype(int)

    predicted = df["Smart_Hybrid_Anomaly"].astype(int)


    TP = ((actual == 1) & (predicted == 1)).sum()

    FP = ((actual == 0) & (predicted == 1)).sum()

    TN = ((actual == 0) & (predicted == 0)).sum()

    FN = ((actual == 1) & (predicted == 0)).sum()


    accuracy = (
        (TP + TN) /
        (TP + TN + FP + FN)
    )


    precision = (
        TP / (TP + FP)
        if TP + FP > 0
        else 0
    )


    recall = (
        TP / (TP + FN)
        if TP + FN > 0
        else 0
    )


    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0
        else 0
    )


    results.append({
        "Error_Type": error_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "True_Positive": TP,
        "False_Positive": FP,
        "True_Negative": TN,
        "False_Negative": FN
    })


# ============================================
# RESULTS
# ============================================

result = pd.DataFrame(results)


result[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score"
    ]
] = result[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score"
    ]
].round(4)


print("\n==============================================")
print("         ERROR TYPE EVALUATION")
print("==============================================")


print(
    result.to_string(index=False)
)


# ============================================
# BEST ERROR TYPE
# ============================================

best = result.loc[
    result["F1_Score"].idxmax()
]


print("\n==============================================")
print("          BEST DETECTED ERROR TYPE")
print("==============================================")


print(
    f"Error Type : {best['Error_Type']}"
)

print(
    f"F1-score   : {best['F1_Score']:.4f}"
)


# ============================================
# SAVE
# ============================================

result.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nError type evaluation saved to:")

print(OUTPUT_FILE)


print("\n==============================================")
print("      ERROR TYPE EVALUATION COMPLETE")
print("==============================================")