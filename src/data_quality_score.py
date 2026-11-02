import pandas as pd


# ============================================
# DATA QUALITY SCORE
# ============================================

INPUT_FILE = "results/metrics/smart_hybrid_results.csv"
OUTPUT_FILE = "results/metrics/data_quality_score.csv"


print("Loading Smart Hybrid results...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df)}")


# ============================================
# ERROR CATEGORIES
# ============================================

error_columns = {
    "Missing Value": "DQ_Missing_Value",
    "Invalid Value": "DQ_Invalid_Value",
    "Outlier": "DQ_Outlier",
    "Inconsistency": "DQ_Inconsistency",
    "Duplicate": "DQ_Duplicate"
}


# ============================================
# CALCULATE QUALITY SCORES
# ============================================

total_records = len(df)

results = []


for error_type, column in error_columns.items():

    error_count = int(df[column].sum())

    error_rate = error_count / total_records

    quality_score = (1 - error_rate) * 100

    results.append({
        "Quality_Dimension": error_type,
        "Total_Records": total_records,
        "Error_Count": error_count,
        "Error_Rate": error_rate,
        "Quality_Score": quality_score
    })


# ============================================
# OVERALL SMART HYBRID ANOMALY SCORE
# ============================================

hybrid_anomalies = int(
    df["Smart_Hybrid_Anomaly"].sum()
)

hybrid_anomaly_rate = (
    hybrid_anomalies / total_records
)

hybrid_quality_score = (
    1 - hybrid_anomaly_rate
) * 100


results.append({
    "Quality_Dimension": "Smart Hybrid Anomaly",
    "Total_Records": total_records,
    "Error_Count": hybrid_anomalies,
    "Error_Rate": hybrid_anomaly_rate,
    "Quality_Score": hybrid_quality_score
})


# ============================================
# GROUND TRUTH QUALITY SCORE
# ============================================

ground_truth_anomalies = int(
    df["Ground_Truth_Anomaly"].sum()
)

ground_truth_anomaly_rate = (
    ground_truth_anomalies / total_records
)

ground_truth_quality_score = (
    1 - ground_truth_anomaly_rate
) * 100


results.append({
    "Quality_Dimension": "Ground Truth Anomaly",
    "Total_Records": total_records,
    "Error_Count": ground_truth_anomalies,
    "Error_Rate": ground_truth_anomaly_rate,
    "Quality_Score": ground_truth_quality_score
})


# ============================================
# CREATE RESULT DATAFRAME
# ============================================

result = pd.DataFrame(results)


# ============================================
# ROUND VALUES
# ============================================

result["Error_Rate"] = result["Error_Rate"].round(4)

result["Quality_Score"] = result["Quality_Score"].round(2)


# ============================================
# DISPLAY RESULTS
# ============================================

print("\n==============================================")
print("        DATA QUALITY SCORE RESULTS")
print("==============================================")


print(
    result.to_string(index=False)
)


# ============================================
# OVERALL DATA QUALITY SCORE
# ============================================

dimension_scores = result[
    result["Quality_Dimension"].isin(
        error_columns.keys()
    )
]["Quality_Score"]


overall_quality_score = dimension_scores.mean()


print("\n==============================================")
print("       OVERALL DATA QUALITY SCORE")
print("==============================================")


print(
    f"Overall Quality Score : "
    f"{overall_quality_score:.2f} / 100"
)


# ============================================
# QUALITY INTERPRETATION
# ============================================

if overall_quality_score >= 90:

    quality_level = "Excellent"

elif overall_quality_score >= 75:

    quality_level = "Good"

elif overall_quality_score >= 50:

    quality_level = "Moderate"

else:

    quality_level = "Poor"


print(
    f"Quality Level          : {quality_level}"
)


# ============================================
# SAVE RESULTS
# ============================================

result.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nData quality score saved to:")
print(OUTPUT_FILE)


print("\n==============================================")
print("       DATA QUALITY SCORE COMPLETE")
print("==============================================")