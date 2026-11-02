import pandas as pd
import os

# ============================================
# EVIDENCE ANALYSIS
# ============================================

INPUT_FILE = "results/metrics/smart_hybrid_results.csv"

EVIDENCE_OUTPUT = "results/metrics/evidence_analysis.csv"
CONFIDENCE_OUTPUT = "results/metrics/confidence_analysis.csv"


print("Loading Smart Hybrid results...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df)}")


# ============================================
# CHECK REQUIRED COLUMNS
# ============================================

required_columns = [
    "Original_Row_ID",
    "DQ_Evidence",
    "ML_Agreement",
    "Ground_Truth_Anomaly",
    "Hybrid_Confidence"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    print("\nERROR: Missing required columns:")
    print(missing_columns)
    raise SystemExit


# ============================================
# EVIDENCE COMBINATION ANALYSIS
# ============================================

print("\nAnalyzing DQ Evidence and ML Agreement...")


evidence_analysis = (
    df.groupby(
        ["DQ_Evidence", "ML_Agreement"],
        as_index=False
    )
    .agg(
        Records=("Original_Row_ID", "count"),
        Actual_Anomalies=("Ground_Truth_Anomaly", "sum")
    )
)


evidence_analysis["Normal_Records"] = (
    evidence_analysis["Records"]
    - evidence_analysis["Actual_Anomalies"]
)


evidence_analysis["Anomaly_Rate"] = (
    evidence_analysis["Actual_Anomalies"]
    / evidence_analysis["Records"]
)


evidence_analysis = evidence_analysis.sort_values(
    ["DQ_Evidence", "ML_Agreement"]
)


# ============================================
# DISPLAY EVIDENCE ANALYSIS
# ============================================

print("\n==============================================")
print("          EVIDENCE ANALYSIS")
print("==============================================")

print(
    evidence_analysis.to_string(
        index=False,
        formatters={
            "Anomaly_Rate": "{:.4f}".format
        }
    )
)


# ============================================
# SAVE EVIDENCE ANALYSIS
# ============================================

evidence_analysis.to_csv(
    EVIDENCE_OUTPUT,
    index=False
)

print("\nEvidence analysis saved to:")
print(EVIDENCE_OUTPUT)


# ============================================
# CONFIDENCE ANALYSIS
# ============================================

print("\nAnalyzing Hybrid Confidence levels...")


confidence_analysis = (
    df.groupby(
        "Hybrid_Confidence",
        as_index=False
    )
    .agg(
        Records=("Original_Row_ID", "count"),
        Actual_Anomalies=("Ground_Truth_Anomaly", "sum")
    )
)


confidence_analysis["Normal_Records"] = (
    confidence_analysis["Records"]
    - confidence_analysis["Actual_Anomalies"]
)


confidence_analysis["Anomaly_Rate"] = (
    confidence_analysis["Actual_Anomalies"]
    / confidence_analysis["Records"]
)


confidence_analysis = confidence_analysis.sort_values(
    "Hybrid_Confidence"
)


# ============================================
# DISPLAY CONFIDENCE ANALYSIS
# ============================================

print("\n==============================================")
print("        HYBRID CONFIDENCE ANALYSIS")
print("==============================================")

print(
    confidence_analysis.to_string(
        index=False,
        formatters={
            "Anomaly_Rate": "{:.4f}".format
        }
    )
)


# ============================================
# SAVE CONFIDENCE ANALYSIS
# ============================================

confidence_analysis.to_csv(
    CONFIDENCE_OUTPUT,
    index=False
)

print("\nConfidence analysis saved to:")
print(CONFIDENCE_OUTPUT)


# ============================================
# THRESHOLD-BASED SUMMARY
# ============================================

print("\n==============================================")
print("       THRESHOLD EVIDENCE SUMMARY")
print("==============================================")


for threshold in sorted(df["Hybrid_Confidence"].unique()):

    subset = df[df["Hybrid_Confidence"] >= threshold]

    records = len(subset)

    anomalies = subset["Ground_Truth_Anomaly"].sum()

    anomaly_rate = anomalies / records if records > 0 else 0

    print(
        f"Confidence >= {threshold}: "
        f"Records={records}, "
        f"Actual Anomalies={anomalies}, "
        f"Anomaly Rate={anomaly_rate:.4f}"
    )


print("\n==============================================")
print("        EVIDENCE ANALYSIS COMPLETE")
print("==============================================")