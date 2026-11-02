import pandas as pd
import os


# ============================================
# EXPLANATION ENGINE
# ============================================

INPUT_FILE = "results/metrics/smart_hybrid_results.csv"
OUTPUT_FILE = "results/metrics/explanation_results.csv"


print("Loading Smart Hybrid results...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df)}")


# ============================================
# REMOVE OLD EXPLANATION COLUMNS IF PRESENT
# ============================================

explanation_columns = [
    "Explanation",
    "Evidence_Sources",
    "Recommended_Action"
]

for column in explanation_columns:
    if column in df.columns:
        df = df.drop(columns=[column])


# Confidence_Level already exists in Smart Hybrid results,
# so we will use that existing column instead of creating
# a duplicate column.


# ============================================
# FUNCTION: CREATE EXPLANATION
# ============================================

def generate_explanation(row):

    evidence = []
    actions = []

    # ----------------------------------------
    # DQ RULE EVIDENCE
    # ----------------------------------------

    if row["DQ_Missing_Value"] == 1:
        evidence.append("Missing Value")
        actions.append(
            "Review and fill the missing value"
        )

    if row["DQ_Invalid_Value"] == 1:
        evidence.append("Invalid Value")
        actions.append(
            "Validate the field value against the expected format or range"
        )

    if row["DQ_Outlier"] == 1:
        evidence.append("Outlier")
        actions.append(
            "Review the unusual numerical value"
        )

    if row["DQ_Inconsistency"] == 1:
        evidence.append("Inconsistency")
        actions.append(
            "Check the record for inconsistent values"
        )

    if row["DQ_Duplicate"] == 1:
        evidence.append("Duplicate")
        actions.append(
            "Check whether the duplicate record should be removed"
        )


    # ----------------------------------------
    # MACHINE LEARNING EVIDENCE
    # ----------------------------------------

    ml_agreement = int(row["ML_Agreement"])

    if ml_agreement >= 2:

        evidence.append(
            f"{ml_agreement} ML models detected an anomaly"
        )

        actions.append(
            "Review the record because multiple anomaly detection models agree"
        )

    elif ml_agreement == 1:

        evidence.append(
            "1 ML model detected an anomaly"
        )

        actions.append(
            "Perform additional review because only one ML model detected the anomaly"
        )


    # ----------------------------------------
    # CONFIDENCE
    # ----------------------------------------

    confidence = int(row["Hybrid_Confidence"])

    # Use the existing Confidence_Level from Smart Hybrid
    confidence_level = row["Confidence_Level"]


    # ----------------------------------------
    # OVERALL EXPLANATION
    # ----------------------------------------

    if confidence == 0:

        explanation = (
            "No significant data-quality or machine-learning "
            "evidence was detected."
        )

        recommended_action = (
            "No immediate action required."
        )

    else:

        if evidence:

            explanation = (
                "Anomaly detected because: "
                + "; ".join(evidence)
                + "."
            )

        else:

            explanation = (
                "Potential anomaly detected based on "
                "combined model evidence."
            )

        if actions:

            recommended_action = (
                "; ".join(actions) + "."
            )

        else:

            recommended_action = (
                "Review the record manually."
            )


    # ----------------------------------------
    # RETURN RESULTS
    # ----------------------------------------

    return pd.Series({

        "Evidence_Sources": (
            "None"
            if not evidence
            else " | ".join(evidence)
        ),

        "Explanation": explanation,

        "Recommended_Action": recommended_action
    })


# ============================================
# GENERATE EXPLANATIONS
# ============================================

print("\nGenerating explanations...")

explanations = df.apply(
    generate_explanation,
    axis=1
)


# ============================================
# COMBINE RESULTS
# ============================================

result = pd.concat(
    [df, explanations],
    axis=1
)


# ============================================
# VERIFY ROW COUNT
# ============================================

if len(result) != len(df):

    print("\nERROR: Row count changed unexpectedly.")

    raise SystemExit


# ============================================
# SAVE RESULTS
# ============================================

result.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================
# SUMMARY
# ============================================

print("\n==============================================")
print("        EXPLANATION ENGINE RESULTS")
print("==============================================")


print("\nConfidence distribution:")

print(
    result["Confidence_Level"]
    .value_counts()
)


print("\nExplanation distribution:")

print(
    result["Evidence_Sources"]
    .value_counts()
    .head(10)
)


# ============================================
# DISPLAY SAMPLE EXPLANATIONS
# ============================================

print("\n==============================================")
print("        SAMPLE EXPLANATIONS")
print("==============================================")


sample_columns = [
    "Original_Row_ID",
    "Hybrid_Confidence",
    "Confidence_Level",
    "Evidence_Sources",
    "Explanation",
    "Recommended_Action"
]


sample = result[
    result["Hybrid_Confidence"] >= 2
][sample_columns].head(10)


print(
    sample.to_string(index=False)
)


# ============================================
# SAVE MESSAGE
# ============================================

print("\nExplanation results saved to:")
print(OUTPUT_FILE)


print("\n==============================================")
print("       EXPLANATION ENGINE COMPLETE")
print("==============================================")