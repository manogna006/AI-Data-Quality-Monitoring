"""
V5.7 Error Explanation Module
--------------------------------
Converts V5.7 row-level predictions into human-readable
data-quality explanations.

Important:
- Does NOT use ground-truth labels for explanations.
- Does NOT use Any_Error_True.
- Uses only observable corrupted-data values and V5.7 predictions.
- Produces explanations for predicted errors.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "v5_7"
    / "v5_7_test_predictions.csv"
)

CORRUPTED_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "corrupted"
    / "online_retail_corrupted.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "v5_7"
)


# ============================================================
# CONFIGURATION
# ============================================================

ERROR_TYPES = [
    "Missing Value",
    "Invalid Value",
    "Outlier",
    "Inconsistency",
    "Duplicate",
]

PROBABILITY_COLUMNS = {
    "Missing Value": "Missing_Value_Probability",
    "Invalid Value": "Invalid_Value_Probability",
    "Outlier": "Outlier_Probability",
    "Inconsistency": "Inconsistency_Probability",
    "Duplicate": "Duplicate_Probability",
}

PREDICTION_COLUMNS = {
    "Missing Value": "Missing_Value_Prediction",
    "Invalid Value": "Invalid_Value_Prediction",
    "Outlier": "Outlier_Prediction",
    "Inconsistency": "Inconsistency_Prediction",
    "Duplicate": "Duplicate_Prediction",
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_float(value):
    """Convert a value to float safely."""
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def format_value(value, max_length=80):
    """Create readable representation of a dataset value."""
    if pd.isna(value):
        return "missing"

    if isinstance(value, float):
        if np.isfinite(value):
            if value.is_integer():
                text = str(int(value))
            else:
                text = f"{value:.4f}".rstrip("0").rstrip(".")
        else:
            text = str(value)
    else:
        text = str(value)

    if len(text) > max_length:
        text = text[:max_length - 3] + "..."

    return text


def probability_to_percent(probability):
    """Convert probability to readable percentage."""
    if pd.isna(probability):
        return "N/A"
    return f"{probability * 100:.1f}%"


def first_non_missing(series):
    """Return first non-missing value."""
    values = series.dropna()

    if len(values) == 0:
        return np.nan

    return values.iloc[0]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    print("=" * 70)
    print("V5.7 ERROR EXPLANATION MODULE")
    print("=" * 70)

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{PREDICTIONS_FILE}"
        )

    if not CORRUPTED_FILE.exists():
        raise FileNotFoundError(
            f"Corrupted dataset not found:\n{CORRUPTED_FILE}"
        )

    print("\nLoading V5.7 predictions...")
    predictions = pd.read_csv(PREDICTIONS_FILE)

    print("Loading corrupted dataset...")
    data = pd.read_csv(CORRUPTED_FILE)

    print(f"\nPrediction rows : {len(predictions):,}")
    print(f"Dataset rows    : {len(data):,}")

    return predictions, data


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(predictions, data):
    """
    Align prediction rows with the corrupted dataset.

    V5.7 predictions contain Corrupted_Row_ID.
    The corrupted dataset does not necessarily contain this column,
    so row order is used as the stable dataset-row identifier.
    """

    data = data.copy()

    # --------------------------------------------------------
    # Create Corrupted_Row_ID if not present
    # --------------------------------------------------------

    if "Corrupted_Row_ID" not in data.columns:
        data["Corrupted_Row_ID"] = np.arange(len(data), dtype=int)

    data["Corrupted_Row_ID"] = pd.to_numeric(
        data["Corrupted_Row_ID"],
        errors="coerce"
    )

    predictions["Corrupted_Row_ID"] = pd.to_numeric(
        predictions["Corrupted_Row_ID"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remove accidental duplicate IDs
    # --------------------------------------------------------

    data = data.drop_duplicates(
        subset=["Corrupted_Row_ID"],
        keep="first"
    )

    predictions = predictions.drop_duplicates(
        subset=["Corrupted_Row_ID"],
        keep="first"
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    merged = predictions.merge(
        data,
        on="Corrupted_Row_ID",
        how="left",
        suffixes=("", "_data"),
        validate="one_to_one",
    )

    print(
        f"Aligned rows   : {len(merged):,}"
    )

    missing_matches = merged["Invoice"].isna().sum() if "Invoice" in merged.columns else 0

    if missing_matches > 0:
        print(
            f"WARNING: {missing_matches:,} prediction rows "
            f"could not be matched to the corrupted dataset."
        )

    return merged


# ============================================================
# DUPLICATE EVIDENCE
# ============================================================

def calculate_duplicate_evidence(df):
    """
    Reconstruct observable exact-duplicate evidence.

    This uses the same transaction-level fields conceptually
    used by V5.7's duplicate signal.

    No ground-truth labels are used.
    """

    duplicate_columns = [
        "Invoice",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Customer ID",
        "Country",
    ]

    available = [
        col for col in duplicate_columns
        if col in df.columns
    ]

    if len(available) < 2:
        df["Explanation_Duplicate_Group_Size"] = 1
        df["Explanation_Duplicate_Other_Count"] = 0
        df["Explanation_Exact_Duplicate"] = 0
        return df

    # Create a normalized signature.
    signature_df = df[available].copy()

    for col in available:
        if pd.api.types.is_numeric_dtype(signature_df[col]):
            signature_df[col] = signature_df[col].round(8)

        signature_df[col] = (
            signature_df[col]
            .fillna("<MISSING>")
            .astype(str)
            .str.strip()
        )

    signature = pd.util.hash_pandas_object(
        signature_df,
        index=False
    )

    group_size = signature.groupby(signature).transform("size")

    df["Explanation_Duplicate_Group_Size"] = group_size.astype(int)
    df["Explanation_Duplicate_Other_Count"] = (
        group_size - 1
    ).astype(int)

    df["Explanation_Exact_Duplicate"] = (
        group_size > 1
    ).astype(int)

    return df


# ============================================================
# CONTEXTUAL INCONSISTENCY EVIDENCE
# ============================================================

def calculate_inconsistency_evidence(df):
    """
    Determine whether Description is unusual for a StockCode.

    This is used only as explanatory evidence.
    It does not use ground truth.
    """

    df["Explanation_Expected_Description"] = np.nan
    df["Explanation_Description_Mismatch"] = 0

    if "StockCode" not in df.columns or "Description" not in df.columns:
        return df

    temp = df[["StockCode", "Description"]].copy()

    # Normalize description for comparison.
    temp["_description_norm"] = (
        temp["Description"]
        .fillna("<MISSING>")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Most frequent description for each StockCode.
    mode_description = (
        temp.groupby("StockCode")["_description_norm"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    )

    df["_description_norm"] = (
        df["Description"]
        .fillna("<MISSING>")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    expected = df["StockCode"].map(mode_description)

    df["Explanation_Expected_Description"] = expected

    df["Explanation_Description_Mismatch"] = (
        expected.notna()
        & (df["_description_norm"] != expected)
    ).astype(int)

    return df


# ============================================================
# OUTLIER EVIDENCE
# ============================================================

def calculate_outlier_evidence(df):
    """
    Calculate simple observable numerical evidence for Quantity.

    This is explanatory evidence only.
    V5.7's actual model prediction remains the authoritative
    detection result.
    """

    df["Explanation_Quantity_IQR_Outlier"] = 0
    df["Explanation_Quantity_Lower_Bound"] = np.nan
    df["Explanation_Quantity_Upper_Bound"] = np.nan

    if "Quantity" not in df.columns:
        return df

    quantity = pd.to_numeric(
        df["Quantity"],
        errors="coerce"
    )

    q1 = quantity.quantile(0.25)
    q3 = quantity.quantile(0.75)

    iqr = q3 - q1

    if pd.isna(iqr) or iqr == 0:
        return df

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    df["Explanation_Quantity_Lower_Bound"] = lower
    df["Explanation_Quantity_Upper_Bound"] = upper

    df["Explanation_Quantity_IQR_Outlier"] = (
        (quantity < lower)
        | (quantity > upper)
    ).astype(int)

    return df


# ============================================================
# ERROR-SPECIFIC EXPLANATIONS
# ============================================================

def explain_missing(row):
    evidence = []

    if "Customer ID" in row.index:
        if pd.isna(row["Customer ID"]):
            evidence.append("Customer ID is missing")

    # General missing-value check.
    if len(evidence) == 0:
        missing_columns = []

        for col in [
            "Invoice",
            "StockCode",
            "Description",
            "Quantity",
            "InvoiceDate",
            "Price",
            "Customer ID",
            "Country",
        ]:
            if col in row.index and pd.isna(row[col]):
                missing_columns.append(col)

        if missing_columns:
            evidence.append(
                "Missing value detected in "
                + ", ".join(missing_columns)
            )

    if evidence:
        problem = evidence[0]
        explanation = (
            f"The record contains a missing required field: {problem}. "
            "The V5.7 missing-value model assigned a high probability "
            "to the missing-value error type."
        )
    else:
        problem = (
            "The record exhibits a strong missing-value pattern "
            "according to the V5.7 model."
        )
        explanation = (
            "The V5.7 missing-value model identified the record as "
            "suspicious based on its missingness-related feature pattern."
        )

    return problem, explanation, " | ".join(evidence)


def explain_invalid(row):
    evidence = []

    if "Price" in row.index:
        price = safe_float(row["Price"])

        if not pd.isna(price) and price <= 0:
            evidence.append(
                f"Price is non-positive ({format_value(price)})"
            )

    if "Quantity" in row.index:
        quantity = safe_float(row["Quantity"])

        if not pd.isna(quantity) and quantity == 0:
            evidence.append("Quantity is zero")

    if evidence:
        problem = evidence[0]

        explanation = (
            "The record contains a value that violates an expected "
            "transaction-value constraint. The V5.7 invalid-value "
            "model assigned a high probability to this error type."
        )
    else:
        problem = (
            "The record contains a value pattern considered invalid "
            "by the V5.7 model."
        )

        explanation = (
            "The V5.7 invalid-value model detected a strong deviation "
            "from valid-value patterns."
        )

    return problem, explanation, " | ".join(evidence)


def explain_outlier(row):
    evidence = []

    if "Quantity" in row.index:
        quantity = safe_float(row["Quantity"])

        lower = safe_float(
            row.get("Explanation_Quantity_Lower_Bound", np.nan)
        )

        upper = safe_float(
            row.get("Explanation_Quantity_Upper_Bound", np.nan)
        )

        iqr_outlier = row.get(
            "Explanation_Quantity_IQR_Outlier",
            0
        )

        if not pd.isna(quantity):

            if iqr_outlier == 1:
                evidence.append(
                    f"Quantity = {format_value(quantity)} "
                    f"is outside the observed IQR range "
                    f"[{format_value(lower)}, {format_value(upper)}]"
                )

            elif quantity > 0 and upper > 0:
                if quantity > upper:
                    evidence.append(
                        f"Quantity = {format_value(quantity)} "
                        "is unusually high"
                    )

    if evidence:
        problem = evidence[0]

        explanation = (
            "The numerical value shows an unusual deviation from "
            "the observed transaction distribution. The V5.7 "
            "outlier model assigned a high probability to the "
            "outlier error type."
        )
    else:
        problem = (
            "The record contains an unusually distributed numerical value."
        )

        explanation = (
            "The V5.7 outlier model detected a strong numerical "
            "anomaly relative to learned patterns."
        )

    return problem, explanation, " | ".join(evidence)


def explain_inconsistency(row):
    evidence = []

    stock_code = format_value(
        row.get("StockCode", np.nan)
    )

    description = format_value(
        row.get("Description", np.nan)
    )

    expected = format_value(
        row.get("Explanation_Expected_Description", np.nan)
    )

    mismatch = row.get(
        "Explanation_Description_Mismatch",
        0
    )

    if mismatch == 1:
        evidence.append(
            f"Description '{description}' does not match "
            f"the commonly observed description '{expected}' "
            f"for StockCode {stock_code}"
        )

    if evidence:
        problem = evidence[0]

        explanation = (
            "The relationship between related fields is inconsistent. "
            "The observed Description differs from the dominant "
            "StockCode–Description relationship in the dataset. "
            "The V5.7 inconsistency model assigned a high probability "
            "to the inconsistency error type."
        )
    else:
        problem = (
            "The record contains an unusual relationship between "
            "related categorical fields."
        )

        explanation = (
            "The V5.7 inconsistency model detected a strong deviation "
            "from learned contextual relationships."
        )

    return problem, explanation, " | ".join(evidence)


def explain_duplicate(row):
    evidence = []

    group_size = row.get(
        "Explanation_Duplicate_Group_Size",
        1
    )

    other_count = row.get(
        "Explanation_Duplicate_Other_Count",
        0
    )

    exact_duplicate = row.get(
        "Explanation_Exact_Duplicate",
        0
    )

    if exact_duplicate == 1:
        evidence.append(
            f"Exact transaction signature occurs "
            f"{int(group_size)} times"
        )

        evidence.append(
            f"{int(other_count)} other matching record(s) found"
        )

    if evidence:
        problem = (
            f"Transaction appears duplicated "
            f"({int(group_size)} occurrences)"
        )

        explanation = (
            "The complete transaction signature matches another "
            "record in the dataset. This provides direct "
            "unsupervised evidence of a potential duplicate. "
            "The V5.7 duplicate model assigned a high probability "
            "to the duplicate error type."
        )
    else:
        problem = (
            "The record exhibits a strong duplicate-like pattern."
        )

        explanation = (
            "The V5.7 duplicate model detected a strong similarity "
            "to duplicate transaction patterns."
        )

    return problem, explanation, " | ".join(evidence)


# ============================================================
# BUILD EXPLANATION
# ============================================================

def generate_explanation(row):
    """
    Select the predicted error type with the highest probability.

    A row may receive multiple type-specific positive predictions.
    We preserve all predicted types while selecting the strongest
    error type as the primary explanation.
    """

    predicted_types = []

    for error_type in ERROR_TYPES:

        prediction_column = PREDICTION_COLUMNS[error_type]

        if (
            prediction_column in row.index
            and row[prediction_column] == 1
        ):
            predicted_types.append(error_type)

    if len(predicted_types) == 0:
        return {
            "Primary_Error_Type": "No Error",
            "Predicted_Error_Types": "",
            "Primary_Confidence": 0.0,
            "Problem": "No data-quality error predicted.",
            "Explanation": "The V5.7 system did not flag this record.",
            "Evidence": "",
        }

    # Highest probability becomes primary type.
    primary_type = max(
        predicted_types,
        key=lambda x: safe_float(
            row[PROBABILITY_COLUMNS[x]]
        )
    )

    primary_probability = safe_float(
        row[PROBABILITY_COLUMNS[primary_type]]
    )

    # --------------------------------------------------------
    # Type-specific explanation
    # --------------------------------------------------------

    if primary_type == "Missing Value":
        problem, explanation, evidence = explain_missing(row)

    elif primary_type == "Invalid Value":
        problem, explanation, evidence = explain_invalid(row)

    elif primary_type == "Outlier":
        problem, explanation, evidence = explain_outlier(row)

    elif primary_type == "Inconsistency":
        problem, explanation, evidence = explain_inconsistency(row)

    elif primary_type == "Duplicate":
        problem, explanation, evidence = explain_duplicate(row)

    else:
        problem = "Potential data-quality issue detected."
        explanation = (
            "The V5.7 hybrid system assigned a high anomaly "
            "probability to this record."
        )
        evidence = ""

    return {
        "Primary_Error_Type": primary_type,
        "Predicted_Error_Types": "; ".join(predicted_types),
        "Primary_Confidence": primary_probability,
        "Problem": problem,
        "Explanation": explanation,
        "Evidence": evidence,
    }


# ============================================================
# BUILD OUTPUT
# ============================================================

def build_explanation_table(df):
    print("\nGenerating explanations...")

    explanations = []

    total_predicted = int(
        df["Any_Error_Prediction"].sum()
    )

    print(
        f"Predicted error rows: {total_predicted:,}"
    )

    for _, row in df.iterrows():

        if row["Any_Error_Prediction"] != 1:
            continue

        result = generate_explanation(row)

        output = {
            "Corrupted_Row_ID": row["Corrupted_Row_ID"],
            "Original_Row_ID": row.get(
                "Original_Row_ID",
                np.nan
            ),

            "Primary_Error_Type": result["Primary_Error_Type"],
            "Predicted_Error_Types": result[
                "Predicted_Error_Types"
            ],

            "Primary_Confidence": result[
                "Primary_Confidence"
            ],

            "Problem": result["Problem"],
            "Explanation": result["Explanation"],
            "Evidence": result["Evidence"],
        }

        # ----------------------------------------------------
        # Original transaction values
        # ----------------------------------------------------

        for column in [
            "Invoice",
            "StockCode",
            "Description",
            "Quantity",
            "InvoiceDate",
            "Price",
            "Customer ID",
            "Country",
        ]:
            if column in row.index:
                output[column] = row[column]

        # ----------------------------------------------------
        # Type-specific probabilities
        # ----------------------------------------------------

        for error_type in ERROR_TYPES:
            probability_column = PROBABILITY_COLUMNS[
                error_type
            ]

            if probability_column in row.index:
                output[probability_column] = row[
                    probability_column
                ]

        explanations.append(output)

    explanation_df = pd.DataFrame(explanations)

    return explanation_df


# ============================================================
# SAVE SUMMARY
# ============================================================

def save_summary(explanation_df, predictions):
    summary_rows = []

    for error_type in ERROR_TYPES:

        prediction_column = PREDICTION_COLUMNS[
            error_type
        ]

        probability_column = PROBABILITY_COLUMNS[
            error_type
        ]

        count = int(
            predictions[prediction_column].sum()
        )

        if count > 0:
            mean_confidence = (
                predictions.loc[
                    predictions[prediction_column] == 1,
                    probability_column
                ].mean()
            )
        else:
            mean_confidence = 0.0

        summary_rows.append({
            "Error_Type": error_type,
            "Predicted_Rows": count,
            "Mean_Confidence": mean_confidence,
        })

    summary = pd.DataFrame(summary_rows)

    summary_file = (
        OUTPUT_DIR
        / "v5_7_error_explanation_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    return summary


# ============================================================
# PRINT REPORT
# ============================================================

def print_report(explanation_df, summary):
    print("\n")
    print("=" * 70)
    print("V5.7 ERROR EXPLANATION SUMMARY")
    print("=" * 70)

    print(
        f"\nTotal predicted error rows: "
        f"{len(explanation_df):,}"
    )

    print("\nPredictions by error type:")
    print("-" * 70)

    for _, row in summary.iterrows():

        print(
            f"{row['Error_Type']:<18} "
            f"{int(row['Predicted_Rows']):>8,} rows   "
            f"Mean confidence: "
            f"{row['Mean_Confidence'] * 100:>6.2f}%"
        )

    print("\n")
    print("=" * 70)
    print("SAMPLE ERROR EXPLANATIONS")
    print("=" * 70)

    if len(explanation_df) == 0:
        print("\nNo predicted errors found.")
        return

    sample = explanation_df.head(10)

    for _, row in sample.iterrows():

        print("\n" + "-" * 70)

        print(
            f"Row ID       : "
            f"{int(row['Corrupted_Row_ID'])}"
        )

        print(
            f"Error Type   : "
            f"{row['Primary_Error_Type']}"
        )

        print(
            f"Confidence   : "
            f"{probability_to_percent(row['Primary_Confidence'])}"
        )

        print(
            f"Problem      : "
            f"{row['Problem']}"
        )

        print(
            f"Explanation  : "
            f"{row['Explanation']}"
        )

        if str(row["Evidence"]).strip():
            print(
                f"Evidence     : "
                f"{row['Evidence']}"
            )

    print("\n")


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    predictions, data = load_data()

    # --------------------------------------------------------
    # Prepare and align
    # --------------------------------------------------------

    merged = prepare_data(
        predictions,
        data
    )

    # --------------------------------------------------------
    # Calculate observable evidence
    # --------------------------------------------------------

    print("\nCalculating duplicate evidence...")
    merged = calculate_duplicate_evidence(merged)

    print("Calculating contextual inconsistency evidence...")
    merged = calculate_inconsistency_evidence(merged)

    print("Calculating numerical outlier evidence...")
    merged = calculate_outlier_evidence(merged)

    # --------------------------------------------------------
    # Generate explanations
    # --------------------------------------------------------

    explanation_df = build_explanation_table(
        merged
    )

    # --------------------------------------------------------
    # Save detailed explanations
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    explanation_file = (
        OUTPUT_DIR
        / "v5_7_error_explanations.csv"
    )

    explanation_df.to_csv(
        explanation_file,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = save_summary(
        explanation_df,
        predictions
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_report(
        explanation_df,
        summary
    )

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    print("=" * 70)
    print("FILES CREATED")
    print("=" * 70)

    print(
        f"\nDetailed explanations:\n"
        f"{explanation_file}"
    )

    print(
        f"\nSummary:\n"
        f"{OUTPUT_DIR / 'v5_7_error_explanation_summary.csv'}"
    )

    print("\nERROR EXPLANATION MODULE COMPLETED SUCCESSFULLY.")
    print("=" * 70)


if __name__ == "__main__":
    main()