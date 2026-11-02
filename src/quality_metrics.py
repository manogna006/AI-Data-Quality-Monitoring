import pandas as pd
import numpy as np


# ---------------------------------------------------------
# 1. COMPLETENESS
# ---------------------------------------------------------

def calculate_completeness(df):
    """
    Measures the percentage of non-missing cells.

    100 = no missing values
    0   = all values are missing
    """

    total_cells = df.shape[0] * df.shape[1]

    if total_cells == 0:
        return 0.0

    missing_cells = df.isnull().sum().sum()

    completeness = (
        1 - (missing_cells / total_cells)
    ) * 100

    return round(completeness, 2)


# ---------------------------------------------------------
# 2. VALIDITY
# ---------------------------------------------------------

def calculate_validity(df):
    """
    Checks whether important values satisfy basic
    validity rules for the Online Retail dataset.
    """

    total_checks = 0
    invalid_checks = 0

    # Quantity must be numeric
    if "Quantity" in df.columns:

        quantity = pd.to_numeric(
            df["Quantity"],
            errors="coerce"
        )

        total_checks += len(df)

        invalid_checks += quantity.isnull().sum()

    # Price must be numeric and non-negative
    if "Price" in df.columns:

        price = pd.to_numeric(
            df["Price"],
            errors="coerce"
        )

        total_checks += len(df)

        invalid_checks += (
            price.isnull() | (price < 0)
        ).sum()

    # InvoiceDate should be a valid date
    if "InvoiceDate" in df.columns:

        invoice_date = pd.to_datetime(
            df["InvoiceDate"],
            errors="coerce"
        )

        total_checks += len(df)

        invalid_checks += invoice_date.isnull().sum()

    if total_checks == 0:
        return 0.0

    validity = (
        1 - (invalid_checks / total_checks)
    ) * 100

    return round(max(0, validity), 2)


# ---------------------------------------------------------
# 3. CONSISTENCY
# ---------------------------------------------------------

def calculate_consistency(df):
    """
    Checks logical consistency between important
    transaction fields.
    """

    checks = []

    # Quantity and Price should be numeric
    if "Quantity" in df.columns:

        quantity = pd.to_numeric(
            df["Quantity"],
            errors="coerce"
        )

        checks.append(quantity.notna())

    if "Price" in df.columns:

        price = pd.to_numeric(
            df["Price"],
            errors="coerce"
        )

        checks.append(price.notna())

    # InvoiceDate should be valid
    if "InvoiceDate" in df.columns:

        invoice_date = pd.to_datetime(
            df["InvoiceDate"],
            errors="coerce"
        )

        checks.append(invoice_date.notna())

    if not checks:
        return 0.0

    # A row is consistent if all available checks pass
    consistency_matrix = pd.concat(
        checks,
        axis=1
    )

    consistent_rows = consistency_matrix.all(axis=1)

    consistency = (
        consistent_rows.mean()
    ) * 100

    return round(consistency, 2)


# ---------------------------------------------------------
# 4. UNIQUENESS
# ---------------------------------------------------------

def calculate_uniqueness(df):
    """
    Measures the percentage of non-duplicate records.
    """

    if len(df) == 0:
        return 0.0

    duplicate_count = df.duplicated().sum()

    uniqueness = (
        1 - (duplicate_count / len(df))
    ) * 100

    return round(max(0, uniqueness), 2)


# ---------------------------------------------------------
# 5. ACCURACY
# ---------------------------------------------------------

def calculate_accuracy(df):
    """
    Estimates accuracy using basic domain validation
    rules where a reference source is unavailable.

    This is a preliminary implementation.
    """

    checks = []

    # Quantity should be numeric
    if "Quantity" in df.columns:

        quantity = pd.to_numeric(
            df["Quantity"],
            errors="coerce"
        )

        checks.append(quantity.notna())

    # Price should be numeric
    if "Price" in df.columns:

        price = pd.to_numeric(
            df["Price"],
            errors="coerce"
        )

        checks.append(price.notna())

    if not checks:
        return 0.0

    accuracy_matrix = pd.concat(
        checks,
        axis=1
    )

    accurate_rows = accuracy_matrix.all(axis=1)

    accuracy = (
        accurate_rows.mean()
    ) * 100

    return round(accuracy, 2)


# ---------------------------------------------------------
# 6. TIMELINESS
# ---------------------------------------------------------

def calculate_timeliness(df):
    """
    Measures whether transaction dates are available
    and valid.

    A more sophisticated freshness calculation will
    be added later.
    """

    if "InvoiceDate" not in df.columns:
        return 0.0

    invoice_date = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce"
    )

    valid_dates = invoice_date.notna().mean()

    timeliness = valid_dates * 100

    return round(timeliness, 2)


# ---------------------------------------------------------
# OVERALL DATA QUALITY SCORE
# ---------------------------------------------------------

def calculate_dq_score(df):

    completeness = calculate_completeness(df)
    validity = calculate_validity(df)
    consistency = calculate_consistency(df)
    uniqueness = calculate_uniqueness(df)
    accuracy = calculate_accuracy(df)
    timeliness = calculate_timeliness(df)

    overall_score = (
        completeness +
        validity +
        consistency +
        uniqueness +
        accuracy +
        timeliness
    ) / 6

    return {
        "Completeness": completeness,
        "Validity": validity,
        "Consistency": consistency,
        "Uniqueness": uniqueness,
        "Accuracy": accuracy,
        "Timeliness": timeliness,
        "Overall_DQ_Score": round(overall_score, 2)
    }


# ---------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------

if __name__ == "__main__":

    from data_loader import load_dataset

    dataset_path = "datasets/raw/online_retail_II.xlsx"

    df = load_dataset(dataset_path)

    scores = calculate_dq_score(df)

    print("\n========================================")
    print("       DATA QUALITY ASSESSMENT")
    print("========================================")

    for dimension, score in scores.items():
        print(f"{dimension:<22}: {score}")

    print("========================================")

    # Save results
    results = pd.DataFrame([scores])

    output_path = (
        "results/tables/"
        "online_retail_dq_scores.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nDQ scores saved to: {output_path}"
    )