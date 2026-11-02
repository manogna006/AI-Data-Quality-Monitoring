import os
import numpy as np
import pandas as pd


# ============================================================
# SMART HYBRID V2 - ADVANCED FEATURE ENGINEERING
# ============================================================

INPUT_FILE = "datasets/corrupted/online_retail_corrupted.csv"
OUTPUT_FILE = "results/metrics/v2/ml_features_v2.csv"


def safe_numeric(series):
    """Convert a series to numeric safely."""
    return pd.to_numeric(series, errors="coerce")


def main():

    print("=" * 60)
    print("SMART HYBRID V2 - FEATURE ENGINEERING")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load corrupted dataset
    # --------------------------------------------------------
    print("\nLoading corrupted dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns)}")

    # --------------------------------------------------------
    # 2. Create unique corrupted-row identifier
    # --------------------------------------------------------
    # Important because duplicate injection creates multiple
    # rows with the same Original_Row_ID.
    # --------------------------------------------------------

    if "Original_Row_ID" not in df.columns:
        raise ValueError(
            "Original_Row_ID not found. "
            "Please run error_injection.py first."
        )

    df["Corrupted_Row_ID"] = np.arange(len(df))

    # Shows which occurrence this is for duplicated
    # Original_Row_ID values.
    df["Original_Row_Occurrence"] = (
        df.groupby("Original_Row_ID").cumcount()
    )

    # --------------------------------------------------------
    # 3. Convert numerical columns safely
    # --------------------------------------------------------

    df["Quantity"] = safe_numeric(df["Quantity"])
    df["Price"] = safe_numeric(df["Price"])

    # --------------------------------------------------------
    # 4. Missing-value indicators
    # --------------------------------------------------------

    print("\nCreating missing-value indicators...")

    df["Missing_Invoice"] = df["Invoice"].isna().astype(int)
    df["Missing_StockCode"] = df["StockCode"].isna().astype(int)
    df["Missing_Description"] = df["Description"].isna().astype(int)
    df["Missing_Quantity"] = df["Quantity"].isna().astype(int)
    df["Missing_InvoiceDate"] = df["InvoiceDate"].isna().astype(int)
    df["Missing_Price"] = df["Price"].isna().astype(int)
    df["Missing_Customer_ID"] = df["Customer ID"].isna().astype(int)
    df["Missing_Country"] = df["Country"].isna().astype(int)

    df["Total_Missing_Count"] = (
        df[
            [
                "Missing_Invoice",
                "Missing_StockCode",
                "Missing_Description",
                "Missing_Quantity",
                "Missing_InvoiceDate",
                "Missing_Price",
                "Missing_Customer_ID",
                "Missing_Country",
            ]
        ]
        .sum(axis=1)
    )

    # --------------------------------------------------------
    # 5. Basic numerical transformations
    # --------------------------------------------------------

    print("Creating numerical features...")

    df["Abs_Quantity"] = df["Quantity"].abs()
    df["Abs_Price"] = df["Price"].abs()

    df["TotalAmount"] = df["Quantity"] * df["Price"]

    df["Abs_TotalAmount"] = df["TotalAmount"].abs()

    # Log transformations reduce the influence of extreme
    # numerical values while retaining useful information.
    df["Log_Abs_Quantity"] = np.log1p(df["Abs_Quantity"])
    df["Log_Abs_Price"] = np.log1p(df["Abs_Price"])
    df["Log_Abs_TotalAmount"] = np.log1p(df["Abs_TotalAmount"])

    # --------------------------------------------------------
    # 6. Invalid / suspicious numerical indicators
    # --------------------------------------------------------

    print("Creating numerical consistency indicators...")

    df["Negative_Quantity"] = (
        df["Quantity"] < 0
    ).astype(int)

    df["Negative_Price"] = (
        df["Price"] < 0
    ).astype(int)

    df["Zero_Quantity"] = (
        df["Quantity"] == 0
    ).astype(int)

    df["Zero_Price"] = (
        df["Price"] == 0
    ).astype(int)

    # --------------------------------------------------------
    # 7. Transaction-value features
    # --------------------------------------------------------

    df["Quantity_Price_Product"] = (
        df["Abs_Quantity"] * df["Abs_Price"]
    )

    # Price relative to quantity.
    # Small epsilon prevents division by zero.
    df["Price_Per_Unit"] = (
        df["Abs_Price"] /
        (df["Abs_Quantity"] + 1e-6)
    )

    df["Quantity_Price_Ratio"] = (
        df["Abs_Quantity"] /
        (df["Abs_Price"] + 1e-6)
    )

    # --------------------------------------------------------
    # 8. Date/time features
    # --------------------------------------------------------

    print("Creating temporal features...")

    date_series = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce"
    )

    df["Hour"] = date_series.dt.hour
    df["DayOfWeek"] = date_series.dt.dayofweek
    df["DayOfMonth"] = date_series.dt.day
    df["Month"] = date_series.dt.month
    df["Year"] = date_series.dt.year

    df["IsWeekend"] = (
        df["DayOfWeek"].isin([5, 6])
    ).astype(int)

    # Cyclical encoding
    # Helps ML models understand that 23:00 and 00:00
    # are close in time.

    df["Hour_Sin"] = np.sin(
        2 * np.pi * df["Hour"] / 24
    )

    df["Hour_Cos"] = np.cos(
        2 * np.pi * df["Hour"] / 24
    )

    df["DayOfWeek_Sin"] = np.sin(
        2 * np.pi * df["DayOfWeek"] / 7
    )

    df["DayOfWeek_Cos"] = np.cos(
        2 * np.pi * df["DayOfWeek"] / 7
    )

    df["Month_Sin"] = np.sin(
        2 * np.pi * df["Month"] / 12
    )

    df["Month_Cos"] = np.cos(
        2 * np.pi * df["Month"] / 12
    )

    # --------------------------------------------------------
    # 9. Description-based features
    # --------------------------------------------------------

    print("Creating description features...")

    description = df["Description"].fillna("").astype(str)

    df["Description_Length"] = description.str.len()

    df["Description_Word_Count"] = (
        description.str.split().str.len()
    )

    df["Description_Is_Empty"] = (
        description.str.strip() == ""
    ).astype(int)

    # Useful for detecting the injected
    # "INCORRECT_PRODUCT_DESCRIPTION".
    df["Description_Is_Suspicious"] = (
        description.str.upper()
        .str.contains("INCORRECT_PRODUCT_DESCRIPTION", na=False)
        .astype(int)
    )

    # --------------------------------------------------------
    # 10. Basic categorical presence indicators
    # --------------------------------------------------------

    df["Missing_StockCode"] = (
        df["StockCode"].isna()
    ).astype(int)

    df["Missing_Invoice"] = (
        df["Invoice"].isna()
    ).astype(int)

    df["Missing_Country"] = (
        df["Country"].isna()
    ).astype(int)

    # --------------------------------------------------------
    # 11. Clean numerical infinities / NaNs
    # --------------------------------------------------------

    numerical_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    df[numerical_columns] = df[numerical_columns].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Do NOT blindly fill everything with zero.
    # Missing indicators preserve information about missingness.
    for column in numerical_columns:
        df[column] = df[column].fillna(0)

    # --------------------------------------------------------
    # 12. Create output directory
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    # --------------------------------------------------------
    # 13. Save V2 features
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # 14. Display summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("V2 FEATURE ENGINEERING COMPLETED")
    print("=" * 60)

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nOutput:")
    print(OUTPUT_FILE)

    print("\nImportant V2 features created:")

    important_features = [
        "Missing_Customer_ID",
        "Total_Missing_Count",
        "Abs_Quantity",
        "Abs_Price",
        "TotalAmount",
        "Abs_TotalAmount",
        "Log_Abs_Quantity",
        "Log_Abs_Price",
        "Negative_Quantity",
        "Negative_Price",
        "Zero_Quantity",
        "Zero_Price",
        "Price_Per_Unit",
        "Hour",
        "DayOfWeek",
        "IsWeekend",
        "Description_Length",
        "Description_Word_Count",
        "Description_Is_Suspicious",
    ]

    for feature in important_features:
        if feature in df.columns:
            print(f"  ✓ {feature}")

    print("\nFeature engineering V2 is ready.")


if __name__ == "__main__":
    main()