import pandas as pd
import numpy as np
import os

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "datasets/corrupted/online_retail_corrupted.csv"
OUTPUT_FILE = "datasets/processed/ml_features.csv"

os.makedirs("datasets/processed", exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading corrupted dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df)}")


# ============================================================
# DATE CONVERSION
# ============================================================

print("\nProcessing dates...")

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    errors="coerce"
)


# ============================================================
# BASIC TRANSACTION FEATURES
# ============================================================

print("Creating transaction features...")

# Total transaction value
df["TotalAmount"] = df["Quantity"] * df["Price"]

# Date/time features
df["Hour"] = df["InvoiceDate"].dt.hour
df["DayOfWeek"] = df["InvoiceDate"].dt.dayofweek
df["Month"] = df["InvoiceDate"].dt.month
df["Year"] = df["InvoiceDate"].dt.year


# ============================================================
# CUSTOMER-LEVEL FEATURES
# ============================================================

print("Creating customer-level features...")

customer_transaction_count = (
    df.groupby("Customer ID")["Invoice"]
    .transform("count")
)

df["CustomerTransactionCount"] = (
    customer_transaction_count
)


# ============================================================
# PRODUCT-LEVEL FEATURES
# ============================================================

print("Creating product-level features...")

product_transaction_count = (
    df.groupby("StockCode")["Invoice"]
    .transform("count")
)

df["ProductTransactionCount"] = (
    product_transaction_count
)


# ============================================================
# INVOICE-LEVEL FEATURES
# ============================================================

print("Creating invoice-level features...")

invoice_quantity = (
    df.groupby("Invoice")["Quantity"]
    .transform("sum")
)

df["InvoiceTotalQuantity"] = invoice_quantity


invoice_amount = (
    df.groupby("Invoice")["TotalAmount"]
    .transform("sum")
)

df["InvoiceTotalAmount"] = invoice_amount


# ============================================================
# SELECT ML FEATURES
# ============================================================

feature_columns = [
    "Original_Row_ID",
    "Quantity",
    "Price",
    "TotalAmount",
    "CustomerTransactionCount",
    "ProductTransactionCount",
    "InvoiceTotalQuantity",
    "InvoiceTotalAmount",
    "Hour",
    "DayOfWeek",
    "Month",
    "Year"
]

ml_df = df[feature_columns].copy()


# ============================================================
# HANDLE MISSING / INFINITE VALUES
# ============================================================

print("Cleaning feature values...")

ml_df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

# Numerical features required by ML algorithms
numeric_columns = [
    col for col in feature_columns
    if col != "Original_Row_ID"
]

ml_df[numeric_columns] = (
    ml_df[numeric_columns]
    .fillna(0)
)


# ============================================================
# SAVE FEATURES
# ============================================================

ml_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("       FEATURE ENGINEERING COMPLETE")
print("========================================")

print(f"Rows          : {len(ml_df)}")
print(f"Features      : {len(ml_df.columns)}")

print("\nFeatures:")
for column in ml_df.columns:
    print(f"- {column}")

print(f"\nSaved to:")
print(OUTPUT_FILE)

print("========================================")