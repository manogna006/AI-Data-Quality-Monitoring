import pandas as pd
import os

CORRUPTED_FILE = "datasets/corrupted/online_retail_corrupted.csv"
GROUND_TRUTH_FILE = "results/metrics/ground_truth.csv"

print("Loading corrupted dataset...")
df = pd.read_csv(CORRUPTED_FILE)

print("Loading ground truth...")
ground_truth = pd.read_csv(GROUND_TRUTH_FILE)

print("\n========================================")
print("       CORRUPTION VALIDATION")
print("========================================")

print(f"Corrupted dataset rows : {len(df)}")
print(f"Corrupted dataset cols : {len(df.columns)}")
print(f"Ground truth records   : {len(ground_truth)}")

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nGround truth error distribution:")
print(ground_truth["Error_Type"].value_counts())

print("\nGround truth columns:")
print(ground_truth.columns.tolist())

print("\nSample ground truth records:")
print(ground_truth.head(10).to_string(index=False))

print("\n========================================")
print("Validation completed.")
print("========================================")