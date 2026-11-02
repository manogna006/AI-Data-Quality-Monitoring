import pandas as pd
import numpy as np
import os

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "datasets/processed/ml_features.csv"
OUTPUT_FILE = "results/metrics/one_class_svm_results.csv"

os.makedirs("results/metrics", exist_ok=True)

RANDOM_STATE = 42

# Use a manageable training sample
TRAIN_SAMPLE_SIZE = 80000


# ============================================================
# LOAD DATA
# ============================================================

print("Loading ML features...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df)}")


# ============================================================
# DEFINE FEATURES
# ============================================================

feature_columns = [
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

X = df[feature_columns].copy()


# ============================================================
# CLEAN FEATURES
# ============================================================

print("\nPreparing features...")

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)


# ============================================================
# SCALE FEATURES
# ============================================================

print("Scaling features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ============================================================
# CREATE TRAINING SAMPLE
# ============================================================

print("\nCreating training sample...")

rng = np.random.default_rng(RANDOM_STATE)

sample_size = min(
    TRAIN_SAMPLE_SIZE,
    len(X_scaled)
)

sample_indices = rng.choice(
    len(X_scaled),
    size=sample_size,
    replace=False
)

X_train = X_scaled[sample_indices]

print(f"Training sample size: {len(X_train)}")


# ============================================================
# ONE-CLASS SVM
# ============================================================

print("\nTraining One-Class SVM...")

model = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.01
)

model.fit(X_train)


# ============================================================
# PREDICTION ON FULL DATASET
# ============================================================

print("Detecting anomalies on full dataset...")

predictions = model.predict(X_scaled)

decision_scores = model.decision_function(X_scaled)


# ============================================================
# STORE RESULTS
# ============================================================

df["SVM_Prediction"] = predictions

df["SVM_Anomaly"] = (
    df["SVM_Prediction"] == -1
).astype(int)

# Higher value = more anomalous
df["SVM_Anomaly_Score"] = -decision_scores


result_columns = [
    "Original_Row_ID",
    "SVM_Prediction",
    "SVM_Anomaly",
    "SVM_Anomaly_Score"
]

results = df[result_columns].copy()


# ============================================================
# SAVE RESULTS
# ============================================================

results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

total_anomalies = results["SVM_Anomaly"].sum()

print("\n========================================")
print("        ONE-CLASS SVM RESULTS")
print("========================================")

print(f"Total records       : {len(results)}")
print(f"Training sample     : {len(X_train)}")
print(f"Anomalies detected  : {total_anomalies}")

print(
    f"Anomaly percentage  : "
    f"{total_anomalies / len(results) * 100:.2f}%"
)

print("\nPrediction distribution:")
print(results["SVM_Prediction"].value_counts())

print("\nAnomaly score statistics:")
print(results["SVM_Anomaly_Score"].describe())

print("\nResults saved to:")
print(OUTPUT_FILE)

print("========================================")