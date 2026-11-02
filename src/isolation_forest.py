import pandas as pd
import numpy as np
import os

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "datasets/processed/ml_features.csv"
GROUND_TRUTH_FILE = "results/metrics/ground_truth.csv"
OUTPUT_FILE = "results/metrics/isolation_forest_results.csv"

os.makedirs("results/metrics", exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading ML features...")

df = pd.read_csv(INPUT_FILE)

ground_truth = pd.read_csv(GROUND_TRUTH_FILE)

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
# HANDLE NUMERICAL ISSUES
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
# ISOLATION FOREST
# ============================================================

print("\nTraining Isolation Forest...")

model = IsolationForest(
    n_estimators=100,
    contamination=0.01,
    random_state=42,
    n_jobs=-1
)

model.fit(X_scaled)


# ============================================================
# PREDICTION
# ============================================================

print("Detecting anomalies...")

predictions = model.predict(X_scaled)

anomaly_scores = model.decision_function(X_scaled)


# Isolation Forest:
#  1  = normal
# -1  = anomaly

df["IF_Prediction"] = predictions

df["IF_Anomaly"] = (
    df["IF_Prediction"] == -1
).astype(int)

# Convert score so that higher = more anomalous
df["IF_Anomaly_Score"] = -anomaly_scores


# ============================================================
# SAVE RESULTS
# ============================================================

result_columns = [
    "Original_Row_ID",
    "IF_Prediction",
    "IF_Anomaly",
    "IF_Anomaly_Score"
]

results = df[result_columns].copy()

results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

total_anomalies = results["IF_Anomaly"].sum()

print("\n========================================")
print("       ISOLATION FOREST RESULTS")
print("========================================")

print(f"Total records      : {len(results)}")
print(f"Anomalies detected  : {total_anomalies}")
print(
    f"Anomaly percentage  : "
    f"{total_anomalies / len(results) * 100:.2f}%"
)

print("\nPrediction distribution:")
print(results["IF_Prediction"].value_counts())

print("\nAnomaly score statistics:")
print(results["IF_Anomaly_Score"].describe())

print("\nResults saved to:")
print(OUTPUT_FILE)

print("========================================")