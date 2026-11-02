import pandas as pd
import numpy as np
import os

from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "datasets/processed/ml_features.csv"
OUTPUT_FILE = "results/metrics/lof_results.csv"

os.makedirs("results/metrics", exist_ok=True)


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
# LOCAL OUTLIER FACTOR
# ============================================================

print("\nTraining Local Outlier Factor...")

model = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.01,
    novelty=False,
    n_jobs=-1
)


# ============================================================
# PREDICTION
# ============================================================

print("Detecting anomalies...")

predictions = model.fit_predict(X_scaled)

# LOF negative_outlier_factor_:
# normal points are closer to -1
# anomalies have more negative values

lof_scores = -model.negative_outlier_factor_


# ============================================================
# STORE RESULTS
# ============================================================

df["LOF_Prediction"] = predictions

df["LOF_Anomaly"] = (
    df["LOF_Prediction"] == -1
).astype(int)

df["LOF_Anomaly_Score"] = lof_scores


result_columns = [
    "Original_Row_ID",
    "LOF_Prediction",
    "LOF_Anomaly",
    "LOF_Anomaly_Score"
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

total_anomalies = results["LOF_Anomaly"].sum()

print("\n========================================")
print("             LOF RESULTS")
print("========================================")

print(f"Total records      : {len(results)}")
print(f"Anomalies detected : {total_anomalies}")

print(
    f"Anomaly percentage : "
    f"{total_anomalies / len(results) * 100:.2f}%"
)

print("\nPrediction distribution:")
print(results["LOF_Prediction"].value_counts())

print("\nAnomaly score statistics:")
print(results["LOF_Anomaly_Score"].describe())

print("\nResults saved to:")
print(OUTPUT_FILE)

print("========================================")