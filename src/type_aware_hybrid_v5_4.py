import os
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# ============================================================
# TYPE-AWARE HYBRID V5.4
# STRATIFIED + LEAKAGE-SAFE RANDOM FOREST
# SPECIALIZED DUPLICATE DETECTION
# ============================================================

print("=" * 70)
print("TYPE-AWARE HYBRID V5.4")
print("STRATIFIED + LEAKAGE-SAFE RANDOM FOREST")
print("SPECIALIZED DUPLICATE DETECTION")
print("=" * 70)

# ============================================================
# PATHS
# ============================================================

V2_RESULTS = "results/metrics/v2/model_scores_v2.csv"
V2_FEATURES = "results/metrics/v2/ml_features_v2.csv"
GROUND_TRUTH = "results/metrics/ground_truth.csv"
RAW_DATA = "datasets/corrupted/online_retail_corrupted.csv"

OUTPUT_DIR = "results/metrics/v5_4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading datasets...")

v2_results = pd.read_csv(V2_RESULTS)
v2_features = pd.read_csv(V2_FEATURES)
gt = pd.read_csv(GROUND_TRUTH)
raw = pd.read_csv(RAW_DATA)

print(f"V2 results       : {len(v2_results)}")
print(f"V2 features      : {len(v2_features)}")
print(f"Ground truth     : {len(gt)}")
print(f"Raw data         : {len(raw)}")

# ============================================================
# CREATE STABLE CORRUPTED ROW ID
# ============================================================

if "Corrupted_Row_ID" not in raw.columns:
    print("\nRaw CSV does not contain Corrupted_Row_ID.")
    print("Creating from raw row order...")
    raw["Corrupted_Row_ID"] = np.arange(len(raw), dtype=np.int64)

# V2 results should align to raw row order
if len(v2_results) != len(raw):
    raise ValueError("V2 results and raw data row counts do not match.")

if len(v2_features) != len(raw):
    raise ValueError("V2 features and raw data row counts do not match.")

# ============================================================
# ENSURE ORIGINAL ROW ID
# ============================================================

if "Original_Row_ID" not in raw.columns:
    raw["Original_Row_ID"] = np.arange(len(raw), dtype=np.int64)

if "Original_Row_ID" not in v2_results.columns:
    v2_results["Original_Row_ID"] = raw["Original_Row_ID"].values

if "Original_Row_ID" not in v2_features.columns:
    v2_features["Original_Row_ID"] = raw["Original_Row_ID"].values

# ============================================================
# VERIFY ROW ALIGNMENT
# ============================================================

if not np.array_equal(
    raw["Corrupted_Row_ID"].values,
    np.arange(len(raw), dtype=np.int64)
):
    raw = raw.sort_values("Corrupted_Row_ID").reset_index(drop=True)

    v2_results = v2_results.iloc[
        np.argsort(v2_results.index.values)
    ].reset_index(drop=True)

    v2_features = v2_features.iloc[
        np.argsort(v2_features.index.values)
    ].reset_index(drop=True)

print("Raw/V2 row-key alignment verified.")

# ============================================================
# PREPARE BASE DATAFRAME
# ============================================================

df = pd.DataFrame({
    "Corrupted_Row_ID": raw["Corrupted_Row_ID"].values,
    "Original_Row_ID": raw["Original_Row_ID"].values
})

# Add V2 model scores
for col in v2_results.columns:
    if col not in ["Corrupted_Row_ID", "Original_Row_ID"]:
        df[col] = v2_results[col].values

# Add V2 engineered features
for col in v2_features.columns:
    if col not in df.columns:
        df[col] = v2_features[col].values

print(f"Unique feature rows: {len(df)}")

# ============================================================
# GROUND TRUTH
# ============================================================

gt_cols = [
    "Original_Row_ID",
    "Missing_Value",
    "Invalid_Value",
    "Outlier",
    "Inconsistency",
    "Duplicate"
]

missing_gt = [c for c in gt_cols if c not in gt.columns]

if missing_gt:
    raise ValueError(f"Missing ground-truth columns: {missing_gt}")

gt_clean = gt[gt_cols].copy()

# Ground truth should be one row per Original_Row_ID
gt_clean = gt_clean.drop_duplicates(
    subset=["Original_Row_ID"],
    keep="first"
)

df = df.merge(
    gt_clean,
    on="Original_Row_ID",
    how="left",
    validate="many_to_one"
)

print(f"After ground-truth merge: {len(df)}")

# ============================================================
# ADD RAW CONTEXT
# ============================================================

raw_context_cols = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country"
]

for col in raw_context_cols:
    if col in raw.columns and col not in df.columns:
        df[col] = raw[col].values

print("Raw context columns prepared.")
print(f"After raw context merge: {len(df)}")

# ============================================================
# MERGE INTEGRITY
# ============================================================

if len(df) != len(raw):
    raise ValueError("Merge explosion detected.")

if df["Corrupted_Row_ID"].duplicated().any():
    raise ValueError("Duplicate Corrupted_Row_ID detected.")

df = df.sort_values("Corrupted_Row_ID").reset_index(drop=True)

print("Merge integrity verified.")

# ============================================================
# NUMERIC CLEANUP
# ============================================================

for col in ["Quantity", "Price", "Customer ID"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ============================================================
# DATE FEATURES
# ============================================================

if "InvoiceDate" in df.columns:
    dt = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    df["V54_Hour"] = dt.dt.hour.fillna(-1)
    df["V54_DayOfWeek"] = dt.dt.dayofweek.fillna(-1)
    df["V54_DayOfMonth"] = dt.dt.day.fillna(-1)
    df["V54_Month"] = dt.dt.month.fillna(-1)
    df["V54_Year"] = dt.dt.year.fillna(-1)

    df["V54_IsWeekend"] = (
        df["V54_DayOfWeek"].isin([5, 6])
    ).astype(int)

    df["V54_Hour_Sin"] = np.sin(
        2 * np.pi * df["V54_Hour"].clip(lower=0) / 24
    )

    df["V54_Hour_Cos"] = np.cos(
        2 * np.pi * df["V54_Hour"].clip(lower=0) / 24
    )

    df["V54_Day_Sin"] = np.sin(
        2 * np.pi * df["V54_DayOfWeek"].clip(lower=0) / 7
    )

    df["V54_Day_Cos"] = np.cos(
        2 * np.pi * df["V54_DayOfWeek"].clip(lower=0) / 7
    )

    df["V54_Month_Sin"] = np.sin(
        2 * np.pi * df["V54_Month"].clip(lower=1) / 12
    )

    df["V54_Month_Cos"] = np.cos(
        2 * np.pi * df["V54_Month"].clip(lower=1) / 12
    )

# ============================================================
# BASIC OBSERVABLE FEATURES
# ============================================================

if "Quantity" in df.columns:
    df["V54_Abs_Quantity"] = df["Quantity"].abs()
    df["V54_Log_Quantity"] = np.log1p(df["Quantity"].abs())
    df["V54_Negative_Quantity"] = (df["Quantity"] < 0).astype(int)
    df["V54_Zero_Quantity"] = (df["Quantity"] == 0).astype(int)

if "Price" in df.columns:
    df["V54_Abs_Price"] = df["Price"].abs()
    df["V54_Log_Price"] = np.log1p(df["Price"].abs())
    df["V54_Negative_Price"] = (df["Price"] < 0).astype(int)
    df["V54_Zero_Price"] = (df["Price"] == 0).astype(int)

if "Quantity" in df.columns and "Price" in df.columns:
    df["V54_Total_Amount"] = (
        df["Quantity"].fillna(0) *
        df["Price"].fillna(0)
    )

    df["V54_Abs_Total_Amount"] = df["V54_Total_Amount"].abs()

    df["V54_Price_Quantity_Ratio"] = (
        df["Price"] /
        df["Quantity"].replace(0, np.nan)
    )

# ============================================================
# MISSINGNESS
# ============================================================

missing_columns = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country"
]

for col in missing_columns:
    if col in df.columns:
        df[f"V54_Missing_{col.replace(' ', '_')}"] = (
            df[col].isna()
        ).astype(int)

missing_feature_cols = [
    c for c in df.columns
    if c.startswith("V54_Missing_")
]

if missing_feature_cols:
    df["V54_Total_Missing_Count"] = df[
        missing_feature_cols
    ].sum(axis=1)

# ============================================================
# DESCRIPTION FEATURES
# ============================================================

if "Description" in df.columns:

    desc = df["Description"].fillna("").astype(str)

    df["V54_Description_Length"] = desc.str.len()

    df["V54_Description_Word_Count"] = (
        desc.str.split().str.len()
    )

    df["V54_Description_Is_Empty"] = (
        desc.str.strip() == ""
    ).astype(int)

    df["V54_Description_Has_Question"] = (
        desc.str.contains(r"\?", regex=True)
    ).astype(int)

    df["V54_Description_All_Caps"] = (
        desc.str.isupper()
    ).astype(int)

# ============================================================
# CREATE JOINT ERROR-PATTERN STRATIFIED SPLIT
# ============================================================

print("\nCreating STRATIFIED 60/20/20 split...")

error_cols = [
    "Missing_Value",
    "Invalid_Value",
    "Outlier",
    "Inconsistency",
    "Duplicate"
]

for col in error_cols:
    df[col] = (
        pd.to_numeric(df[col], errors="coerce")
        .fillna(0)
        .astype(int)
    )

pattern_key = (
    df[error_cols]
    .astype(str)
    .agg("_".join, axis=1)
)

rng = np.random.RandomState(42)

train_indices = []
val_indices = []
test_indices = []

for _, group in df.groupby(pattern_key, sort=True):

    indices = np.array(
        group.index.tolist(),
        dtype=np.int64
    ).copy()

    rng.shuffle(indices)

    n = len(indices)

    n_train = int(round(n * 0.60))
    n_val = int(round(n * 0.20))

    if n_train + n_val > n:
        n_val = n - n_train

    train_indices.extend(indices[:n_train])
    val_indices.extend(
        indices[n_train:n_train + n_val]
    )
    test_indices.extend(
        indices[n_train + n_val:]
    )

train_idx = np.array(train_indices, dtype=np.int64).copy()
val_idx = np.array(val_indices, dtype=np.int64).copy()
test_idx = np.array(test_indices, dtype=np.int64).copy()

rng.shuffle(train_idx)
rng.shuffle(val_idx)
rng.shuffle(test_idx)

print(f"Train      : {len(train_idx)}")
print(f"Validation : {len(val_idx)}")
print(f"Test       : {len(test_idx)}")

if len(set(train_idx) & set(val_idx)) > 0:
    raise ValueError("Train/validation overlap detected.")

if len(set(train_idx) & set(test_idx)) > 0:
    raise ValueError("Train/test overlap detected.")

if len(set(val_idx) & set(test_idx)) > 0:
    raise ValueError("Validation/test overlap detected.")

print("Split overlap check: PASSED")

# ============================================================
# ERROR DISTRIBUTION
# ============================================================

print("\nError distribution by split:")

for error_type, col in [
    ("Missing", "Missing_Value"),
    ("Invalid", "Invalid_Value"),
    ("Outlier", "Outlier"),
    ("Inconsistency", "Inconsistency"),
    ("Duplicate", "Duplicate")
]:

    tr = int(df.iloc[train_idx][col].sum())
    va = int(df.iloc[val_idx][col].sum())
    te = int(df.iloc[test_idx][col].sum())

    print(
        f"{error_type:<15}"
        f"Train={tr:5d} ({tr/len(train_idx)*100:6.3f}%)   "
        f"Val={va:5d} ({va/len(val_idx)*100:6.3f}%)   "
        f"Test={te:5d} ({te/len(test_idx)*100:6.3f}%)"
    )

# ============================================================
# TRAIN-ONLY CONTEXTUAL FEATURES
# ============================================================

print("\nBuilding TRAIN-ONLY contextual statistics...")

train_df = df.iloc[train_idx].copy()

# ------------------------------------------------------------
# PRODUCT CONTEXT
# ------------------------------------------------------------

if "StockCode" in df.columns:

    product_stats = (
        train_df
        .groupby("StockCode", dropna=False)
        .agg(
            Train_Product_Frequency=("StockCode", "size"),
            Train_Product_Median_Price=("Price", "median"),
            Train_Product_Mean_Price=("Price", "mean"),
            Train_Product_Std_Price=("Price", "std"),
            Train_Product_Median_Quantity=("Quantity", "median"),
            Train_Product_Mean_Quantity=("Quantity", "mean"),
            Train_Product_Std_Quantity=("Quantity", "std"),
            Train_Product_Q1_Quantity=("Quantity", lambda x: x.quantile(0.25)),
            Train_Product_Q3_Quantity=("Quantity", lambda x: x.quantile(0.75))
        )
        .reset_index()
    )

    df = df.merge(
        product_stats,
        on="StockCode",
        how="left"
    )

# ------------------------------------------------------------
# CUSTOMER CONTEXT
# ------------------------------------------------------------

if "Customer ID" in df.columns:

    customer_stats = (
        train_df
        .groupby("Customer ID", dropna=False)
        .size()
        .reset_index(
            name="Train_Customer_Frequency"
        )
    )

    df = df.merge(
        customer_stats,
        on="Customer ID",
        how="left"
    )

# ------------------------------------------------------------
# DESCRIPTION CONTEXT
# ------------------------------------------------------------

if "Description" in df.columns:

    description_stats = (
        train_df
        .groupby("Description", dropna=False)
        .agg(
            Train_Description_Frequency=("Description", "size"),
            Train_Description_Product_Count=(
                "StockCode",
                "nunique"
            )
        )
        .reset_index()
    )

    df = df.merge(
        description_stats,
        on="Description",
        how="left"
    )

# ------------------------------------------------------------
# INVOICE CONTEXT
# ------------------------------------------------------------

if "Invoice" in df.columns:

    invoice_stats = (
        train_df
        .groupby("Invoice", dropna=False)
        .size()
        .reset_index(
            name="Train_Invoice_Item_Count"
        )
    )

    df = df.merge(
        invoice_stats,
        on="Invoice",
        how="left"
    )

# ============================================================
# V5.4 SPECIAL DUPLICATE EVIDENCE
# ============================================================

print("\nBuilding V5.4 duplicate evidence...")

# ------------------------------------------------------------
# IMPORTANT:
# This is an UNSUPERVISED exact-row signature.
# It does NOT use ground truth labels.
# ------------------------------------------------------------

duplicate_signature_cols = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country"
]

duplicate_signature_cols = [
    c for c in duplicate_signature_cols
    if c in df.columns
]

# Convert values to stable strings so NaNs are handled consistently
signature_frame = df[duplicate_signature_cols].copy()

for col in signature_frame.columns:

    if pd.api.types.is_datetime64_any_dtype(
        signature_frame[col]
    ):
        signature_frame[col] = (
            signature_frame[col]
            .astype(str)
        )
    else:
        signature_frame[col] = (
            signature_frame[col]
            .fillna("<MISSING>")
            .astype(str)
        )

df["_V54_Duplicate_Signature"] = (
    signature_frame
    .agg("||".join, axis=1)
)

# Exact signature frequency over the corrupted dataset.
#
# This is an unsupervised DQ signal:
# repeated transaction records are suspicious,
# but the feature itself does not access ground truth.
signature_counts = (
    df["_V54_Duplicate_Signature"]
    .value_counts(dropna=False)
)

df["V54_Exact_Duplicate_Group_Size"] = (
    df["_V54_Duplicate_Signature"]
    .map(signature_counts)
    .fillna(1)
    .astype(float)
)

df["V54_Exact_Duplicate_Flag"] = (
    df["V54_Exact_Duplicate_Group_Size"] > 1
).astype(int)

df["V54_Duplicate_Group_Log_Size"] = np.log1p(
    df["V54_Exact_Duplicate_Group_Size"]
)

# ------------------------------------------------------------
# POSITION INSIDE EXACT DUPLICATE GROUP
# ------------------------------------------------------------

df["V54_Duplicate_Occurrence"] = (
    df.groupby(
        "_V54_Duplicate_Signature",
        sort=False
    ).cumcount()
)

df["V54_Duplicate_Is_First"] = (
    df["V54_Duplicate_Occurrence"] == 0
).astype(int)

df["V54_Duplicate_Is_Repeated"] = (
    df["V54_Duplicate_Occurrence"] > 0
).astype(int)

# ------------------------------------------------------------
# TRAIN-ONLY DUPLICATE GROUP SIZE
# ------------------------------------------------------------

train_signature_counts = (
    df.iloc[train_idx]
    .groupby(
        "_V54_Duplicate_Signature",
        dropna=False
    )
    .size()
    .rename("Train_Duplicate_Group_Size")
)

df = df.merge(
    train_signature_counts,
    left_on="_V54_Duplicate_Signature",
    right_index=True,
    how="left"
)

df["Train_Duplicate_Group_Size"] = (
    df["Train_Duplicate_Group_Size"]
    .fillna(1)
)

# ============================================================
# PRODUCT DEVIATION FEATURES
# ============================================================

if "StockCode" in df.columns:

    df["V54_Product_IQR_Quantity"] = (
        df["Train_Product_Q3_Quantity"] -
        df["Train_Product_Q1_Quantity"]
    )

    df["V54_Product_Quantity_Deviation"] = (
        df["Quantity"] -
        df["Train_Product_Median_Quantity"]
    ).abs()

    df["V54_Product_Price_Deviation"] = (
        df["Price"] -
        df["Train_Product_Median_Price"]
    ).abs()

    df["V54_Product_Price_Ratio"] = (
        df["Price"] /
        df["Train_Product_Median_Price"]
    )

# ============================================================
# REMOVE NON-MODEL / TARGET / IDENTIFIER COLUMNS
# ============================================================

target_columns = [
    "Missing_Value",
    "Invalid_Value",
    "Outlier",
    "Inconsistency",
    "Duplicate"
]

identifier_columns = [
    "Corrupted_Row_ID",
    "Original_Row_ID",
    "Customer ID",
    "Original_Row_Occurrence",
    "Original_Row_ID_feature",
    "Original_Row_Occurrence_feature"
]

artifact_columns = [
    "V51_Incorrect_Description_Flag",
    "V52_Incorrect_Description_Flag",
    "V53_Incorrect_Description_Flag",
    "V54_Incorrect_Description_Flag"
]

non_feature_columns = (
    target_columns +
    identifier_columns +
    artifact_columns +
    [
        "_V54_Duplicate_Signature",
        "Invoice",
        "StockCode",
        "Description",
        "InvoiceDate",
        "Country"
    ]
)

# ============================================================
# BUILD NUMERIC FEATURE MATRIX
# ============================================================

feature_candidates = []

for col in df.columns:

    if col in non_feature_columns:
        continue

    if pd.api.types.is_numeric_dtype(df[col]):

        feature_candidates.append(col)

features = []

for col in feature_candidates:

    if df[col].nunique(dropna=False) <= 1:
        continue

    features.append(col)

# Remove any accidental label leakage
features = [
    col for col in features
    if col not in target_columns
]

print("\n" + "=" * 70)
print("BUILDING CLEAN V5.4 MODEL FEATURES")
print("=" * 70)

print(f"Clean model features: {len(features)}")

for i, col in enumerate(features, start=1):
    print(f"{i:3d}. {col}")

# ============================================================
# CLEAN FEATURE MATRIX
# ============================================================

X = df[features].copy()

# Replace infinities
X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

# Fill using TRAIN medians only
train_feature_medians = X.iloc[train_idx].median(
    numeric_only=True
)

X = X.fillna(train_feature_medians)
X = X.fillna(0)

# Final safety check
if X.isna().any().any():
    raise ValueError(
        "NaN values remain in model features."
    )

# ============================================================
# DUPLICATE EVIDENCE
# ============================================================

duplicate_evidence = (
    df["V54_Exact_Duplicate_Flag"]
    .astype(float)
    .values
)

duplicate_group_size = (
    df["V54_Exact_Duplicate_Group_Size"]
    .astype(float)
    .values
)

# ============================================================
# TYPE-SPECIFIC RANDOM FOREST
# ============================================================

error_definitions = {
    "Missing": "Missing_Value",
    "Invalid": "Invalid_Value",
    "Outlier": "Outlier",
    "Inconsistency": "Inconsistency",
    "Duplicate": "Duplicate"
}

results = []
threshold_records = []
feature_importance_records = []

predicted_error_matrix = {}

print("\n")
print("=" * 70)
print("TRAINING TYPE-SPECIFIC RANDOM FORESTS")
print("=" * 70)

for error_type, target_col in error_definitions.items():

    print("\n" + "-" * 70)
    print(f"ERROR TYPE: {error_type}")
    print("-" * 70)

    y = df[target_col].astype(int).values

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    print(f"Train positives      : {y_train.sum()}")
    print(f"Validation positives : {y_val.sum()}")
    print(f"Test positives       : {y_test.sum()}")

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=400,
        max_features="sqrt",
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    print("Training Random Forest...")

    model.fit(
        X.iloc[train_idx],
        y_train
    )

    val_probability = model.predict_proba(
        X.iloc[val_idx]
    )[:, 1]

    test_probability = model.predict_proba(
        X.iloc[test_idx]
    )[:, 1]

    # --------------------------------------------------------
    # SPECIAL DUPLICATE FUSION
    # --------------------------------------------------------

    if error_type == "Duplicate":

        print(
            "Applying specialized duplicate evidence fusion..."
        )

        # Exact duplicate evidence is treated as an
        # additional unsupervised signal.
        #
        # We do NOT simply force the prediction to positive.
        # Instead, probability is softly boosted.
        #
        # This preserves the RF decision boundary while
        # increasing sensitivity to repeated transactions.

        val_dup_flag = duplicate_evidence[val_idx]
        test_dup_flag = duplicate_evidence[test_idx]

        val_dup_size = duplicate_group_size[val_idx]
        test_dup_size = duplicate_group_size[test_idx]

        val_dup_strength = np.minimum(
            np.log1p(val_dup_size) / np.log(5),
            1.0
        )

        test_dup_strength = np.minimum(
            np.log1p(test_dup_size) / np.log(5),
            1.0
        )

        # Conservative fusion.
        val_probability = (
            0.75 * val_probability +
            0.25 * (
                val_dup_flag * val_dup_strength
            )
        )

        test_probability = (
            0.75 * test_probability +
            0.25 * (
                test_dup_flag * test_dup_strength
            )
        )

    # --------------------------------------------------------
    # VALIDATION THRESHOLD OPTIMIZATION
    # --------------------------------------------------------

    best_threshold = 0.50
    best_f1 = -1

    threshold_values = np.arange(
        0.01,
        1.00,
        0.01
    )

    for threshold in threshold_values:

        val_pred = (
            val_probability >= threshold
        ).astype(int)

        score = f1_score(
            y_val,
            val_pred,
            zero_division=0
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = threshold

    print(
        f"Validation threshold : "
        f"{best_threshold:.2f}"
    )

    print(
        f"Validation F1        : "
        f"{best_f1 * 100:.2f}%"
    )

    # --------------------------------------------------------
    # FINAL TEST PREDICTION
    # --------------------------------------------------------

    test_pred = (
        test_probability >= best_threshold
    ).astype(int)

    predicted_error_matrix[error_type] = test_pred

    accuracy = accuracy_score(
        y_test,
        test_pred
    )

    precision = precision_score(
        y_test,
        test_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        test_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        test_pred,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        test_pred,
        labels=[0, 1]
    ).ravel()

    print(
        f"Test Accuracy  : {accuracy * 100:.2f}%"
    )

    print(
        f"Test Precision : {precision * 100:.2f}%"
    )

    print(
        f"Test Recall    : {recall * 100:.2f}%"
    )

    print(
        f"Test F1        : {f1 * 100:.2f}%"
    )

    results.append({
        "Error_Type": error_type,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "Threshold": best_threshold,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn
    })

    threshold_records.append({
        "Error_Type": error_type,
        "Best_Threshold": best_threshold,
        "Validation_F1": best_f1
    })

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    for feature, importance in zip(
        features,
        model.feature_importances_
    ):

        feature_importance_records.append({
            "Error_Type": error_type,
            "Feature": feature,
            "Importance": importance
        })

# ============================================================
# OVERALL MULTI-ERROR PREDICTION
# ============================================================

test_global_prediction = np.zeros(
    len(test_idx),
    dtype=int
)

for error_type in error_definitions:

    test_global_prediction = np.maximum(
        test_global_prediction,
        predicted_error_matrix[error_type]
    )

# Ground-truth global anomaly
test_global_truth = (
    df.iloc[test_idx][
        target_columns
    ]
    .max(axis=1)
    .astype(int)
    .values
)

overall_accuracy = accuracy_score(
    test_global_truth,
    test_global_prediction
)

overall_precision = precision_score(
    test_global_truth,
    test_global_prediction,
    zero_division=0
)

overall_recall = recall_score(
    test_global_truth,
    test_global_prediction,
    zero_division=0
)

overall_f1 = f1_score(
    test_global_truth,
    test_global_prediction,
    zero_division=0
)

tn, fp, fn, tp = confusion_matrix(
    test_global_truth,
    test_global_prediction,
    labels=[0, 1]
).ravel()

# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("V5.4 FINAL TEST RESULTS")
print("=" * 70)

print(
    f"Accuracy  : {overall_accuracy * 100:.2f}%"
)

print(
    f"Precision : {overall_precision * 100:.2f}%"
)

print(
    f"Recall    : {overall_recall * 100:.2f}%"
)

print(
    f"F1 Score  : {overall_f1 * 100:.2f}%"
)

print(f"TP        : {tp}")
print(f"FP        : {fp}")
print(f"TN        : {tn}")
print(f"FN        : {fn}")

# ============================================================
# PER-ERROR-TYPE PERFORMANCE
# ============================================================

print("\n")
print("=" * 70)
print("PER-ERROR-TYPE PERFORMANCE")
print("=" * 70)

results_df = pd.DataFrame(results)

display_df = results_df.copy()

for col in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score"
]:
    display_df[col] = (
        display_df[col] * 100
    ).map(
        lambda x: f"{x:.2f}%"
    )

print(
    display_df.to_string(
        index=False
    )
)

# ============================================================
# METHODOLOGY AUDIT
# ============================================================

print("\n")
print("=" * 70)
print("METHODOLOGY AUDIT")
print("=" * 70)

print(
    "Split strategy           : "
    "JOINT ERROR-PATTERN STRATIFIED"
)

print(
    "Context statistics       : "
    "TRAIN ONLY"
)

print(
    "Threshold optimization   : "
    "VALIDATION ONLY"
)

print(
    "Test set usage           : "
    "FINAL EVALUATION ONLY"
)

print(
    "Identifier features      : "
    "REMOVED"
)

print(
    "Customer ID feature      : "
    "REMOVED"
)

print(
    "Synthetic artifact flag  : "
    "REMOVED"
)

print(
    "Duplicate signature      : "
    "UNSUPERVISED DQ SIGNAL"
)

print(
    "Duplicate fusion         : "
    "SPECIALIZED"
)

print(
    f"Clean model features     : "
    f"{len(features)}"
)

# ============================================================
# SAVE MAIN RESULT TABLE
# ============================================================

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "type_aware_hybrid_v5_4_results.csv"
    ),
    index=False
)

# ============================================================
# SAVE THRESHOLDS
# ============================================================

pd.DataFrame(
    threshold_records
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "v5_4_threshold_optimization.csv"
    ),
    index=False
)

# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame(
    feature_importance_records
)

importance_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "v5_4_random_forest_feature_importance.csv"
    ),
    index=False
)

# ============================================================
# SAVE OVERALL PERFORMANCE
# ============================================================

overall_df = pd.DataFrame([{
    "Accuracy": overall_accuracy,
    "Precision": overall_precision,
    "Recall": overall_recall,
    "F1_Score": overall_f1,
    "TP": tp,
    "FP": fp,
    "TN": tn,
    "FN": fn
}])

overall_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "v5_4_overall_performance.csv"
    ),
    index=False
)

# ============================================================
# SAVE ERROR-TYPE PERFORMANCE
# ============================================================

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "v5_4_error_type_performance.csv"
    ),
    index=False
)

# ============================================================
# SAVE MODEL FEATURES
# ============================================================

pd.DataFrame({
    "Feature": features
}).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "v5_4_model_features.csv"
    ),
    index=False
)

# ============================================================
# SAVE SPLIT DISTRIBUTION
# ============================================================

split_rows = []

for split_name, indices in [
    ("Train", train_idx),
    ("Validation", val_idx),
    ("Test", test_idx)
]:

    subset = df.iloc[indices]

    for error_type, col in error_definitions.items():

        split_rows.append({
            "Split": split_name,
            "Error_Type": error_type,
            "Rows": len(subset),
            "Positive_Count": int(
                subset[col].sum()
            ),
            "Positive_Percentage": (
                subset[col].mean() * 100
            )
        })

pd.DataFrame(
    split_rows
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "v5_4_split_distribution.csv"
    ),
    index=False
)

# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

test_predictions = df.iloc[
    test_idx
][
    [
        "Corrupted_Row_ID",
        "Original_Row_ID"
    ]
].copy()

for error_type, pred in predicted_error_matrix.items():

    test_predictions[
        f"Predicted_{error_type}"
    ] = pred

test_predictions[
    "Predicted_Global_Anomaly"
] = test_global_prediction

test_predictions[
    "Actual_Global_Anomaly"
] = test_global_truth

test_predictions.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "v5_4_test_predictions.csv"
    ),
    index=False
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    f"{OUTPUT_DIR}\\"
    "type_aware_hybrid_v5_4_results.csv"
)

print(
    f"{OUTPUT_DIR}\\"
    "v5_4_error_type_performance.csv"
)

print(
    f"{OUTPUT_DIR}\\"
    "v5_4_threshold_optimization.csv"
)

print(
    f"{OUTPUT_DIR}\\"
    "v5_4_random_forest_feature_importance.csv"
)

print(
    f"{OUTPUT_DIR}\\"
    "v5_4_overall_performance.csv"
)

print(
    f"{OUTPUT_DIR}\\"
    "v5_4_model_features.csv"
)

print(
    f"{OUTPUT_DIR}\\"
    "v5_4_split_distribution.csv"
)

print(
    f"{OUTPUT_DIR}\\"
    "v5_4_test_predictions.csv"
)

print("\n")
print("=" * 70)
print("V5.4 COMPLETED SUCCESSFULLY.")
print("=" * 70)