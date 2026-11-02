# ============================================================
# TYPE-AWARE HYBRID V4
# AI-Based Data Quality Monitoring
#
# V4:
# - Leakage-safe train / validation / test split
# - Type-specific models
# - Logistic Regression + Random Forest ensemble
# - Validation-only threshold optimization
# - V3-compatible contextual features
# - No test-set tuning
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

V2_RESULTS = os.path.join(
    BASE_DIR,
    "results",
    "metrics",
    "v2",
    "smart_hybrid_v2_results.csv"
)

V2_FEATURES = os.path.join(
    BASE_DIR,
    "results",
    "metrics",
    "v2",
    "ml_features_v2.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "metrics",
    "v4"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# ERROR TYPES
# ============================================================

ERROR_TYPES = {

    "Missing": {
        "gt": "Missing_Value",
        "dq": "DQ_Missing_Value"
    },

    "Invalid": {
        "gt": "Invalid_Value",
        "dq": "DQ_Invalid_Value"
    },

    "Outlier": {
        "gt": "Outlier",
        "dq": "DQ_Outlier"
    },

    "Inconsistency": {
        "gt": "Inconsistency",
        "dq": "DQ_Inconsistency"
    },

    "Duplicate": {
        "gt": "Duplicate",
        "dq": "DQ_Duplicate"
    }
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    return {

        "Accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "F1_Score": f1_score(
            y_true,
            y_pred,
            zero_division=0
        )
    }


def find_best_threshold(
    y_true,
    scores
):

    best_threshold = 0.50
    best_f1 = -1

    threshold_results = []

    for threshold in np.arange(
        0.05,
        0.96,
        0.01
    ):

        predictions = (
            scores >= threshold
        ).astype(int)

        metrics = calculate_metrics(
            y_true,
            predictions
        )

        threshold_results.append({

            "Threshold": threshold,

            "Accuracy": metrics["Accuracy"],

            "Precision": metrics["Precision"],

            "Recall": metrics["Recall"],

            "F1_Score": metrics["F1_Score"]

        })

        if metrics["F1_Score"] > best_f1:

            best_f1 = metrics["F1_Score"]

            best_threshold = threshold

    return (
        best_threshold,
        best_f1,
        pd.DataFrame(
            threshold_results
        )
    )


# ============================================================
# START
# ============================================================

print("=" * 65)
print("TYPE-AWARE HYBRID V4")
print("=" * 65)


# ============================================================
# LOAD V2 RESULTS
# ============================================================

print("\nLoading V2 results...")

results = pd.read_csv(
    V2_RESULTS
)

print(
    f"V2 result rows: "
    f"{len(results):,}"
)


# ============================================================
# LOAD FEATURES
# ============================================================

print("\nLoading V2 features...")

features = pd.read_csv(
    V2_FEATURES
)

print(
    f"Feature rows: "
    f"{len(features):,}"
)


# ============================================================
# VALIDATE V2 RESULTS
# ============================================================

required_columns = [

    "Original_Row_ID",

    "Corrupted_Row_ID",

    "Data_Split",

    "IF_Normalized_Score",

    "LOF_Normalized_Score",

    "SVM_Normalized_Score",

    "ML_Combined_Score",

    "DQ_Missing_Value",

    "DQ_Invalid_Value",

    "DQ_Outlier",

    "DQ_Inconsistency",

    "DQ_Duplicate",

    "Missing_Value",

    "Invalid_Value",

    "Outlier",

    "Inconsistency",

    "Duplicate",

    "Ground_Truth_Anomaly"

]


missing_columns = [

    column

    for column in required_columns

    if column not in results.columns

]


if missing_columns:

    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(
            missing_columns
        )
    )


# ============================================================
# CONTEXTUAL FEATURES
# ============================================================

print(
    "\nPreparing contextual features..."
)


# Only one source row per Original_Row_ID
features_unique = (

    features

    .drop_duplicates(
        subset=[
            "Original_Row_ID"
        ],
        keep="first"
    )

    .copy()

)


print(
    "Unique contextual source rows: "
    f"{len(features_unique):,}"
)


context = pd.DataFrame()

context[
    "Original_Row_ID"
] = features_unique[
    "Original_Row_ID"
]


# ============================================================
# PRODUCT CONTEXT
# ============================================================

if "StockCode" in features_unique.columns:

    product_group = (

        features_unique

        .groupby(
            "StockCode",
            dropna=False
        )

    )


    if "Abs_Price" in features_unique.columns:

        context[
            "Product_Median_Price"
        ] = (

            features_unique[
                "StockCode"
            ]

            .map(
                product_group[
                    "Abs_Price"
                ].median()
            )

        )


    if "Abs_Quantity" in features_unique.columns:

        context[
            "Product_Median_Quantity"
        ] = (

            features_unique[
                "StockCode"
            ]

            .map(
                product_group[
                    "Abs_Quantity"
                ].median()
            )

        )


    context[
        "Product_Frequency"
    ] = (

        features_unique[
            "StockCode"
        ]

        .map(
            product_group.size()
        )

    )


# ============================================================
# CUSTOMER CONTEXT
# ============================================================

if "Customer ID" in features_unique.columns:

    customer_group = (

        features_unique

        .groupby(
            "Customer ID",
            dropna=False
        )

    )

    context[
        "Customer_Frequency"
    ] = (

        features_unique[
            "Customer ID"
        ]

        .map(
            customer_group.size()
        )

    )


# ============================================================
# INVOICE CONTEXT
# ============================================================

if "Invoice" in features_unique.columns:

    invoice_group = (

        features_unique

        .groupby(
            "Invoice",
            dropna=False
        )

    )

    context[
        "Invoice_Item_Count"
    ] = (

        features_unique[
            "Invoice"
        ]

        .map(
            invoice_group.size()
        )

    )


# ============================================================
# DUPLICATE CONTEXT
# ============================================================

duplicate_counts = (

    results

    .groupby(
        "Original_Row_ID"
    )

    .size()

)


context[
    "Duplicate_Group_Size"
] = (

    context[
        "Original_Row_ID"
    ]

    .map(
        duplicate_counts
    )

    .fillna(1)

)


context[
    "Duplicate_Group_Evidence"
] = (

    context[
        "Duplicate_Group_Size"
    ] > 1

).astype(int)


# ============================================================
# MERGE
# ============================================================

results = results.merge(

    context,

    on="Original_Row_ID",

    how="left",

    validate="many_to_one"

)


print(
    f"Merged dataset rows: "
    f"{len(results):,}"
)


if len(results) != 530715:

    raise ValueError(
        "ERROR: Expected 530715 rows "
        f"after merge but got "
        f"{len(results)}"
    )


# ============================================================
# CLEAN CONTEXT
# ============================================================

context_columns = [

    column

    for column in context.columns

    if column != "Original_Row_ID"

]


for column in context_columns:

    results[column] = pd.to_numeric(

        results[column],

        errors="coerce"

    )

    results[column] = (

        results[column]

        .replace(
            [np.inf, -np.inf],
            np.nan
        )

        .fillna(0)

    )


# ============================================================
# NORMALIZE CONTEXT
# ============================================================

for column in context_columns:

    low = results[
        column
    ].quantile(0.01)

    high = results[
        column
    ].quantile(0.99)


    if high > low:

        results[column] = (

            (
                results[column]
                - low
            )

            /

            (
                high
                - low
            )

        ).clip(
            0,
            1
        )

    else:

        results[column] = 0.0


# ============================================================
# DATA SPLIT
# ============================================================

print(
    "\nChecking data split..."
)

print(
    results[
        "Data_Split"
    ].value_counts()
)


train = results[
    results["Data_Split"] == "Train"
].copy()


validation = results[
    results["Data_Split"] == "Validation"
].copy()


test = results[
    results["Data_Split"] == "Test"
].copy()


print(
    "\nData split:"
)

print(
    f"Train      : "
    f"{len(train):,}"
)

print(
    f"Validation : "
    f"{len(validation):,}"
)

print(
    f"Test       : "
    f"{len(test):,}"
)


# ============================================================
# V4 FEATURES
# ============================================================

base_features = [

    "IF_Normalized_Score",

    "LOF_Normalized_Score",

    "SVM_Normalized_Score",

    "ML_Combined_Score",

    "DQ_Missing_Value",

    "DQ_Invalid_Value",

    "DQ_Outlier",

    "DQ_Inconsistency",

    "DQ_Duplicate"

]


model_features = (

    base_features

    +

    context_columns

)


print(
    "\nTotal V4 model features: "
    f"{len(model_features)}"
)


for feature in model_features:

    print(
        f"  - {feature}"
    )


# ============================================================
# PREPARE MATRICES
# ============================================================

X_train = (

    train[
        model_features
    ]

    .replace(
        [np.inf, -np.inf],
        np.nan
    )

    .fillna(0)

)


X_validation = (

    validation[
        model_features
    ]

    .replace(
        [np.inf, -np.inf],
        np.nan
    )

    .fillna(0)

)


X_test = (

    test[
        model_features
    ]

    .replace(
        [np.inf, -np.inf],
        np.nan
    )

    .fillna(0)

)


# ============================================================
# STANDARDIZATION FOR LOGISTIC REGRESSION
# ============================================================

scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)


X_validation_scaled = scaler.transform(
    X_validation
)


X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# OUTPUT DATAFRAME
# ============================================================

v4_test = test[
    [
        "Original_Row_ID",

        "Corrupted_Row_ID",

        "Data_Split"

    ]
].copy()


# ============================================================
# PERFORMANCE STORAGE
# ============================================================

performance_rows = []

threshold_rows = []

importance_rows = []


# ============================================================
# TRAIN TYPE-SPECIFIC V4 MODELS
# ============================================================

print("\n")
print("=" * 65)
print("TRAINING TYPE-SPECIFIC V4 ENSEMBLES")
print("=" * 65)


for error_name, config in ERROR_TYPES.items():

    print("\n")
    print("-" * 65)

    print(
        f"Training model for: "
        f"{error_name}"
    )

    print("-" * 65)


    target_column = config["gt"]


    # --------------------------------------------------------
    # TARGETS
    # --------------------------------------------------------

    y_train = (

        train[
            target_column
        ]

        .astype(int)

        .values

    )


    y_validation = (

        validation[
            target_column
        ]

        .astype(int)

        .values

    )


    y_test = (

        test[
            target_column
        ]

        .astype(int)

        .values

    )


    print(
        f"Training positives   : "
        f"{y_train.sum():,}"
    )

    print(
        f"Validation positives : "
        f"{y_validation.sum():,}"
    )

    print(
        f"Test positives       : "
        f"{y_test.sum():,}"
    )


    # ========================================================
    # LOGISTIC REGRESSION
    # ========================================================

    logistic_model = LogisticRegression(

        class_weight="balanced",

        max_iter=1000,

        C=1.0,

        random_state=42

    )


    logistic_model.fit(

        X_train_scaled,

        y_train

    )


    logistic_validation = (

        logistic_model

        .predict_proba(
            X_validation_scaled
        )[:, 1]

    )


    logistic_test = (

        logistic_model

        .predict_proba(
            X_test_scaled
        )[:, 1]

    )


    # ========================================================
    # RANDOM FOREST
    # ========================================================

    print(
        "\nTraining Random Forest..."
    )


    rf_model = RandomForestClassifier(

        n_estimators=300,

        max_depth=12,

        min_samples_leaf=3,

        max_features="sqrt",

        class_weight="balanced_subsample",

        random_state=42,

        n_jobs=-1

    )


    rf_model.fit(

        X_train,

        y_train

    )


    rf_validation = (

        rf_model

        .predict_proba(
            X_validation
        )[:, 1]

    )


    rf_test = (

        rf_model

        .predict_proba(
            X_test
        )[:, 1]

    )


    # ========================================================
    # ENSEMBLE
    # ========================================================

    # Equal contribution from:
    #
    # Logistic Regression
    # Random Forest
    #
    # This combines linear evidence with nonlinear evidence.

    validation_scores = (

        0.50 * logistic_validation

        +

        0.50 * rf_validation

    )


    test_scores = (

        0.50 * logistic_test

        +

        0.50 * rf_test

    )


    # ========================================================
    # VALIDATION THRESHOLD
    # ========================================================

    best_threshold, validation_f1, threshold_df = (

        find_best_threshold(

            y_validation,

            validation_scores

        )

    )


    threshold_df[
        "Error_Type"
    ] = error_name


    threshold_rows.append(
        threshold_df
    )


    print(
        f"\nBest threshold : "
        f"{best_threshold:.2f}"
    )


    print(
        f"Validation F1  : "
        f"{validation_f1 * 100:.2f}%"
    )


    # ========================================================
    # TEST
    # ========================================================

    test_predictions = (

        test_scores

        >= best_threshold

    ).astype(int)


    metrics = calculate_metrics(

        y_test,

        test_predictions

    )


    print(
        f"\nTest Accuracy  : "
        f"{metrics['Accuracy'] * 100:.2f}%"
    )

    print(
        f"Test Precision : "
        f"{metrics['Precision'] * 100:.2f}%"
    )

    print(
        f"Test Recall    : "
        f"{metrics['Recall'] * 100:.2f}%"
    )

    print(
        f"Test F1        : "
        f"{metrics['F1_Score'] * 100:.2f}%"
    )


    # ========================================================
    # STORE SCORES
    # ========================================================

    v4_test[
        f"V4_{error_name}_Score"
    ] = test_scores


    v4_test[
        f"V4_{error_name}_Prediction"
    ] = test_predictions


    # ========================================================
    # PERFORMANCE
    # ========================================================

    performance_rows.append({

        "Error_Type":
            error_name,

        "Best_Validation_Threshold":
            best_threshold,

        "Validation_F1":
            validation_f1,

        "Test_Accuracy":
            metrics["Accuracy"],

        "Test_Precision":
            metrics["Precision"],

        "Test_Recall":
            metrics["Recall"],

        "Test_F1":
            metrics["F1_Score"],

        "Actual_Positive":
            int(y_test.sum()),

        "Predicted_Positive":
            int(
                test_predictions.sum()
            )

    })


    # ========================================================
    # RANDOM FOREST FEATURE IMPORTANCE
    # ========================================================

    for feature, importance in zip(

        model_features,

        rf_model.feature_importances_

    ):

        importance_rows.append({

            "Error_Type":
                error_name,

            "Feature":
                feature,

            "RandomForest_Importance":
                importance

        })


# ============================================================
# SAVE ERROR TYPE PERFORMANCE
# ============================================================

performance_df = pd.DataFrame(
    performance_rows
)


performance_path = os.path.join(

    OUTPUT_DIR,

    "v4_error_type_performance.csv"

)


performance_df.to_csv(

    performance_path,

    index=False

)


# ============================================================
# SAVE THRESHOLDS
# ============================================================

thresholds_df = pd.concat(

    threshold_rows,

    ignore_index=True

)


threshold_path = os.path.join(

    OUTPUT_DIR,

    "v4_threshold_optimization.csv"

)


thresholds_df.to_csv(

    threshold_path,

    index=False

)


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame(
    importance_rows
)


importance_path = os.path.join(

    OUTPUT_DIR,

    "v4_random_forest_feature_importance.csv"

)


importance_df.to_csv(

    importance_path,

    index=False

)


# ============================================================
# ADD GROUND TRUTH
# ============================================================

for error_name, config in ERROR_TYPES.items():

    v4_test[
        config["gt"]
    ] = (

        test[
            config["gt"]
        ]

        .astype(int)

        .values

    )


v4_test[
    "Ground_Truth_Anomaly"
] = (

    test[
        "Ground_Truth_Anomaly"
    ]

    .astype(int)

    .values

)


# ============================================================
# TYPE-AWARE FUSION
# ============================================================

score_columns = [

    f"V4_{error_name}_Score"

    for error_name in ERROR_TYPES

]


prediction_columns = [

    f"V4_{error_name}_Prediction"

    for error_name in ERROR_TYPES

]


# ------------------------------------------------------------
# Maximum probability
# ------------------------------------------------------------

v4_test[
    "V4_Max_Type_Score"
] = (

    v4_test[
        score_columns
    ]

    .max(axis=1)

)


# ------------------------------------------------------------
# Final anomaly
# ------------------------------------------------------------

v4_test[
    "V4_Anomaly"
] = (

    v4_test[
        prediction_columns
    ]

    .max(axis=1)

)


# ============================================================
# ERROR TYPE CLASSIFICATION
# ============================================================

def determine_error_type(row):

    active = []

    for error_name in ERROR_TYPES:

        prediction_column = (

            f"V4_{error_name}_Prediction"

        )

        score_column = (

            f"V4_{error_name}_Score"

        )


        if row[
            prediction_column
        ] == 1:

            active.append(

                (
                    error_name,

                    row[
                        score_column
                    ]

                )

            )


    if not active:

        return "Normal"


    active.sort(

        key=lambda x: x[1],

        reverse=True

    )


    return active[0][0]


v4_test[
    "V4_Error_Type"
] = (

    v4_test

    .apply(
        determine_error_type,
        axis=1
    )

)


# ============================================================
# OVERALL TEST PERFORMANCE
# ============================================================

y_true = (

    v4_test[
        "Ground_Truth_Anomaly"
    ]

    .astype(int)

)


y_pred = (

    v4_test[
        "V4_Anomaly"
    ]

    .astype(int)

)


overall = calculate_metrics(

    y_true,

    y_pred

)


tn, fp, fn, tp = confusion_matrix(

    y_true,

    y_pred,

    labels=[0, 1]

).ravel()


# ============================================================
# PRINT OVERALL
# ============================================================

print("\n")
print("=" * 65)
print("V4 OVERALL TEST PERFORMANCE")
print("=" * 65)


print(

    f"Accuracy  : "
    f"{overall['Accuracy'] * 100:.2f}%"

)


print(

    f"Precision : "
    f"{overall['Precision'] * 100:.2f}%"

)


print(

    f"Recall    : "
    f"{overall['Recall'] * 100:.2f}%"

)


print(

    f"F1 Score  : "
    f"{overall['F1_Score'] * 100:.2f}%"

)


print(
    f"TP        : {tp:,}"
)


print(
    f"FP        : {fp:,}"
)


print(
    f"TN        : {tn:,}"
)


print(
    f"FN        : {fn:,}"
)


print(

    f"Predicted anomalies : "
    f"{y_pred.sum():,}"

)


print(

    f"Actual anomalies    : "
    f"{y_true.sum():,}"

)


# ============================================================
# ERROR TYPE PERFORMANCE
# ============================================================

print("\n")
print("=" * 65)
print("V4 ERROR-TYPE PERFORMANCE")
print("=" * 65)


for error_name, config in ERROR_TYPES.items():

    y_type_true = (

        v4_test[
            config["gt"]
        ]

        .astype(int)

    )


    y_type_pred = (

        v4_test[
            f"V4_{error_name}_Prediction"
        ]

        .astype(int)

    )


    metrics = calculate_metrics(

        y_type_true,

        y_type_pred

    )


    print(
        f"\n{error_name}"
    )


    print(

        f"  Precision : "
        f"{metrics['Precision'] * 100:.2f}%"

    )


    print(

        f"  Recall    : "
        f"{metrics['Recall'] * 100:.2f}%"

    )


    print(

        f"  F1 Score  : "
        f"{metrics['F1_Score'] * 100:.2f}%"

    )


# ============================================================
# MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame([

    {

        "Model":
            "Smart Hybrid V1",

        "Accuracy":
            0.9356,

        "Precision":
            0.3468,

        "Recall":
            0.3260,

        "F1_Score":
            0.3361

    },

    {

        "Model":
            "Smart Hybrid V2",

        "Accuracy":
            0.9025,

        "Precision":
            0.2921,

        "Recall":
            0.4832,

        "F1_Score":
            0.3641

    },

    {

        "Model":
            "Type-Aware Hybrid V3",

        "Accuracy":
            0.9701,

        "Precision":
            0.6743,

        "Recall":
            0.9326,

        "F1_Score":
            0.7827

    },

    {

        "Model":
            "Type-Aware Hybrid V4",

        "Accuracy":
            overall["Accuracy"],

        "Precision":
            overall["Precision"],

        "Recall":
            overall["Recall"],

        "F1_Score":
            overall["F1_Score"]

    }

])


comparison_path = os.path.join(

    OUTPUT_DIR,

    "v4_model_comparison.csv"

)


comparison.to_csv(

    comparison_path,

    index=False

)


# ============================================================
# SAVE FINAL RESULTS
# ============================================================

final_path = os.path.join(

    OUTPUT_DIR,

    "type_aware_hybrid_v4_results.csv"

)


v4_test.to_csv(

    final_path,

    index=False

)


# ============================================================
# TRAINING SUMMARY
# ============================================================

summary = pd.DataFrame({

    "Metric": [

        "Train Rows",

        "Validation Rows",

        "Test Rows",

        "Number of Features",

        "Test Accuracy",

        "Test Precision",

        "Test Recall",

        "Test F1",

        "True Positives",

        "False Positives",

        "True Negatives",

        "False Negatives"

    ],

    "Value": [

        len(train),

        len(validation),

        len(test),

        len(model_features),

        overall["Accuracy"],

        overall["Precision"],

        overall["Recall"],

        overall["F1_Score"],

        tp,

        fp,

        tn,

        fn

    ]

})


summary_path = os.path.join(

    OUTPUT_DIR,

    "v4_training_summary.csv"

)


summary.to_csv(

    summary_path,

    index=False

)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 65)
print("V4 COMPLETE")
print("=" * 65)


print("\nSaved files:")

print(
    f"1. {final_path}"
)

print(
    f"2. {performance_path}"
)

print(
    f"3. {threshold_path}"
)

print(
    f"4. {importance_path}"
)

print(
    f"5. {comparison_path}"
)

print(
    f"6. {summary_path}"
)


print(
    "\nV4 execution completed successfully."
)