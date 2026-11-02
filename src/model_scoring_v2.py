import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


# ============================================================
# SMART HYBRID V2 - MODEL SCORING
# ============================================================

INPUT_FILE = "results/metrics/v2/ml_features_v2.csv"
OUTPUT_FILE = "results/metrics/v2/model_scores_v2.csv"

RANDOM_STATE = 42

# Train / validation / test split
TRAIN_SIZE = 0.60
VALIDATION_SIZE = 0.20
TEST_SIZE = 0.20

# Model settings
IF_CONTAMINATION = 0.01
LOF_CONTAMINATION = 0.01
LOF_NEIGHBORS = 20

SVM_NU = 0.01
SVM_KERNEL = "rbf"

# SVM can become expensive on hundreds of thousands
# of observations, so use a representative training sample.
SVM_TRAIN_SAMPLE = 80000


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def get_feature_columns(df):
    """
    Select numerical ML features.

    Identifier columns and row-tracking columns are excluded.
    Ground-truth information is not used here.
    """

    excluded_columns = {
    "Original_Row_ID",
    "Corrupted_Row_ID",
    "Original_Row_Occurrence",
    "Customer ID",
}
    feature_columns = []

    for column in df.columns:

        if column in excluded_columns:
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            feature_columns.append(column)

    return feature_columns


def clean_features(df, feature_columns):
    """
    Clean numerical feature matrix.
    """

    X = df[feature_columns].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)

    return X


# ============================================================
# MAIN
# ============================================================

def main():

    print_section(
        "SMART HYBRID V2 - MODEL SCORING"
    )

    # --------------------------------------------------------
    # 1. Load engineered features
    # --------------------------------------------------------

    print("\nLoading V2 engineered features...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # --------------------------------------------------------
    # 2. Verify required columns
    # --------------------------------------------------------

    required_columns = [
        "Original_Row_ID",
        "Corrupted_Row_ID",
    ]

    for column in required_columns:

        if column not in df.columns:
            raise ValueError(
                f"Required column missing: {column}"
            )

    # --------------------------------------------------------
    # 3. Create split using Original_Row_ID
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # Duplicate rows share the same Original_Row_ID.
    #
    # Therefore, splitting individual rows could put an
    # original record in training and its duplicate in test.
    #
    # We split Original_Row_ID instead.
    #
    # This prevents duplicate leakage between train,
    # validation and test.
    # --------------------------------------------------------

    print_section(
        "CREATING TRAIN / VALIDATION / TEST SPLIT"
    )

    unique_ids = df["Original_Row_ID"].drop_duplicates()

    train_ids, temp_ids = train_test_split(
        unique_ids,
        test_size=(1 - TRAIN_SIZE),
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    validation_ratio_of_temp = (
        VALIDATION_SIZE /
        (VALIDATION_SIZE + TEST_SIZE)
    )

    validation_ids, test_ids = train_test_split(
        temp_ids,
        test_size=(1 - validation_ratio_of_temp),
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    train_mask = df["Original_Row_ID"].isin(train_ids)
    validation_mask = df["Original_Row_ID"].isin(validation_ids)
    test_mask = df["Original_Row_ID"].isin(test_ids)

    df["Data_Split"] = "Unknown"

    df.loc[
        train_mask,
        "Data_Split"
    ] = "Train"

    df.loc[
        validation_mask,
        "Data_Split"
    ] = "Validation"

    df.loc[
        test_mask,
        "Data_Split"
    ] = "Test"

    print(
        f"\nUnique Original IDs:"
    )

    print(
        f"  Train:      {len(train_ids):,}"
    )

    print(
        f"  Validation: {len(validation_ids):,}"
    )

    print(
        f"  Test:       {len(test_ids):,}"
    )

    print(
        f"\nActual rows:"
    )

    print(
        f"  Train:      {train_mask.sum():,}"
    )

    print(
        f"  Validation: {validation_mask.sum():,}"
    )

    print(
        f"  Test:       {test_mask.sum():,}"
    )

    # --------------------------------------------------------
    # 4. Select ML features
    # --------------------------------------------------------

    print_section(
        "SELECTING ML FEATURES"
    )

    feature_columns = get_feature_columns(df)

    print(
        f"\nNumber of ML features: "
        f"{len(feature_columns)}"
    )

    for feature in feature_columns:
        print(f"  ✓ {feature}")

    # --------------------------------------------------------
    # 5. Prepare feature matrices
    # --------------------------------------------------------

    X_all = clean_features(
        df,
        feature_columns
    )

    X_train = X_all.loc[train_mask]
    X_validation = X_all.loc[validation_mask]
    X_test = X_all.loc[test_mask]

    # --------------------------------------------------------
    # 6. Standardization
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # The scaler is fitted ONLY on training data.
    #
    # Validation and test data are transformed using the
    # training scaler.
    #
    # This prevents preprocessing leakage.
    # --------------------------------------------------------

    print_section(
        "STANDARDIZING FEATURES"
    )

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

    print(
        "\nScaler fitted only on training data."
    )

    # --------------------------------------------------------
    # 7. Prepare output dataframe
    # --------------------------------------------------------

    results = df[
        [
            "Original_Row_ID",
            "Corrupted_Row_ID",
            "Original_Row_Occurrence",
            "Data_Split",
        ]
    ].copy()

    # --------------------------------------------------------
    # 8. Isolation Forest
    # --------------------------------------------------------

    print_section(
        "TRAINING ISOLATION FOREST"
    )

    print(
        f"\nContamination: "
        f"{IF_CONTAMINATION}"
    )

    isolation_forest = IsolationForest(
        n_estimators=200,
        contamination=IF_CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    isolation_forest.fit(
        X_train_scaled
    )

    # Predict all splits
    if_train_prediction = isolation_forest.predict(
        X_train_scaled
    )

    if_validation_prediction = isolation_forest.predict(
        X_validation_scaled
    )

    if_test_prediction = isolation_forest.predict(
        X_test_scaled
    )

    # Continuous anomaly score
    # Larger value = more anomalous
    if_train_score = -isolation_forest.decision_function(
        X_train_scaled
    )

    if_validation_score = -isolation_forest.decision_function(
        X_validation_scaled
    )

    if_test_score = -isolation_forest.decision_function(
        X_test_scaled
    )

    results["IF_Prediction"] = 0
    results["IF_Anomaly_Score"] = 0.0

    results.loc[
        train_mask,
        "IF_Prediction"
    ] = (
        if_train_prediction == -1
    ).astype(int)

    results.loc[
        validation_mask,
        "IF_Prediction"
    ] = (
        if_validation_prediction == -1
    ).astype(int)

    results.loc[
        test_mask,
        "IF_Prediction"
    ] = (
        if_test_prediction == -1
    ).astype(int)

    results.loc[
        train_mask,
        "IF_Anomaly_Score"
    ] = if_train_score

    results.loc[
        validation_mask,
        "IF_Anomaly_Score"
    ] = if_validation_score

    results.loc[
        test_mask,
        "IF_Anomaly_Score"
    ] = if_test_score

    print(
        "\nIsolation Forest completed."
    )

    # --------------------------------------------------------
    # 9. Local Outlier Factor
    # --------------------------------------------------------
    #
    # novelty=True allows us to train on training data and
    # score unseen validation/test data.
    # --------------------------------------------------------

    print_section(
        "TRAINING LOCAL OUTLIER FACTOR"
    )

    print(
        f"\nNeighbors: {LOF_NEIGHBORS}"
    )

    print(
        f"Contamination: "
        f"{LOF_CONTAMINATION}"
    )

    lof = LocalOutlierFactor(
        n_neighbors=LOF_NEIGHBORS,
        contamination=LOF_CONTAMINATION,
        novelty=True,
        n_jobs=-1,
    )

    lof.fit(
        X_train_scaled
    )

    # Predictions
    lof_train_prediction = lof.predict(
        X_train_scaled
    )

    lof_validation_prediction = lof.predict(
        X_validation_scaled
    )

    lof_test_prediction = lof.predict(
        X_test_scaled
    )

    # Continuous anomaly score
    lof_train_score = -lof.decision_function(
        X_train_scaled
    )

    lof_validation_score = -lof.decision_function(
        X_validation_scaled
    )

    lof_test_score = -lof.decision_function(
        X_test_scaled
    )

    results["LOF_Prediction"] = 0
    results["LOF_Anomaly_Score"] = 0.0

    results.loc[
        train_mask,
        "LOF_Prediction"
    ] = (
        lof_train_prediction == -1
    ).astype(int)

    results.loc[
        validation_mask,
        "LOF_Prediction"
    ] = (
        lof_validation_prediction == -1
    ).astype(int)

    results.loc[
        test_mask,
        "LOF_Prediction"
    ] = (
        lof_test_prediction == -1
    ).astype(int)

    results.loc[
        train_mask,
        "LOF_Anomaly_Score"
    ] = lof_train_score

    results.loc[
        validation_mask,
        "LOF_Anomaly_Score"
    ] = lof_validation_score

    results.loc[
        test_mask,
        "LOF_Anomaly_Score"
    ] = lof_test_score

    print(
        "\nLOF completed."
    )

    # --------------------------------------------------------
    # 10. One-Class SVM
    # --------------------------------------------------------

    print_section(
        "TRAINING ONE-CLASS SVM"
    )

    print(
        f"\nTraining sample size: "
        f"{SVM_TRAIN_SAMPLE:,}"
    )

    print(
        f"nu: {SVM_NU}"
    )

    # --------------------------------------------------------
    # Sample from training data only.
    # --------------------------------------------------------

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    if len(X_train_scaled) > SVM_TRAIN_SAMPLE:

        svm_indices = rng.choice(
            len(X_train_scaled),
            size=SVM_TRAIN_SAMPLE,
            replace=False,
        )

        X_svm_train = (
            X_train_scaled[svm_indices]
        )

    else:

        X_svm_train = X_train_scaled

    print(
        f"Actual SVM training rows: "
        f"{len(X_svm_train):,}"
    )

    svm = OneClassSVM(
        kernel=SVM_KERNEL,
        gamma="scale",
        nu=SVM_NU,
    )

    svm.fit(
        X_svm_train
    )

    # Predictions
    svm_train_prediction = svm.predict(
        X_train_scaled
    )

    svm_validation_prediction = svm.predict(
        X_validation_scaled
    )

    svm_test_prediction = svm.predict(
        X_test_scaled
    )

    # Continuous anomaly score
    svm_train_score = -svm.decision_function(
        X_train_scaled
    )

    svm_validation_score = -svm.decision_function(
        X_validation_scaled
    )

    svm_test_score = -svm.decision_function(
        X_test_scaled
    )

    results["SVM_Prediction"] = 0
    results["SVM_Anomaly_Score"] = 0.0

    results.loc[
        train_mask,
        "SVM_Prediction"
    ] = (
        svm_train_prediction == -1
    ).astype(int)

    results.loc[
        validation_mask,
        "SVM_Prediction"
    ] = (
        svm_validation_prediction == -1
    ).astype(int)

    results.loc[
        test_mask,
        "SVM_Prediction"
    ] = (
        svm_test_prediction == -1
    ).astype(int)

    results.loc[
        train_mask,
        "SVM_Anomaly_Score"
    ] = svm_train_score

    results.loc[
        validation_mask,
        "SVM_Anomaly_Score"
    ] = svm_validation_score

    results.loc[
        test_mask,
        "SVM_Anomaly_Score"
    ] = svm_test_score

    print(
        "\nOne-Class SVM completed."
    )

    # --------------------------------------------------------
    # 11. Normalize continuous anomaly scores
    # --------------------------------------------------------
    #
    # The three algorithms produce scores on different scales.
    #
    # We therefore normalize them using training-data
    # percentiles.
    #
    # The percentile boundaries are learned ONLY from training.
    # --------------------------------------------------------

    print_section(
        "NORMALIZING ANOMALY SCORES"
    )

    def percentile_normalize(
        train_scores,
        all_scores
    ):

        lower = np.percentile(
            train_scores,
            1
        )

        upper = np.percentile(
            train_scores,
            99
        )

        if upper <= lower:
            return np.zeros_like(
                all_scores,
                dtype=float
            )

        normalized = (
            (all_scores - lower) /
            (upper - lower)
        )

        return np.clip(
            normalized,
            0,
            1
        )

    # Store normalized scores for each split.

    results["IF_Normalized_Score"] = 0.0
    results["LOF_Normalized_Score"] = 0.0
    results["SVM_Normalized_Score"] = 0.0

    results.loc[
        train_mask,
        "IF_Normalized_Score"
    ] = percentile_normalize(
        if_train_score,
        if_train_score
    )

    results.loc[
        validation_mask,
        "IF_Normalized_Score"
    ] = percentile_normalize(
        if_train_score,
        if_validation_score
    )

    results.loc[
        test_mask,
        "IF_Normalized_Score"
    ] = percentile_normalize(
        if_train_score,
        if_test_score
    )

    results.loc[
        train_mask,
        "LOF_Normalized_Score"
    ] = percentile_normalize(
        lof_train_score,
        lof_train_score
    )

    results.loc[
        validation_mask,
        "LOF_Normalized_Score"
    ] = percentile_normalize(
        lof_train_score,
        lof_validation_score
    )

    results.loc[
        test_mask,
        "LOF_Normalized_Score"
    ] = percentile_normalize(
        lof_train_score,
        lof_test_score
    )

    results.loc[
        train_mask,
        "SVM_Normalized_Score"
    ] = percentile_normalize(
        svm_train_score,
        svm_train_score
    )

    results.loc[
        validation_mask,
        "SVM_Normalized_Score"
    ] = percentile_normalize(
        svm_train_score,
        svm_validation_score
    )

    results.loc[
        test_mask,
        "SVM_Normalized_Score"
    ] = percentile_normalize(
        svm_train_score,
        svm_test_score
    )

    # --------------------------------------------------------
    # 12. Combined ML score
    # --------------------------------------------------------
    #
    # This is NOT the final Smart Hybrid decision.
    #
    # It is simply an equal-weight baseline of the three
    # normalized anomaly scores.
    #
    # Later, the validation set will be used to determine
    # whether weighted fusion improves this.
    # --------------------------------------------------------

    results["ML_Combined_Score"] = (
        results["IF_Normalized_Score"] +
        results["LOF_Normalized_Score"] +
        results["SVM_Normalized_Score"]
    ) / 3.0

    # --------------------------------------------------------
    # 13. Save results
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # 14. Summary
    # --------------------------------------------------------

    print_section(
        "MODEL SCORING COMPLETED"
    )

    print(
        f"\nOutput file:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        f"\nRows scored: "
        f"{len(results):,}"
    )

    print(
        "\nSplit distribution:"
    )

    print(
        results["Data_Split"]
        .value_counts()
        .to_string()
    )

    print(
        "\nAnomaly counts:"
    )

    for column in [
        "IF_Prediction",
        "LOF_Prediction",
        "SVM_Prediction",
    ]:

        print(
            f"  {column}: "
            f"{results[column].sum():,}"
        )

    print(
        "\nAverage normalized scores:"
    )

    print(
        f"  IF:  "
        f"{results['IF_Normalized_Score'].mean():.4f}"
    )

    print(
        f"  LOF: "
        f"{results['LOF_Normalized_Score'].mean():.4f}"
    )

    print(
        f"  SVM: "
        f"{results['SVM_Normalized_Score'].mean():.4f}"
    )

    print(
        "\nV2 model scoring is ready."
    )


if __name__ == "__main__":
    main()