import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# SMART HYBRID V2
# DQ EVIDENCE + CONTINUOUS ML ANOMALY SCORES
# ============================================================

MODEL_SCORE_FILE = "results/metrics/v2/model_scores_v2.csv"

DQ_FILE = "results/metrics/dq_rule_results.csv"

GROUND_TRUTH_FILE = "results/metrics/ground_truth.csv"

OUTPUT_FILE = "results/metrics/v2/smart_hybrid_v2_results.csv"

THRESHOLD_FILE = "results/metrics/v2/v2_threshold_optimization.csv"

RANDOM_STATE = 42


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_metrics(y_true, y_pred):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
    }


def print_metrics(title, metrics):

    print("\n" + "-" * 55)
    print(title)
    print("-" * 55)

    print(
        f"Accuracy : {metrics['Accuracy']:.4f}"
    )

    print(
        f"Precision: {metrics['Precision']:.4f}"
    )

    print(
        f"Recall   : {metrics['Recall']:.4f}"
    )

    print(
        f"F1 Score : {metrics['F1_Score']:.4f}"
    )

    print(
        f"TP: {metrics['TP']:,}"
    )

    print(
        f"FP: {metrics['FP']:,}"
    )

    print(
        f"TN: {metrics['TN']:,}"
    )

    print(
        f"FN: {metrics['FN']:,}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SMART HYBRID V2")
    print("DQ EVIDENCE + CONTINUOUS ML SCORES")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load files
    # --------------------------------------------------------

    print("\nLoading model scores...")

    model_df = pd.read_csv(
        MODEL_SCORE_FILE
    )

    print(
        f"Model score rows: "
        f"{len(model_df):,}"
    )

    print("\nLoading DQ results...")

    dq_df = pd.read_csv(
        DQ_FILE
    )

    print(
        f"DQ rows: "
        f"{len(dq_df):,}"
    )

    print("\nLoading ground truth...")

    gt_df = pd.read_csv(
        GROUND_TRUTH_FILE
    )

    print(
        f"Ground truth rows: "
        f"{len(gt_df):,}"
    )

    # --------------------------------------------------------
    # 2. Check required columns
    # --------------------------------------------------------

    required_model_columns = [
        "Original_Row_ID",
        "Corrupted_Row_ID",
        "Data_Split",
        "IF_Normalized_Score",
        "LOF_Normalized_Score",
        "SVM_Normalized_Score",
        "ML_Combined_Score",
    ]

    for column in required_model_columns:

        if column not in model_df.columns:

            raise ValueError(
                f"Missing model column: {column}"
            )

    required_dq_columns = [
        "Original_Row_ID",
        "DQ_Missing_Value",
        "DQ_Invalid_Value",
        "DQ_Outlier",
        "DQ_Inconsistency",
        "DQ_Duplicate",
    ]

    for column in required_dq_columns:

        if column not in dq_df.columns:

            raise ValueError(
                f"Missing DQ column: {column}"
            )

    required_gt_columns = [
        "Original_Row_ID",
        "Missing_Value",
        "Invalid_Value",
        "Outlier",
        "Inconsistency",
        "Duplicate",
    ]

    for column in required_gt_columns:

        if column not in gt_df.columns:

            raise ValueError(
                f"Missing ground-truth column: {column}"
            )

    # --------------------------------------------------------
    # 3. Aggregate DQ evidence by Original_Row_ID
    # --------------------------------------------------------
    #
    # DQ results are associated with Original_Row_ID.
    #
    # Duplicate instances therefore inherit the same
    # row-level DQ evidence.
    # --------------------------------------------------------

    dq_columns = [
        "DQ_Missing_Value",
        "DQ_Invalid_Value",
        "DQ_Outlier",
        "DQ_Inconsistency",
        "DQ_Duplicate",
    ]

    dq_subset = dq_df[
        [
            "Original_Row_ID"
        ] + dq_columns
    ].copy()

    # If duplicates exist in DQ results, retain maximum evidence.
    dq_grouped = (
        dq_subset
        .groupby("Original_Row_ID", as_index=False)
        [dq_columns]
        .max()
    )

    # --------------------------------------------------------
    # 4. Prepare ground truth
    # --------------------------------------------------------

    gt_subset = gt_df[
        required_gt_columns
    ].copy()

    # --------------------------------------------------------
    # 5. Merge model + DQ + ground truth
    # --------------------------------------------------------

    print("\nCombining V2 evidence...")

    results = model_df.merge(
        dq_grouped,
        on="Original_Row_ID",
        how="left"
    )

    results = results.merge(
        gt_subset,
        on="Original_Row_ID",
        how="left"
    )

    # --------------------------------------------------------
    # 6. Handle missing evidence
    # --------------------------------------------------------

    all_evidence_columns = (
        dq_columns +
        [
            "Missing_Value",
            "Invalid_Value",
            "Outlier",
            "Inconsistency",
            "Duplicate",
        ]
    )

    for column in all_evidence_columns:

        results[column] = (
            results[column]
            .fillna(0)
            .astype(int)
        )

    # --------------------------------------------------------
    # 7. Create DQ evidence score
    # --------------------------------------------------------

    results["DQ_Evidence"] = (
        results["DQ_Missing_Value"] +
        results["DQ_Invalid_Value"] +
        results["DQ_Outlier"] +
        results["DQ_Inconsistency"] +
        results["DQ_Duplicate"]
    )

    # --------------------------------------------------------
    # 8. Create ground truth anomaly
    # --------------------------------------------------------

    results["Ground_Truth_Anomaly"] = (
        (
            results["Missing_Value"] +
            results["Invalid_Value"] +
            results["Outlier"] +
            results["Inconsistency"] +
            results["Duplicate"]
        ) > 0
    ).astype(int)

    # --------------------------------------------------------
    # 9. ML evidence
    # --------------------------------------------------------
    #
    # ML_Combined_Score is already normalized to [0,1].
    #
    # We additionally create a stronger ML score that gives
    # slightly more weight to the strongest detector.
    #
    # This is still unsupervised: no ground truth is involved.
    # --------------------------------------------------------

    results["ML_Max_Score"] = results[
        [
            "IF_Normalized_Score",
            "LOF_Normalized_Score",
            "SVM_Normalized_Score",
        ]
    ].max(axis=1)

    results["ML_Median_Score"] = results[
        [
            "IF_Normalized_Score",
            "LOF_Normalized_Score",
            "SVM_Normalized_Score",
        ]
    ].median(axis=1)

    # --------------------------------------------------------
    # 10. Continuous hybrid evidence
    # --------------------------------------------------------
    #
    # DQ evidence is converted into a bounded score.
    #
    # 0 DQ errors -> 0.00
    # 1 DQ error  -> 0.20
    # 2 DQ errors -> 0.40
    # ...
    # 5 DQ errors -> 1.00
    #
    # ML score remains continuous.
    # --------------------------------------------------------

    results["DQ_Normalized_Score"] = (
        results["DQ_Evidence"] / 5.0
    ).clip(0, 1)

    # --------------------------------------------------------
    # 11. Initial fusion score
    # --------------------------------------------------------
    #
    # DQ gets 50%
    # ML average gets 50%
    #
    # The validation set will determine the final threshold.
    # --------------------------------------------------------

    results["Hybrid_Score"] = (
        0.50 *
        results["DQ_Normalized_Score"]
        +
        0.50 *
        results["ML_Combined_Score"]
    )

    # --------------------------------------------------------
    # 12. Validation / test separation
    # --------------------------------------------------------

    validation_mask = (
        results["Data_Split"] == "Validation"
    )

    test_mask = (
        results["Data_Split"] == "Test"
    )

    train_mask = (
        results["Data_Split"] == "Train"
    )

    y_validation = results.loc[
        validation_mask,
        "Ground_Truth_Anomaly"
    ]

    validation_scores = results.loc[
        validation_mask,
        "Hybrid_Score"
    ]

    # --------------------------------------------------------
    # 13. Optimize threshold ONLY on validation set
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("VALIDATION THRESHOLD OPTIMIZATION")
    print("=" * 60)

    threshold_values = np.arange(
        0.10,
        0.96,
        0.02
    )

    threshold_results = []

    for threshold in threshold_values:

        validation_prediction = (
            validation_scores >= threshold
        ).astype(int)

        metrics = calculate_metrics(
            y_validation,
            validation_prediction
        )

        threshold_results.append(
            {
                "Threshold": threshold,
                **metrics,
            }
        )

    threshold_df = pd.DataFrame(
        threshold_results
    )

    # Sort by F1.
    threshold_df = threshold_df.sort_values(
        by=[
            "F1_Score",
            "Precision",
            "Recall",
        ],
        ascending=False
    ).reset_index(drop=True)

    best_threshold = float(
        threshold_df.loc[
            0,
            "Threshold"
        ]
    )

    best_validation_metrics = (
        threshold_df.iloc[0].to_dict()
    )

    # --------------------------------------------------------
    # 14. Save threshold optimization
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(THRESHOLD_FILE),
        exist_ok=True
    )

    threshold_df.to_csv(
        THRESHOLD_FILE,
        index=False
    )

    print(
        f"\nBest validation threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Validation F1: "
        f"{best_validation_metrics['F1_Score']:.4f}"
    )

    print(
        f"Validation Precision: "
        f"{best_validation_metrics['Precision']:.4f}"
    )

    print(
        f"Validation Recall: "
        f"{best_validation_metrics['Recall']:.4f}"
    )

    # --------------------------------------------------------
    # 15. Apply selected threshold
    # --------------------------------------------------------

    results["Smart_Hybrid_V2_Anomaly"] = (
        results["Hybrid_Score"] >= best_threshold
    ).astype(int)

    # --------------------------------------------------------
    # 16. Confidence level
    # --------------------------------------------------------

    results["V2_Confidence"] = pd.cut(
        results["Hybrid_Score"],
        bins=[
            -np.inf,
            0.20,
            0.40,
            0.60,
            0.80,
            np.inf,
        ],
        labels=[
            "Very Low",
            "Low",
            "Medium",
            "High",
            "Very High",
        ],
        right=False,
    )

    # --------------------------------------------------------
    # 17. Evaluate validation
    # --------------------------------------------------------

    validation_prediction = results.loc[
        validation_mask,
        "Smart_Hybrid_V2_Anomaly"
    ]

    validation_metrics = calculate_metrics(
        y_validation,
        validation_prediction
    )

    print_metrics(
        "VALIDATION PERFORMANCE",
        validation_metrics
    )

    # --------------------------------------------------------
    # 18. Evaluate TEST set
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # The test set is NOT used to choose the threshold.
    #
    # This is our final unbiased evaluation.
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)

    y_test = results.loc[
        test_mask,
        "Ground_Truth_Anomaly"
    ]

    test_prediction = results.loc[
        test_mask,
        "Smart_Hybrid_V2_Anomaly"
    ]

    test_metrics = calculate_metrics(
        y_test,
        test_prediction
    )

    print_metrics(
        "SMART HYBRID V2 - TEST SET",
        test_metrics
    )

    # --------------------------------------------------------
    # 19. Evaluate individual ML models on test set
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("INDIVIDUAL MODEL TEST PERFORMANCE")
    print("=" * 60)

    model_test_results = []

    y_test_array = y_test.values

    for model_name, prediction_column in [
        (
            "Isolation Forest",
            "IF_Prediction"
        ),
        (
            "LOF",
            "LOF_Prediction"
        ),
        (
            "One-Class SVM",
            "SVM_Prediction"
        ),
    ]:

        prediction = results.loc[
            test_mask,
            prediction_column
        ].values

        metrics = calculate_metrics(
            y_test_array,
            prediction
        )

        model_test_results.append(
            {
                "Model": model_name,
                **metrics,
            }
        )

        print_metrics(
            model_name,
            metrics
        )

    # Add Smart Hybrid V2
    model_test_results.append(
        {
            "Model": "Smart Hybrid V2",
            **test_metrics,
        }
    )

    # --------------------------------------------------------
    # 20. Add DQ-only test performance
    # --------------------------------------------------------

    dq_prediction = (
        results.loc[
            test_mask,
            "DQ_Evidence"
        ] > 0
    ).astype(int).values

    dq_metrics = calculate_metrics(
        y_test_array,
        dq_prediction
    )

    model_test_results.insert(
        0,
        {
            "Model": "DQ Only",
            **dq_metrics,
        }
    )

    # --------------------------------------------------------
    # 21. Create model comparison
    # --------------------------------------------------------

    comparison_df = pd.DataFrame(
        model_test_results
    )

    comparison_file = (
        "results/metrics/v2/"
        "v2_model_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_file,
        index=False
    )

    # --------------------------------------------------------
    # 22. Error-type predictions
    # --------------------------------------------------------
    #
    # These are evidence classifications, not separate
    # trained classifiers.
    # --------------------------------------------------------

    def classify_error(row):

        if row["DQ_Missing_Value"] == 1:
            return "Missing Value"

        if row["DQ_Invalid_Value"] == 1:
            return "Invalid Value"

        if row["DQ_Outlier"] == 1:
            return "Outlier"

        if row["DQ_Inconsistency"] == 1:
            return "Inconsistency"

        if row["DQ_Duplicate"] == 1:
            return "Duplicate"

        if row["Smart_Hybrid_V2_Anomaly"] == 1:
            return "ML Anomaly"

        return "Normal"

    results["V2_Error_Type"] = results.apply(
        classify_error,
        axis=1
    )

    # --------------------------------------------------------
    # 23. Save final results
    # --------------------------------------------------------

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # 24. Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SMART HYBRID V2 COMPLETED")
    print("=" * 60)

    print(
        f"\nSelected validation threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"\nFinal TEST F1: "
        f"{test_metrics['F1_Score']:.4f}"
    )

    print(
        f"Final TEST Precision: "
        f"{test_metrics['Precision']:.4f}"
    )

    print(
        f"Final TEST Recall: "
        f"{test_metrics['Recall']:.4f}"
    )

    print(
        f"Final TEST Accuracy: "
        f"{test_metrics['Accuracy']:.4f}"
    )

    print(
        f"\nV2 anomaly predictions: "
        f"{results['Smart_Hybrid_V2_Anomaly'].sum():,}"
    )

    print(
        "\nOutput files:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    print(
        f"  {THRESHOLD_FILE}"
    )

    print(
        f"  {comparison_file}"
    )

    print(
        "\nSmart Hybrid V2 is ready."
    )


if __name__ == "__main__":
    main()