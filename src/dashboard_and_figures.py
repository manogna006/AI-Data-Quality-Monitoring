"""
AI Data Quality Monitoring
V5.7 Dashboard + Publication Figure Generator

Purpose
-------
1. Load FINAL V5.7 results.
2. Build an interactive Streamlit dashboard.
3. Generate publication-quality PNG figures.
4. Compute ROC-AUC and PR-AUC from actual V5.7 test probabilities.
5. Display human-readable error explanations.
6. Compare V2 -> V5.7 model progression.

IMPORTANT
---------
This script does NOT retrain any model.
It only uses saved V5.7 outputs and ground truth.

Run dashboard:
    streamlit run src/dashboard_and_figures.py

Figures are saved to:
    paper/figures/
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

V57_DIR = PROJECT_ROOT / "results" / "metrics" / "v5_7"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FILES
# ============================================================

PREDICTIONS_FILE = V57_DIR / "v5_7_test_predictions.csv"
RESULTS_FILE = V57_DIR / "type_aware_hybrid_v5_7_results.csv"
OVERALL_FILE = V57_DIR / "v5_7_overall_performance.csv"
ERROR_PERFORMANCE_FILE = V57_DIR / "v5_7_error_type_performance.csv"
EXPLANATIONS_FILE = V57_DIR / "v5_7_error_explanations.csv"
FEATURE_IMPORTANCE_FILE = V57_DIR / "v5_7_random_forest_feature_importance.csv"
AUDIT_FILE = V57_DIR / "v5_7_methodology_audit.csv"

GROUND_TRUTH_FILE = METRICS_DIR / "ground_truth.csv"


# ============================================================
# MODEL PROGRESSION
# ============================================================

MODEL_PROGRESSION = pd.DataFrame(
    [
        ["V2", 90.25, 29.21, 48.32, 36.41],
        ["V3", 97.01, 67.43, 93.26, 78.27],
        ["V4", 98.00, 77.99, 91.09, 84.03],
        ["V5.3", 98.67, 94.28, 82.73, 88.13],
        ["V5.4", 98.87, 88.08, 93.66, 90.78],
        ["V5.5", 98.93, 88.94, 93.66, 91.24],
        ["V5.6", 97.99, 92.26, 72.26, 81.05],
        ["V5.7", 98.90, 90.29, 91.34, 90.81],
    ],
    columns=["Model", "Accuracy", "Precision", "Recall", "F1"],
)


# ============================================================
# ERROR TYPE PERFORMANCE
# ============================================================

DEFAULT_ERROR_PERFORMANCE = pd.DataFrame(
    [
        ["Missing Value", 74.70],
        ["Invalid Value", 99.24],
        ["Outlier", 95.06],
        ["Inconsistency", 100.00],
        ["Duplicate", 87.90],
    ],
    columns=["Error_Type", "F1"],
)


# ============================================================
# LOAD FUNCTIONS
# ============================================================

def load_csv(path, required=False):
    """
    Load CSV safely.
    """
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")
        return None

    return pd.read_csv(path)


def load_all_data():
    """
    Load all V5.7 artifacts.
    """

    predictions = load_csv(PREDICTIONS_FILE, required=True)
    results = load_csv(RESULTS_FILE)
    overall = load_csv(OVERALL_FILE)
    error_performance = load_csv(ERROR_PERFORMANCE_FILE)
    explanations = load_csv(EXPLANATIONS_FILE)
    feature_importance = load_csv(FEATURE_IMPORTANCE_FILE)
    audit = load_csv(AUDIT_FILE)
    ground_truth = load_csv(GROUND_TRUTH_FILE, required=True)

    return {
        "predictions": predictions,
        "results": results,
        "overall": overall,
        "error_performance": error_performance,
        "explanations": explanations,
        "feature_importance": feature_importance,
        "audit": audit,
        "ground_truth": ground_truth,
    }


# ============================================================
# COLUMN HELPERS
# ============================================================

def find_column(df, candidates):
    """
    Find the first matching column from candidates.
    """
    if df is None:
        return None

    normalized = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in normalized:
            return normalized[key]

    # Partial match
    for candidate in candidates:
        key = candidate.strip().lower()

        for normalized_name, original_name in normalized.items():
            if key in normalized_name:
                return original_name

    return None


# ============================================================
# PREPARE GROUND TRUTH
# ============================================================

def prepare_test_labels(predictions, ground_truth):
    """
    Align ground truth with V5.7 test predictions.

    The primary alignment key is Original_Row_ID.

    Duplicate Original_Row_ID values can occur because the corrupted
    dataset contains appended duplicate transactions. Therefore,
    when duplicate IDs exist, the first available ground-truth record
    is used.

    The resulting labels are used for ROC/PR visualization only.
    """

    gt = ground_truth.copy()
    pred = predictions.copy()

    original_id_pred = find_column(
        pred,
        ["Original_Row_ID", "Original Row ID"]
    )

    original_id_gt = find_column(
        gt,
        ["Original_Row_ID", "Original Row ID"]
    )

    if original_id_pred is None or original_id_gt is None:
        raise ValueError(
            "Original_Row_ID could not be found in predictions "
            "or ground truth."
        )

    # Remove duplicate ground-truth keys if present.
    gt_unique = gt.drop_duplicates(
        subset=[original_id_gt],
        keep="first"
    ).copy()

    label_columns = [
        "Missing_Value",
        "Invalid_Value",
        "Outlier",
        "Inconsistency",
        "Duplicate",
    ]

    available_labels = [
        c for c in label_columns
        if c in gt_unique.columns
    ]

    if not available_labels:
        raise ValueError(
            "No per-error ground-truth columns found."
        )

    gt_small = gt_unique[
        [original_id_gt] + available_labels
    ].copy()

    gt_small = gt_small.rename(
        columns={original_id_gt: "__Original_Row_ID__"}
    )

    pred_work = pred.copy()

    pred_work["__Original_Row_ID__"] = pred_work[
        original_id_pred
    ]

    merged = pred_work.merge(
        gt_small,
        on="__Original_Row_ID__",
        how="left",
        sort=False,
    )

    # Missing ground truth values become zero for alignment.
    for col in available_labels:
        merged[col] = (
            pd.to_numeric(
                merged[col],
                errors="coerce"
            )
            .fillna(0)
            .astype(int)
        )

    return merged, available_labels


# ============================================================
# CALCULATE OVERALL METRICS
# ============================================================

def calculate_overall_metrics(predictions):
    """
    Calculate final overall metrics from saved V5.7 predictions.
    """

    y_true = pd.to_numeric(
        predictions["Any_Error_True"],
        errors="coerce"
    ).fillna(0).astype(int)

    y_pred = pd.to_numeric(
        predictions["Any_Error_Prediction"],
        errors="coerce"
    ).fillna(0).astype(int)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    precision = tp / max(tp + fp, 1)

    recall = tp / max(tp + fn, 1)

    f1 = (
        2 * precision * recall /
        max(precision + recall, 1e-12)
    )

    return {
        "Accuracy": accuracy * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1": f1 * 100,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


# ============================================================
# FIND ERROR PERFORMANCE COLUMNS
# ============================================================

def prepare_error_performance(df):
    """
    Normalize saved error-type performance file.
    """

    if df is None or df.empty:
        return DEFAULT_ERROR_PERFORMANCE.copy()

    work = df.copy()

    type_col = find_column(
        work,
        [
            "Error_Type",
            "Error Type",
            "Type",
            "Error",
        ]
    )

    f1_col = find_column(
        work,
        [
            "F1",
            "F1_Score",
            "F1 Score",
        ]
    )

    if type_col is None or f1_col is None:
        return DEFAULT_ERROR_PERFORMANCE.copy()

    result = work[[type_col, f1_col]].copy()

    result.columns = ["Error_Type", "F1"]

    result["F1"] = pd.to_numeric(
        result["F1"],
        errors="coerce"
    )

    # Convert fractions to percentages if necessary.
    if result["F1"].max() <= 1.0:
        result["F1"] *= 100

    return result.dropna()


# ============================================================
# FIGURE 1: OVERALL PERFORMANCE
# ============================================================

def figure_overall_performance(metrics):
    """
    Publication Figure 1:
    Final V5.7 overall performance.
    """

    labels = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
    ]

    values = [
        metrics["Accuracy"],
        metrics["Precision"],
        metrics["Recall"],
        metrics["F1"],
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(labels, values)

    ax.set_ylim(0, 105)

    ax.set_ylabel("Score (%)")

    ax.set_title(
        "Overall Performance of the V5.7 Hybrid Framework"
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10
        )

    fig.tight_layout()

    path = FIGURES_DIR / "figure_1_overall_performance.png"

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    return path


# ============================================================
# FIGURE 2: ERROR TYPE F1
# ============================================================

def figure_error_type_performance(error_df):
    """
    Publication Figure 2:
    F1 score by error type.
    """

    work = error_df.copy()

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        work["Error_Type"],
        work["F1"]
    )

    ax.set_ylim(0, 105)

    ax.set_ylabel("F1 Score (%)")

    ax.set_title(
        "V5.7 Performance Across Data Quality Error Types"
    )

    ax.tick_params(axis="x", rotation=25)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    for bar, value in zip(
        bars,
        work["F1"]
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.2f}",
            ha="center",
            fontsize=9
        )

    fig.tight_layout()

    path = FIGURES_DIR / "figure_2_error_type_f1.png"

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    return path


# ============================================================
# FIGURE 3: CONFUSION MATRIX
# ============================================================

def figure_confusion_matrix(metrics):
    """
    Publication Figure 3:
    Overall binary confusion matrix.
    """

    cm = np.array(
        [
            [metrics["TN"], metrics["FP"]],
            [metrics["FN"], metrics["TP"]],
        ]
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(cm)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["Predicted Clean", "Predicted Error"]
    )

    ax.set_yticklabels(
        ["Actual Clean", "Actual Error"]
    )

    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")

    ax.set_title(
        "Confusion Matrix — V5.7"
    )

    for i in range(2):
        for j in range(2):

            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold"
            )

    fig.colorbar(image, ax=ax)

    fig.tight_layout()

    path = FIGURES_DIR / "figure_3_confusion_matrix.png"

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    return path


# ============================================================
# FIGURE 4: MODEL PROGRESSION
# ============================================================

def figure_model_progression():
    """
    Publication Figure 4:
    V2 -> V5.7 performance progression.
    """

    df = MODEL_PROGRESSION.copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        df["Model"],
        df["Accuracy"],
        marker="o",
        label="Accuracy"
    )

    ax.plot(
        df["Model"],
        df["Precision"],
        marker="o",
        label="Precision"
    )

    ax.plot(
        df["Model"],
        df["Recall"],
        marker="o",
        label="Recall"
    )

    ax.plot(
        df["Model"],
        df["F1"],
        marker="o",
        label="F1"
    )

    ax.set_ylim(0, 105)

    ax.set_ylabel("Score (%)")

    ax.set_xlabel("Model Version")

    ax.set_title(
        "Performance Progression from V2 to V5.7"
    )

    ax.grid(
        linestyle="--",
        alpha=0.3
    )

    ax.legend()

    fig.tight_layout()

    path = FIGURES_DIR / "figure_4_model_progression.png"

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    return path


# ============================================================
# FIGURE 5: ERROR DISTRIBUTION
# ============================================================

def figure_error_distribution(predictions):
    """
    Publication Figure 5:
    Number of predicted errors by type.
    """

    types = [
        "Missing_Value_Prediction",
        "Invalid_Value_Prediction",
        "Outlier_Prediction",
        "Inconsistency_Prediction",
        "Duplicate_Prediction",
    ]

    names = [
        "Missing",
        "Invalid",
        "Outlier",
        "Inconsistency",
        "Duplicate",
    ]

    counts = []

    for col in types:
        if col in predictions.columns:
            counts.append(
                int(
                    pd.to_numeric(
                        predictions[col],
                        errors="coerce"
                    )
                    .fillna(0)
                    .sum()
                )
            )
        else:
            counts.append(0)

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(names, counts)

    ax.set_ylabel("Predicted Error Rows")

    ax.set_title(
        "Distribution of V5.7 Predicted Error Types"
    )

    ax.tick_params(axis="x", rotation=20)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    for bar, value in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(counts) * 0.01,
            f"{value:,}",
            ha="center",
            fontsize=9
        )

    fig.tight_layout()

    path = FIGURES_DIR / "figure_5_error_distribution.png"

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    return path


# ============================================================
# FIGURE 6: ROC CURVES
# ============================================================

def figure_roc_curves(aligned):
    """
    Publication Figure 6:
    ROC curves for individual error types.

    Uses actual V5.7 probability outputs and aligned
    ground-truth labels.
    """

    mapping = {
        "Missing_Value": "Missing_Value_Probability",
        "Invalid_Value": "Invalid_Value_Probability",
        "Outlier": "Outlier_Probability",
        "Inconsistency": "Inconsistency_Probability",
        "Duplicate": "Duplicate_Probability",
    }

    fig, ax = plt.subplots(figsize=(8, 7))

    auc_values = {}

    for error_type, prob_col in mapping.items():

        if error_type not in aligned.columns:
            continue

        if prob_col not in aligned.columns:
            continue

        y_true = aligned[error_type].astype(int)

        y_score = pd.to_numeric(
            aligned[prob_col],
            errors="coerce"
        ).fillna(0)

        # ROC-AUC requires both classes.
        if y_true.nunique() < 2:
            continue

        fpr, tpr, _ = roc_curve(
            y_true,
            y_score
        )

        auc = roc_auc_score(
            y_true,
            y_score
        )

        auc_values[error_type] = auc

        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{error_type} (AUC={auc:.3f})"
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1
    )

    ax.set_xlabel("False Positive Rate")

    ax.set_ylabel("True Positive Rate")

    ax.set_title(
        "ROC Curves for V5.7 Error-Type Detection"
    )

    ax.legend(
        loc="lower right",
        fontsize=8
    )

    ax.grid(
        linestyle="--",
        alpha=0.3
    )

    fig.tight_layout()

    path = FIGURES_DIR / "figure_6_roc_curves.png"

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    return path, auc_values


# ============================================================
# FIGURE 7: PRECISION-RECALL CURVES
# ============================================================

def figure_pr_curves(aligned):
    """
    Publication Figure 7:
    Precision-Recall curves.
    """

    mapping = {
        "Missing_Value": "Missing_Value_Probability",
        "Invalid_Value": "Invalid_Value_Probability",
        "Outlier": "Outlier_Probability",
        "Inconsistency": "Inconsistency_Probability",
        "Duplicate": "Duplicate_Probability",
    }

    fig, ax = plt.subplots(figsize=(8, 7))

    ap_values = {}

    for error_type, prob_col in mapping.items():

        if error_type not in aligned.columns:
            continue

        if prob_col not in aligned.columns:
            continue

        y_true = aligned[error_type].astype(int)

        y_score = pd.to_numeric(
            aligned[prob_col],
            errors="coerce"
        ).fillna(0)

        if y_true.nunique() < 2:
            continue

        precision, recall, _ = precision_recall_curve(
            y_true,
            y_score
        )

        ap = average_precision_score(
            y_true,
            y_score
        )

        ap_values[error_type] = ap

        ax.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{error_type} (AP={ap:.3f})"
        )

    ax.set_xlabel("Recall")

    ax.set_ylabel("Precision")

    ax.set_title(
        "Precision–Recall Curves for V5.7"
    )

    ax.legend(
        loc="lower left",
        fontsize=8
    )

    ax.grid(
        linestyle="--",
        alpha=0.3
    )

    fig.tight_layout()

    path = FIGURES_DIR / "figure_7_precision_recall_curves.png"

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    return path, ap_values


# ============================================================
# FIGURE 8: FEATURE IMPORTANCE
# ============================================================

def figure_feature_importance(feature_importance):
    """
    Publication Figure 8:
    Top 20 RF features.
    """

    if (
        feature_importance is None
        or feature_importance.empty
    ):
        return None

    df = feature_importance.copy()

    feature_col = find_column(
        df,
        [
            "Feature",
            "Feature_Name",
            "Feature Name",
        ]
    )

    importance_col = find_column(
        df,
        [
            "Importance",
            "Mean_Importance",
            "Feature_Importance",
        ]
    )

    if feature_col is None or importance_col is None:
        return None

    df[importance_col] = pd.to_numeric(
        df[importance_col],
        errors="coerce"
    )

    df = (
        df.dropna(
            subset=[importance_col]
        )
        .sort_values(
            importance_col,
            ascending=False
        )
        .head(20)
        .sort_values(
            importance_col,
            ascending=True
        )
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.barh(
        df[feature_col].astype(str),
        df[importance_col]
    )

    ax.set_xlabel("Random Forest Importance")

    ax.set_title(
        "Top 20 V5.7 Model Features"
    )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.3
    )

    fig.tight_layout()

    path = FIGURES_DIR / "figure_8_feature_importance.png"

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    return path


# ============================================================
# GENERATE ALL FIGURES
# ============================================================

def generate_all_figures(data):
    """
    Generate all paper figures.
    """

    predictions = data["predictions"]

    metrics = calculate_overall_metrics(
        predictions
    )

    error_df = prepare_error_performance(
        data["error_performance"]
    )

    aligned, labels = prepare_test_labels(
        predictions,
        data["ground_truth"]
    )

    generated = []

    generated.append(
        figure_overall_performance(metrics)
    )

    generated.append(
        figure_error_type_performance(error_df)
    )

    generated.append(
        figure_confusion_matrix(metrics)
    )

    generated.append(
        figure_model_progression()
    )

    generated.append(
        figure_error_distribution(predictions)
    )

    roc_path, roc_auc = figure_roc_curves(
        aligned
    )

    generated.append(roc_path)

    pr_path, pr_auc = figure_pr_curves(
        aligned
    )

    generated.append(pr_path)

    feature_path = figure_feature_importance(
        data["feature_importance"]
    )

    if feature_path is not None:
        generated.append(feature_path)

    # Save actual AUC values.
    auc_rows = []

    for error_type, value in roc_auc.items():
        auc_rows.append(
            [
                error_type,
                value
            ]
        )

    auc_df = pd.DataFrame(
        auc_rows,
        columns=["Error_Type", "ROC_AUC"]
    )

    auc_df.to_csv(
        V57_DIR / "v5_7_roc_auc.csv",
        index=False
    )

    ap_rows = []

    for error_type, value in pr_auc.items():
        ap_rows.append(
            [
                error_type,
                value
            ]
        )

    ap_df = pd.DataFrame(
        ap_rows,
        columns=["Error_Type", "PR_AUC"]
    )

    ap_df.to_csv(
        V57_DIR / "v5_7_pr_auc.csv",
        index=False
    )

    return (
        metrics,
        error_df,
        aligned,
        roc_auc,
        pr_auc,
        generated
    )


# ============================================================
# STREAMLIT DASHBOARD
# ============================================================

def run_dashboard(
    data,
    metrics,
    error_df,
    aligned,
    roc_auc,
    pr_auc,
):
    """
    Interactive Streamlit dashboard.
    """

    import streamlit as st

    st.set_page_config(
        page_title="AI Data Quality Monitoring — V5.7",
        page_icon="📊",
        layout="wide",
    )

    st.title(
        "AI-Based Data Quality Monitoring"
    )

    st.subheader(
        "Type-Aware Hybrid Framework — V5.7"
    )

    st.caption(
        "Rule-based Data Quality + Unsupervised Anomaly Detection "
        "+ Type-Specific Machine Learning"
    )

    # --------------------------------------------------------
    # KPI SECTION
    # --------------------------------------------------------

    st.markdown("## Overall Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Accuracy",
        f"{metrics['Accuracy']:.2f}%"
    )

    c2.metric(
        "Precision",
        f"{metrics['Precision']:.2f}%"
    )

    c3.metric(
        "Recall",
        f"{metrics['Recall']:.2f}%"
    )

    c4.metric(
        "F1 Score",
        f"{metrics['F1']:.2f}%"
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX VALUES
    # --------------------------------------------------------

    st.markdown("## Confusion Matrix")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "True Positives",
        f"{metrics['TP']:,}"
    )

    c2.metric(
        "False Positives",
        f"{metrics['FP']:,}"
    )

    c3.metric(
        "True Negatives",
        f"{metrics['TN']:,}"
    )

    c4.metric(
        "False Negatives",
        f"{metrics['FN']:,}"
    )

    # --------------------------------------------------------
    # ERROR TYPE PERFORMANCE
    # --------------------------------------------------------

    st.markdown("## Error-Type Performance")

    st.dataframe(
        error_df.style.format(
            {"F1": "{:.2f}%"}
        ),
        use_container_width=True,
    )

    st.bar_chart(
        error_df.set_index(
            "Error_Type"
        )["F1"]
    )

    # --------------------------------------------------------
    # MODEL PROGRESSION
    # --------------------------------------------------------

    st.markdown("## Model Progression")

    st.dataframe(
        MODEL_PROGRESSION.style.format(
            {
                "Accuracy": "{:.2f}%",
                "Precision": "{:.2f}%",
                "Recall": "{:.2f}%",
                "F1": "{:.2f}%",
            }
        ),
        use_container_width=True,
    )

    st.line_chart(
        MODEL_PROGRESSION.set_index(
            "Model"
        )[
            [
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
            ]
        ]
    )

    # --------------------------------------------------------
    # ROC AUC
    # --------------------------------------------------------

    st.markdown("## ROC-AUC")

    if roc_auc:
        roc_df = pd.DataFrame(
            [
                [k, v]
                for k, v in roc_auc.items()
            ],
            columns=[
                "Error Type",
                "ROC-AUC"
            ]
        )

        st.dataframe(
            roc_df.style.format(
                {"ROC-AUC": "{:.4f}"}
            ),
            use_container_width=True,
        )

    # --------------------------------------------------------
    # PR-AUC
    # --------------------------------------------------------

    st.markdown("## Precision–Recall AUC")

    if pr_auc:
        pr_df = pd.DataFrame(
            [
                [k, v]
                for k, v in pr_auc.items()
            ],
            columns=[
                "Error Type",
                "PR-AUC"
            ]
        )

        st.dataframe(
            pr_df.style.format(
                {"PR-AUC": "{:.4f}"}
            ),
            use_container_width=True,
        )

    # --------------------------------------------------------
    # ERROR DISTRIBUTION
    # --------------------------------------------------------

    st.markdown("## Predicted Error Distribution")

    prediction_columns = {
        "Missing Value": "Missing_Value_Prediction",
        "Invalid Value": "Invalid_Value_Prediction",
        "Outlier": "Outlier_Prediction",
        "Inconsistency": "Inconsistency_Prediction",
        "Duplicate": "Duplicate_Prediction",
    }

    distribution = {}

    for name, column in prediction_columns.items():

        if column in data["predictions"].columns:
            distribution[name] = int(
                pd.to_numeric(
                    data["predictions"][column],
                    errors="coerce"
                )
                .fillna(0)
                .sum()
            )

    if distribution:

        dist_df = pd.DataFrame(
            list(distribution.items()),
            columns=[
                "Error Type",
                "Predicted Rows"
            ]
        )

        st.bar_chart(
            dist_df.set_index(
                "Error Type"
            )
        )

    # --------------------------------------------------------
    # EXPLANATIONS
    # --------------------------------------------------------

    st.markdown("## Human-Readable Error Explanations")

    explanations = data["explanations"]

    if explanations is not None and not explanations.empty:

        st.write(
            f"Total predicted error rows: "
            f"**{len(explanations):,}**"
        )

        display_df = explanations.copy()

        search = st.text_input(
            "Search explanations",
            ""
        )

        if search:

            mask = display_df.astype(str).apply(
                lambda col: col.str.contains(
                    search,
                    case=False,
                    na=False
                )
            ).any(axis=1)

            display_df = display_df[mask]

        st.dataframe(
            display_df.head(500),
            use_container_width=True,
        )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.markdown("## Feature Importance")

    fi = data["feature_importance"]

    if fi is not None and not fi.empty:

        feature_col = find_column(
            fi,
            [
                "Feature",
                "Feature_Name",
                "Feature Name",
            ]
        )

        importance_col = find_column(
            fi,
            [
                "Importance",
                "Mean_Importance",
                "Feature_Importance",
            ]
        )

        if feature_col and importance_col:

            fi_display = fi.copy()

            fi_display[importance_col] = pd.to_numeric(
                fi_display[importance_col],
                errors="coerce"
            )

            fi_display = (
                fi_display
                .dropna(
                    subset=[importance_col]
                )
                .sort_values(
                    importance_col,
                    ascending=False
                )
                .head(20)
            )

            st.dataframe(
                fi_display,
                use_container_width=True,
            )

    # --------------------------------------------------------
    # METHODOLOGY AUDIT
    # --------------------------------------------------------

    st.markdown("## Methodology Audit")

    audit = data["audit"]

    if audit is not None and not audit.empty:

        st.dataframe(
            audit,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    st.markdown("---")

    st.caption(
        "V5.7 is the final project benchmark. "
        "The dashboard uses saved outputs only and does not retrain models."
    )


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Main execution.

    If Streamlit is running this file, the dashboard is shown.
    Otherwise, figures are generated from the command line.
    """

    print("=" * 70)
    print("AI DATA QUALITY MONITORING — V5.7")
    print("Dashboard + Publication Figure Generator")
    print("=" * 70)

    print("\nLoading V5.7 results...")

    data = load_all_data()

    predictions = data["predictions"]

    print(
        f"Prediction rows : {len(predictions):,}"
    )

    metrics = calculate_overall_metrics(
        predictions
    )

    print("\nFinal V5.7 Metrics")
    print("-" * 40)

    print(
        f"Accuracy  : {metrics['Accuracy']:.2f}%"
    )

    print(
        f"Precision : {metrics['Precision']:.2f}%"
    )

    print(
        f"Recall    : {metrics['Recall']:.2f}%"
    )

    print(
        f"F1 Score  : {metrics['F1']:.2f}%"
    )

    print("\nGenerating publication figures...")

    (
        metrics,
        error_df,
        aligned,
        roc_auc,
        pr_auc,
        generated,
    ) = generate_all_figures(data)

    print("\nFigures generated:")

    for path in generated:

        if path is not None:
            print(
                f"  ✓ {path.relative_to(PROJECT_ROOT)}"
            )

    print("\nActual ROC-AUC values:")

    for error_type, value in roc_auc.items():
        print(
            f"  {error_type:<18} {value:.4f}"
        )

    print("\nActual PR-AUC values:")

    for error_type, value in pr_auc.items():
        print(
            f"  {error_type:<18} {value:.4f}"
        )

    print("\n" + "=" * 70)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 70)

    print(
        f"\nPaper figures saved to:\n"
        f"{FIGURES_DIR}"
    )

    print(
        "\nTo launch the dashboard:"
    )

    print(
        "streamlit run src/dashboard_and_figures.py"
    )

    # --------------------------------------------------------
    # STREAMLIT DASHBOARD
    # --------------------------------------------------------

    try:

        import streamlit as st

        # Streamlit automatically runs the script.
        # Display dashboard only when running under Streamlit.

        if st.runtime.exists():

            run_dashboard(
                data,
                metrics,
                error_df,
                aligned,
                roc_auc,
                pr_auc,
            )

    except Exception:
        pass


if __name__ == "__main__":
    main()