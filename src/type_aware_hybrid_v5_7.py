# ============================================================
# TYPE-AWARE HYBRID V5.7
# FINAL DUPLICATE-OPTIMIZED VERSION
#
# Leakage-safe type-aware Random Forest
#
# V5.7 changes from V5.6:
#   - Keeps train-only contextual features
#   - Keeps validation-only threshold optimization
#   - Keeps identifier removal
#   - Keeps Customer ID out of model features
#   - Keeps synthetic artifact removal
#   - Keeps specialized Missing detection
#   - IMPROVES DUPLICATE DETECTION USING A LABEL-FREE
#     DATASET-LEVEL DUPLICATE SIGNATURE
#
# Important:
# Duplicate signatures are unsupervised DQ evidence.
# They do NOT use ground-truth labels.
#
# ============================================================

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
    confusion_matrix,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

V2_RESULTS_PATH = os.path.join(
    BASE_DIR,
    "results",
    "metrics",
    "v2",
    "model_scores_v2.csv",
)

V2_FEATURES_PATH = os.path.join(
    BASE_DIR,
    "results",
    "metrics",
    "v2",
    "ml_features_v2.csv",
)

GROUND_TRUTH_PATH = os.path.join(
    BASE_DIR,
    "results",
    "metrics",
    "ground_truth.csv",
)

RAW_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "corrupted",
    "online_retail_corrupted.csv",
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "metrics",
    "v5_7",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# UTILITIES
# ============================================================

def safe_numeric(series):

    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]

    return pd.to_numeric(
        series,
        errors="coerce"
    )


def safe_series(df, column):

    if column not in df.columns:
        return pd.Series(
            np.nan,
            index=df.index
        )

    obj = df.loc[:, column]

    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]

    return obj


def clean_after_merge(df, name):

    duplicate_columns = (
        df.columns[
            df.columns.duplicated()
        ].tolist()
    )

    if duplicate_columns:

        print(
            f"Duplicate columns detected after {name}: "
            f"{sorted(set(duplicate_columns))}"
        )

        result = pd.DataFrame(
            index=df.index
        )

        processed = set()

        for column in df.columns:

            if column in processed:
                continue

            same = df.loc[
                :,
                df.columns == column
            ]

            if same.shape[1] == 1:

                result[column] = (
                    same.iloc[:, 0]
                )

            else:

                result[column] = (
                    same.bfill(
                        axis=1
                    ).iloc[:, 0]
                )

            processed.add(column)

        df = result

    if df.columns.duplicated().any():

        raise RuntimeError(
            f"Duplicate columns remain after {name}."
        )

    return df


def make_binary_label(series):

    return (
        safe_numeric(series)
        .fillna(0)
        .astype(int)
        .clip(0, 1)
    )


def choose_threshold(
    y_true,
    probabilities
):

    best_threshold = 0.50
    best_f1 = -1

    for threshold in np.arange(
        0.01,
        1.00,
        0.01
    ):

        prediction = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            prediction,
            zero_division=0
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = threshold

    return (
        float(best_threshold),
        float(best_f1)
    )


def evaluate(
    y_true,
    probability,
    threshold
):

    prediction = (
        probability >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        prediction
    )

    precision = precision_score(
        y_true,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        prediction,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1]
    ).ravel()

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "Prediction": prediction,
    }


# ============================================================
# DUPLICATE SIGNATURE
# ============================================================

def create_duplicate_signature(df):

    columns = [
        "Invoice",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Customer ID",
        "Country",
    ]

    available = [
        c
        for c in columns
        if c in df.columns
    ]

    if not available:

        return pd.Series(
            "",
            index=df.index
        )

    temp = df[
        available
    ].copy()

    for column in available:

        if pd.api.types.is_datetime64_any_dtype(
            temp[column]
        ):

            temp[column] = (
                temp[column]
                .astype(str)
            )

        else:

            temp[column] = (
                temp[column]
                .fillna("<NA>")
                .astype(str)
            )

    return (
        temp
        .astype(str)
        .agg(
            "||".join,
            axis=1
        )
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 72)
    print(
        "TYPE-AWARE HYBRID V5.7"
    )
    print(
        "FINAL DUPLICATE-OPTIMIZED VERSION"
    )
    print(
        "STRICT LEAKAGE CONTROL + UNSUPERVISED DUPLICATE SIGNAL"
    )
    print("=" * 72)

    # ========================================================
    # LOAD DATA
    # ========================================================

    print()
    print("Loading datasets...")

    v2_results = pd.read_csv(
        V2_RESULTS_PATH
    )

    v2_features = pd.read_csv(
        V2_FEATURES_PATH
    )

    ground_truth = pd.read_csv(
        GROUND_TRUTH_PATH
    )

    raw = pd.read_csv(
        RAW_PATH,
        low_memory=False
    )

    print(
        f"V2 results       : {len(v2_results)}"
    )

    print(
        f"V2 features      : {len(v2_features)}"
    )

    print(
        f"Ground truth     : {len(ground_truth)}"
    )

    print(
        f"Raw data         : {len(raw)}"
    )

    # ========================================================
    # CLEAN DUPLICATE COLUMN NAMES
    # ========================================================

    raw = clean_after_merge(
        raw,
        "raw loading"
    )

    v2_features = clean_after_merge(
        v2_features,
        "V2 features loading"
    )

    v2_results = clean_after_merge(
        v2_results,
        "V2 results loading"
    )

    ground_truth = clean_after_merge(
        ground_truth,
        "ground truth loading"
    )

    # ========================================================
    # CREATE CORRUPTED ROW ID
    # ========================================================

    if (
        "Corrupted_Row_ID"
        not in v2_features.columns
    ):

        v2_features[
            "Corrupted_Row_ID"
        ] = np.arange(
            len(v2_features)
        )

    if (
        "Corrupted_Row_ID"
        not in v2_results.columns
    ):

        v2_results[
            "Corrupted_Row_ID"
        ] = np.arange(
            len(v2_results)
        )

    if (
        "Corrupted_Row_ID"
        not in raw.columns
    ):

        print()
        print(
            "Raw CSV does not contain "
            "Corrupted_Row_ID."
        )

        raw[
            "Corrupted_Row_ID"
        ] = np.arange(
            len(raw)
        )

        print(
            "Creating from raw row order..."
        )

    print(
        "Unique feature rows:",
        v2_features[
            "Corrupted_Row_ID"
        ].nunique()
    )

    # ========================================================
    # MERGE V2 RESULTS
    # ========================================================

    result_columns = [
        c
        for c in v2_results.columns
        if c != "Corrupted_Row_ID"
    ]

    result_columns = [
        c
        for c in result_columns
        if c not in v2_features.columns
    ]

    result_subset = v2_results[
        ["Corrupted_Row_ID"]
        + result_columns
    ].copy()

    df = v2_features.merge(
        result_subset,
        on="Corrupted_Row_ID",
        how="left",
        validate="one_to_one"
    )

    df = clean_after_merge(
        df,
        "V2 result merge"
    )

    # ========================================================
    # GROUND TRUTH MERGE
    # ========================================================

    if (
        "Original_Row_ID"
        not in df.columns
    ):

        if (
            "Original_Row_ID"
            in raw.columns
        ):

            id_map = raw[
                [
                    "Corrupted_Row_ID",
                    "Original_Row_ID",
                ]
            ].drop_duplicates(
                "Corrupted_Row_ID"
            )

            df = df.merge(
                id_map,
                on="Corrupted_Row_ID",
                how="left",
                validate="one_to_one"
            )

        else:

            raise RuntimeError(
                "Original_Row_ID missing."
            )

    gt_columns = [
        "Original_Row_ID",
        "Missing_Value",
        "Invalid_Value",
        "Outlier",
        "Inconsistency",
        "Duplicate",
    ]

    gt = ground_truth[
        gt_columns
    ].drop_duplicates(
        "Original_Row_ID"
    )

    df = df.merge(
        gt,
        on="Original_Row_ID",
        how="left",
        validate="many_to_one"
    )

    df = clean_after_merge(
        df,
        "ground-truth merge"
    )

    print(
        f"After ground-truth merge: {len(df)}"
    )

    # ========================================================
    # RAW CONTEXT MERGE
    # ========================================================

    raw_columns = [
        "Corrupted_Row_ID",
        "Invoice",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Customer ID",
        "Country",
    ]

    raw_columns = [
        c
        for c in raw_columns
        if c in raw.columns
    ]

    raw_context = raw[
        raw_columns
    ].copy()

    # Only add columns not already present.
    add_columns = [
        c
        for c in raw_context.columns
        if (
            c == "Corrupted_Row_ID"
            or c not in df.columns
        )
    ]

    raw_context = raw_context[
        add_columns
    ]

    print(
        "Raw context columns prepared."
    )

    df = df.merge(
        raw_context,
        on="Corrupted_Row_ID",
        how="left",
        validate="one_to_one"
    )

    df = clean_after_merge(
        df,
        "raw context merge"
    )

    print(
        f"After raw context merge: {len(df)}"
    )

    # ========================================================
    # MERGE INTEGRITY
    # ========================================================

    if len(df) != len(v2_features):

        raise RuntimeError(
            "Row count changed unexpectedly."
        )

    if df[
        "Corrupted_Row_ID"
    ].duplicated().any():

        raise RuntimeError(
            "Corrupted_Row_ID is not unique."
        )

    print(
        "Merge integrity verified."
    )

    # ========================================================
    # RAW NUMERIC FEATURES
    # ========================================================

    for column in [
        "Quantity",
        "Price",
        "Customer ID",
    ]:

        if column in df.columns:

            df[column] = safe_numeric(
                df[column]
            )

    # ========================================================
    # DATE FEATURES
    # ========================================================

    if "InvoiceDate" in df.columns:

        dt = pd.to_datetime(
            df["InvoiceDate"],
            errors="coerce"
        )

        df["V57_Year"] = (
            dt.dt.year
            .fillna(0)
        )

        df["V57_Month"] = (
            dt.dt.month
            .fillna(0)
        )

        df["V57_Day"] = (
            dt.dt.day
            .fillna(0)
        )

        df["V57_Weekday"] = (
            dt.dt.weekday
            .fillna(-1)
        )

        df["V57_Hour"] = (
            dt.dt.hour
            .fillna(0)
        )

    # ========================================================
    # QUANTITY FEATURES
    # ========================================================

    if "Quantity" in df.columns:

        quantity = safe_numeric(
            df["Quantity"]
        )

        df["V57_Quantity_Abs"] = (
            quantity.abs()
        )

        df["V57_Quantity_LogAbs"] = (
            np.log1p(
                quantity.abs()
            )
        )

        df["V57_Quantity_Zero"] = (
            quantity == 0
        ).astype(int)

        df["V57_Quantity_Negative"] = (
            quantity < 0
        ).astype(int)

    # ========================================================
    # PRICE FEATURES
    # ========================================================

    if "Price" in df.columns:

        price = safe_numeric(
            df["Price"]
        )

        df["V57_Price_Abs"] = (
            price.abs()
        )

        df["V57_Price_LogAbs"] = (
            np.log1p(
                price.abs()
            )
        )

        df["V57_Price_Zero"] = (
            price == 0
        ).astype(int)

        df["V57_Price_Negative"] = (
            price < 0
        ).astype(int)

    # ========================================================
    # TRANSACTION FEATURES
    # ========================================================

    if (
        "Quantity" in df.columns
        and
        "Price" in df.columns
    ):

        quantity = safe_numeric(
            df["Quantity"]
        )

        price = safe_numeric(
            df["Price"]
        )

        value = (
            quantity * price
        )

        df[
            "V57_Transaction_Value"
        ] = value

        df[
            "V57_Transaction_Value_Abs"
        ] = value.abs()

        df[
            "V57_Transaction_Value_LogAbs"
        ] = np.log1p(
            value.abs()
        )

    # ========================================================
    # DESCRIPTION FEATURES
    # ========================================================

    if "Description" in df.columns:

        description = (
            df["Description"]
            .fillna("")
            .astype(str)
        )

        df[
            "V57_Description_Length"
        ] = (
            description.str.len()
        )

        df[
            "V57_Description_Words"
        ] = (
            description
            .str.split()
            .str.len()
        )

        df[
            "V57_Description_Missing"
        ] = (
            description.str.len() == 0
        ).astype(int)

    # ========================================================
    # SIMPLE CATEGORICAL LENGTH FEATURES
    # ========================================================

    for column, name in [
        (
            "StockCode",
            "V57_StockCode_Length"
        ),
        (
            "Country",
            "V57_Country_Length"
        ),
    ]:

        if column in df.columns:

            df[name] = (
                df[column]
                .fillna("")
                .astype(str)
                .str.len()
            )

    # ========================================================
    # MISSING PATTERN
    # ========================================================

    pattern_columns = [
        c
        for c in [
            "Invoice",
            "StockCode",
            "Description",
            "Quantity",
            "InvoiceDate",
            "Price",
            "Customer ID",
            "Country",
        ]
        if c in df.columns
    ]

    if pattern_columns:

        missing_matrix = (
            df[
                pattern_columns
            ]
            .isna()
            .astype(int)
        )

        df[
            "V57_Missing_Pattern"
        ] = (
            missing_matrix
            .astype(str)
            .agg(
                "".join,
                axis=1
            )
        )

    else:

        df[
            "V57_Missing_Pattern"
        ] = "00000000"

    # ========================================================
    # ERROR LABELS
    # ========================================================

    error_columns = [
        "Missing_Value",
        "Invalid_Value",
        "Outlier",
        "Inconsistency",
        "Duplicate",
    ]

    for column in error_columns:

        df[column] = make_binary_label(
            df[column]
        )

    # ========================================================
    # JOINT ERROR PATTERN
    # ========================================================

    df[
        "V57_Error_Pattern"
    ] = (
        df[
            error_columns
        ]
        .astype(int)
        .astype(str)
        .agg(
            "".join,
            axis=1
        )
    )

    # ========================================================
    # STRATIFIED SPLIT
    # ========================================================

    print()
    print(
        "Creating JOINT ERROR-PATTERN "
        "STRATIFIED split..."
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    train_indices = []
    val_indices = []
    test_indices = []

    for pattern, group in df.groupby(
        "V57_Error_Pattern",
        sort=False
    ):

        indices = (
            group.index
            .to_numpy()
            .copy()
        )

        rng.shuffle(indices)

        n = len(indices)

        n_train = int(
            round(n * 0.60)
        )

        n_val = int(
            round(n * 0.20)
        )

        train_indices.extend(
            indices[
                :n_train
            ].tolist()
        )

        val_indices.extend(
            indices[
                n_train:
                n_train + n_val
            ].tolist()
        )

        test_indices.extend(
            indices[
                n_train + n_val:
            ].tolist()
        )

    train_indices = np.array(
        train_indices,
        dtype=int
    )

    val_indices = np.array(
        val_indices,
        dtype=int
    )

    test_indices = np.array(
        test_indices,
        dtype=int
    )

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    print(
        f"Train: {len(train_indices)}"
    )

    print(
        f"Validation: {len(val_indices)}"
    )

    print(
        f"Test: {len(test_indices)}"
    )

    # ========================================================
    # TRAIN-ONLY CONTEXT FEATURES
    # ========================================================

    print()
    print(
        "Creating TRAIN-ONLY contextual features..."
    )

    contextual_columns = [
        (
            "StockCode",
            "V57_StockCode_Train_Frequency"
        ),
        (
            "Description",
            "V57_Description_Train_Frequency"
        ),
        (
            "Country",
            "V57_Country_Train_Frequency"
        ),
        (
            "Invoice",
            "V57_Invoice_Train_Frequency"
        ),
    ]

    for source, output in contextual_columns:

        if source not in df.columns:
            continue

        train_values = (
            df.loc[
                train_indices,
                source
            ]
            .fillna("<NA>")
            .astype(str)
        )

        counts = (
            train_values
            .value_counts()
            .astype(float)
        )

        mapped = (
            df[source]
            .fillna("<NA>")
            .astype(str)
            .map(counts)
            .fillna(0)
        )

        train_max = (
            float(counts.max())
            if len(counts)
            else 1.0
        )

        if train_max > 0:

            df[output] = (
                mapped / train_max
            ).clip(0, 1)

        else:

            df[output] = 0.0

    # ========================================================
    # TRAIN-ONLY MISSING PATTERN
    # ========================================================

    missing_counts = (
        df.loc[
            train_indices,
            "V57_Missing_Pattern"
        ]
        .value_counts()
        .astype(float)
    )

    df[
        "V57_Missing_Pattern_Frequency"
    ] = (
        df[
            "V57_Missing_Pattern"
        ]
        .map(missing_counts)
        .fillna(0)
    )

    missing_max = (
        float(missing_counts.max())
        if len(missing_counts)
        else 1.0
    )

    if missing_max > 0:

        df[
            "V57_Missing_Pattern_Frequency"
        ] = (
            df[
                "V57_Missing_Pattern_Frequency"
            ]
            / missing_max
        ).clip(0, 1)

    df[
        "V57_Missing_Pattern_Rarity"
    ] = (
        1.0 /
        (
            df[
                "V57_Missing_Pattern_Frequency"
            ] + 1.0
        )
    )

    # ========================================================
    # DUPLICATE SIGNATURE
    #
    # This is deliberately computed from the complete
    # unlabeled dataset.
    #
    # It is an UNSUPERVISED DQ SIGNAL:
    # no ground-truth labels are used.
    #
    # This is appropriate because duplicate detection itself
    # is a dataset-level data-quality operation.
    # ========================================================

    print()
    print(
        "Creating UNSUPERVISED duplicate signature..."
    )

    df[
        "_V57_DUP_SIGNATURE"
    ] = create_duplicate_signature(
        df
    )

    signature_counts = (
        df[
            "_V57_DUP_SIGNATURE"
        ]
        .value_counts()
        .astype(float)
    )

    df[
        "V57_Duplicate_Group_Size"
    ] = (
        df[
            "_V57_DUP_SIGNATURE"
        ]
        .map(signature_counts)
        .fillna(1)
    )

    # Number of other rows with same signature.
    df[
        "V57_Duplicate_Other_Count"
    ] = (
        df[
            "V57_Duplicate_Group_Size"
        ] - 1
    ).clip(lower=0)

    # Binary exact duplicate signal.
    df[
        "V57_Exact_Duplicate_Flag"
    ] = (
        df[
            "V57_Duplicate_Other_Count"
        ] > 0
    ).astype(int)

    # Log-scaled duplicate group size.
    df[
        "V57_Duplicate_Group_Log"
    ] = np.log1p(
        df[
            "V57_Duplicate_Other_Count"
        ]
    )

    # Normalized duplicate strength.
    max_duplicate = float(
        df[
            "V57_Duplicate_Other_Count"
        ].max()
    )

    if max_duplicate > 0:

        df[
            "V57_Duplicate_Strength"
        ] = (
            df[
                "V57_Duplicate_Other_Count"
            ] / max_duplicate
        ).clip(0, 1)

    else:

        df[
            "V57_Duplicate_Strength"
        ] = 0.0

    # ========================================================
    # MODEL FEATURES
    # ========================================================

    forbidden = {
        # identifiers
        "Corrupted_Row_ID",
        "Original_Row_ID",

        # internal helper
        "V57_Error_Pattern",
        "V57_Missing_Pattern",
        "_V57_DUP_SIGNATURE",

        # customer identifier
        "Customer ID",

        # raw categorical/text fields
        "Invoice",
        "StockCode",
        "Description",
        "InvoiceDate",
        "Country",

        # ground truth
        "Missing_Value",
        "Invalid_Value",
        "Outlier",
        "Inconsistency",
        "Duplicate",

        # synthetic artifact protection
        "V51_Incorrect_Description_Flag",
        "V55_Incorrect_Description_Flag",
        "V56_Incorrect_Description_Flag",
        "V57_Incorrect_Description_Flag",
    }

    model_features = []

    for column in df.columns:

        if column in forbidden:
            continue

        if column.startswith("_"):
            continue

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):

            model_features.append(
                column
            )

    # Convert and clean.
    for column in model_features:

        df[column] = (
            safe_numeric(
                df[column]
            )
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(0)
        )

    # Remove constant features.
    model_features = [
        c
        for c in model_features
        if df[c].nunique(
            dropna=False
        ) > 1
    ]

    print()
    print(
        f"Clean model features: "
        f"{len(model_features)}"
    )

    if len(model_features) < 5:

        raise RuntimeError(
            "Too few model features."
        )

    # ========================================================
    # SAVE FEATURES
    # ========================================================

    pd.DataFrame(
        {
            "Feature": model_features
        }
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "v5_7_model_features.csv"
        ),
        index=False
    )

    # ========================================================
    # SPLIT MATRICES
    # ========================================================

    X_train = df.loc[
        train_indices,
        model_features
    ]

    X_val = df.loc[
        val_indices,
        model_features
    ]

    X_test = df.loc[
        test_indices,
        model_features
    ]

    # ========================================================
    # TYPE-SPECIFIC MODELS
    # ========================================================

    print()
    print("=" * 72)
    print(
        "TRAINING TYPE-SPECIFIC RANDOM FORESTS"
    )
    print("=" * 72)

    results = []
    thresholds = []
    feature_importance = []

    test_predictions = pd.DataFrame(
        index=test_indices
    )

    for error_type in error_columns:

        print()
        print(
            f"Processing: {error_type}"
        )

        y_train = df.loc[
            train_indices,
            error_type
        ].astype(int)

        y_val = df.loc[
            val_indices,
            error_type
        ].astype(int)

        y_test = df.loc[
            test_indices,
            error_type
        ].astype(int)

        # ----------------------------------------------------
        # RANDOM FOREST
        # ----------------------------------------------------

        model = RandomForestClassifier(
            n_estimators=350,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train
        )

        val_probability = (
            model.predict_proba(
                X_val
            )[:, 1]
        )

        test_probability = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        # ----------------------------------------------------
        # VALIDATION-ONLY THRESHOLD
        # ----------------------------------------------------

        threshold, validation_f1 = (
            choose_threshold(
                y_val,
                val_probability
            )
        )

        # ----------------------------------------------------
        # MISSING SPECIALIZATION
        # ----------------------------------------------------

        if error_type == "Missing_Value":

            missing_signal = (
                X_test[
                    "V57_Missing_Pattern_Rarity"
                ]
                .to_numpy()
            )

            test_probability = (
                0.85 * test_probability
                + 0.15 * missing_signal
            )

        # ----------------------------------------------------
        # DUPLICATE SPECIALIZATION
        #
        # Stronger weight for exact unsupervised
        # duplicate evidence.
        # ----------------------------------------------------

        elif error_type == "Duplicate":

            duplicate_flag = (
                X_test[
                    "V57_Exact_Duplicate_Flag"
                ]
                .to_numpy()
            )

            duplicate_strength = (
                X_test[
                    "V57_Duplicate_Strength"
                ]
                .to_numpy()
            )

            duplicate_group = (
                X_test[
                    "V57_Duplicate_Group_Log"
                ]
                .to_numpy()
            )

            # Combine RF probability with independent
            # exact duplicate evidence.
            #
            # Exact duplicate flag receives strongest
            # contribution because it is a direct,
            # label-free DQ signal.
            test_probability = (
                0.50 * test_probability
                + 0.35 * duplicate_flag
                + 0.10 * duplicate_strength
                + 0.05 * np.clip(
                    duplicate_group,
                    0,
                    1
                )
            )

        # ----------------------------------------------------
        # TEST EVALUATION
        # ----------------------------------------------------

        evaluation = evaluate(
            y_test,
            test_probability,
            threshold
        )

        results.append(
            {
                "Error_Type": error_type,
                "Threshold": threshold,
                "Accuracy": evaluation[
                    "Accuracy"
                ],
                "Precision": evaluation[
                    "Precision"
                ],
                "Recall": evaluation[
                    "Recall"
                ],
                "F1_Score": evaluation[
                    "F1_Score"
                ],
                "TP": evaluation["TP"],
                "FP": evaluation["FP"],
                "TN": evaluation["TN"],
                "FN": evaluation["FN"],
            }
        )

        thresholds.append(
            {
                "Error_Type": error_type,
                "Threshold": threshold,
                "Validation_F1": validation_f1,
            }
        )

        test_predictions[
            f"{error_type}_Probability"
        ] = test_probability

        test_predictions[
            f"{error_type}_Prediction"
        ] = evaluation[
            "Prediction"
        ]

        # ----------------------------------------------------
        # FEATURE IMPORTANCE
        # ----------------------------------------------------

        for feature, importance in zip(
            model_features,
            model.feature_importances_
        ):

            feature_importance.append(
                {
                    "Error_Type": error_type,
                    "Feature": feature,
                    "Importance": importance,
                }
            )

        print(
            f"Threshold: {threshold:.2f}"
        )

        print(
            f"Validation F1: "
            f"{validation_f1 * 100:.2f}%"
        )

        print(
            f"Test Precision: "
            f"{evaluation['Precision'] * 100:.2f}%"
        )

        print(
            f"Test Recall: "
            f"{evaluation['Recall'] * 100:.2f}%"
        )

        print(
            f"Test F1: "
            f"{evaluation['F1_Score'] * 100:.2f}%"
        )

    # ========================================================
    # OVERALL ANY-ERROR METRIC
    # ========================================================

    y_test_any = (
        df.loc[
            test_indices,
            error_columns
        ]
        .max(axis=1)
        .astype(int)
        .to_numpy()
    )

    prediction_columns = [
        f"{c}_Prediction"
        for c in error_columns
    ]

    prediction_matrix = (
        test_predictions[
            prediction_columns
        ]
        .to_numpy()
    )

    any_error_prediction = (
        prediction_matrix
        .max(axis=1)
        .astype(int)
    )

    overall_accuracy = accuracy_score(
        y_test_any,
        any_error_prediction
    )

    overall_precision = precision_score(
        y_test_any,
        any_error_prediction,
        zero_division=0
    )

    overall_recall = recall_score(
        y_test_any,
        any_error_prediction,
        zero_division=0
    )

    overall_f1 = f1_score(
        y_test_any,
        any_error_prediction,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test_any,
        any_error_prediction,
        labels=[0, 1]
    ).ravel()

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    thresholds_df = pd.DataFrame(
        thresholds
    )

    importance_df = pd.DataFrame(
        feature_importance
    )

    overall_df = pd.DataFrame(
        [
            {
                "Model":
                    "TYPE-AWARE HYBRID V5.7",
                "Accuracy":
                    overall_accuracy,
                "Precision":
                    overall_precision,
                "Recall":
                    overall_recall,
                "F1_Score":
                    overall_f1,
                "TP":
                    int(tp),
                "FP":
                    int(fp),
                "TN":
                    int(tn),
                "FN":
                    int(fn),
            }
        ]
    )

    results_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "type_aware_hybrid_v5_7_results.csv"
        ),
        index=False
    )

    thresholds_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "v5_7_threshold_optimization.csv"
        ),
        index=False
    )

    importance_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "v5_7_random_forest_feature_importance.csv"
        ),
        index=False
    )

    overall_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "v5_7_overall_performance.csv"
        ),
        index=False
    )

    # ========================================================
    # TEST PREDICTIONS
    # ========================================================

    test_predictions[
        "Corrupted_Row_ID"
    ] = df.loc[
        test_indices,
        "Corrupted_Row_ID"
    ].to_numpy()

    test_predictions[
        "Original_Row_ID"
    ] = df.loc[
        test_indices,
        "Original_Row_ID"
    ].to_numpy()

    test_predictions[
        "Any_Error_True"
    ] = y_test_any

    test_predictions[
        "Any_Error_Prediction"
    ] = any_error_prediction

    test_predictions.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "v5_7_test_predictions.csv"
        ),
        index=False
    )

    # ========================================================
    # SPLIT DISTRIBUTION
    # ========================================================

    split_rows = []

    for split_name, indices in [
        (
            "Train",
            train_indices
        ),
        (
            "Validation",
            val_indices
        ),
        (
            "Test",
            test_indices
        ),
    ]:

        subset = df.loc[
            indices
        ]

        for error_type in error_columns:

            count = int(
                subset[
                    error_type
                ].sum()
            )

            percentage = (
                count
                / len(subset)
                * 100
            )

            split_rows.append(
                {
                    "Split":
                        split_name,
                    "Error_Type":
                        error_type,
                    "Count":
                        count,
                    "Percentage":
                        percentage,
                }
            )

    pd.DataFrame(
        split_rows
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "v5_7_split_distribution.csv"
        ),
        index=False
    )

    # ========================================================
    # METHODOLOGY AUDIT
    # ========================================================

    audit = pd.DataFrame(
        {
            "Audit_Item": [
                "Split Strategy",
                "Context Statistics",
                "Missing Pattern Statistics",
                "Duplicate Signal",
                "Duplicate Ground Truth Used",
                "Threshold Optimization",
                "Test Evaluation",
                "Identifier Features",
                "Customer ID Model Feature",
                "Synthetic Artifact Flag",
                "Missing Specialization",
                "Duplicate Specialization",
                "Clean Model Features",
            ],
            "Status": [
                "JOINT ERROR-PATTERN STRATIFIED",
                "TRAIN ONLY",
                "TRAIN ONLY",
                "UNSUPERVISED DATASET-LEVEL",
                "NO",
                "VALIDATION ONLY",
                "FINAL TEST ONLY",
                "REMOVED",
                "REMOVED",
                "REMOVED",
                "ENABLED",
                "ENABLED",
                str(
                    len(model_features)
                ),
            ],
        }
    )

    audit.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "v5_7_methodology_audit.csv"
        ),
        index=False
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 72)
    print(
        "TYPE-AWARE HYBRID V5.7 RESULTS"
    )
    print("=" * 72)

    print()

    print(
        results_df[
            [
                "Error_Type",
                "Precision",
                "Recall",
                "F1_Score",
            ]
        ].to_string(
            index=False,
            formatters={
                "Precision":
                    lambda x:
                    f"{x * 100:.2f}%",
                "Recall":
                    lambda x:
                    f"{x * 100:.2f}%",
                "F1_Score":
                    lambda x:
                    f"{x * 100:.2f}%",
            },
        )
    )

    print()
    print("=" * 72)

    print(
        f"Overall Accuracy : "
        f"{overall_accuracy * 100:.2f}%"
    )

    print(
        f"Overall Precision: "
        f"{overall_precision * 100:.2f}%"
    )

    print(
        f"Overall Recall   : "
        f"{overall_recall * 100:.2f}%"
    )

    print(
        f"Overall F1       : "
        f"{overall_f1 * 100:.2f}%"
    )

    print(
        f"TP: {tp}"
    )

    print(
        f"FP: {fp}"
    )

    print(
        f"TN: {tn}"
    )

    print(
        f"FN: {fn}"
    )

    print("=" * 72)

    print()
    print(
        "METHODOLOGY AUDIT"
    )

    print("-" * 72)

    print(
        "Split strategy                 : "
        "JOINT ERROR-PATTERN STRATIFIED"
    )

    print(
        "Context statistics             : "
        "TRAIN ONLY"
    )

    print(
        "Missing-pattern statistics     : "
        "TRAIN ONLY"
    )

    print(
        "Duplicate signal               : "
        "UNSUPERVISED DATASET-LEVEL"
    )

    print(
        "Duplicate ground truth used    : "
        "NO"
    )

    print(
        "Threshold optimization         : "
        "VALIDATION ONLY"
    )

    print(
        "Test evaluation                : "
        "FINAL TEST ONLY"
    )

    print(
        "Identifier features            : "
        "REMOVED"
    )

    print(
        "Customer ID model feature      : "
        "REMOVED"
    )

    print(
        "Synthetic artifact flag        : "
        "REMOVED"
    )

    print(
        "Missing specialization         : "
        "ENABLED"
    )

    print(
        "Duplicate specialization       : "
        "ENABLED"
    )

    print(
        f"Clean model features           : "
        f"{len(model_features)}"
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "V5.7 execution completed successfully."
    )