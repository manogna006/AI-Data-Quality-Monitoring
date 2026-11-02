# ============================================================
# TYPE-AWARE HYBRID V5.6
# STRICT LEAKAGE-SAFE RANDOM FOREST
#
# TRAIN-ONLY CONTEXT
# VALIDATION-ONLY THRESHOLD OPTIMIZATION
# FINAL TEST EVALUATION
# SPECIALIZED MISSING / DUPLICATE DETECTION
#
# NO:
# - Identifier features
# - Customer ID as model feature
# - Synthetic corruption artifact flags
# - Full-dataset contextual statistics
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
    "v5_6",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# BASIC UTILITIES
# ============================================================

def safe_numeric(series):
    """
    Safely convert a Series to numeric.

    If duplicate column names somehow remain and pandas returns
    a DataFrame instead of a Series, use the first column.
    """
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]

    return pd.to_numeric(series, errors="coerce")


def safe_series(df, column):
    """
    Always return exactly one Series for a requested column.
    """
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index)

    obj = df.loc[:, column]

    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]

    return obj


def normalize_series(series, train_min=None, train_max=None):
    """
    Normalize using supplied training parameters.

    This prevents validation/test values from influencing
    transformation parameters.
    """
    s = safe_numeric(series).replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    if train_min is None:
        train_min = float(s.min())

    if train_max is None:
        train_max = float(s.max())

    if train_max <= train_min:
        return pd.Series(
            0.0,
            index=s.index
        )

    return ((s - train_min) / (train_max - train_min)).clip(
        0,
        1
    )


def coalesce_duplicate_columns(dataframe):
    """
    Permanently removes duplicate column names.

    If multiple columns have the same name, take the first
    non-null value from left to right.
    """
    if not dataframe.columns.duplicated().any():
        return dataframe

    result = pd.DataFrame(index=dataframe.index)

    processed = set()

    for column in dataframe.columns:

        if column in processed:
            continue

        same = dataframe.loc[
            :,
            dataframe.columns == column
        ]

        if same.shape[1] == 1:
            result[column] = same.iloc[:, 0]

        else:
            result[column] = (
                same.bfill(axis=1)
                .iloc[:, 0]
            )

        processed.add(column)

    return result


def clean_after_merge(df, name="dataframe"):
    """
    Defensive dataframe cleanup after every merge.
    """
    duplicate_columns = df.columns[
        df.columns.duplicated()
    ].tolist()

    if duplicate_columns:
        print(
            f"Duplicate columns detected after {name}: "
            f"{sorted(set(duplicate_columns))}"
        )

        df = coalesce_duplicate_columns(df)

        print(
            f"Duplicate columns coalesced after {name}."
        )

    remaining = df.columns[
        df.columns.duplicated()
    ].tolist()

    if remaining:
        raise RuntimeError(
            f"Duplicate columns remain after {name}: "
            f"{remaining}"
        )

    return df


def make_binary_label(series):
    return (
        safe_numeric(series)
        .fillna(0)
        .astype(int)
        .clip(0, 1)
    )


def choose_threshold(y_true, probabilities):
    """
    Validation-only threshold optimization.

    Selects the threshold giving maximum F1.
    """
    best_threshold = 0.50
    best_f1 = -1

    thresholds = np.arange(
        0.01,
        1.00,
        0.01
    )

    for threshold in thresholds:

        pred = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            pred,
            zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return float(best_threshold), float(best_f1)


def evaluate_binary(
    y_true,
    probabilities,
    threshold
):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
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
        "Predictions": predictions,
    }


# ============================================================
# TRAIN-ONLY GROUP STATISTICS
# ============================================================

def train_group_frequency(
    train_df,
    all_df,
    column
):
    """
    Compute category frequencies using TRAIN ONLY.

    Unknown validation/test categories receive 0.
    """

    if column not in train_df.columns:
        return (
            pd.Series(
                0.0,
                index=all_df.index
            ),
            0.0,
            0.0,
        )

    train_values = train_df[column].astype(str)

    counts = train_values.value_counts()

    mapped = (
        all_df[column]
        .astype(str)
        .map(counts)
        .fillna(0)
        .astype(float)
    )

    train_counts = counts.astype(float)

    min_value = (
        float(train_counts.min())
        if len(train_counts)
        else 0.0
    )

    max_value = (
        float(train_counts.max())
        if len(train_counts)
        else 1.0
    )

    return mapped, min_value, max_value


def add_train_only_frequency(
    df,
    train_indices,
    column,
    output_name
):
    train_part = df.loc[
        train_indices
    ]

    mapped, min_value, max_value = (
        train_group_frequency(
            train_part,
            df,
            column
        )
    )

    df[output_name] = normalize_series(
        mapped,
        min_value,
        max_value
    )

    return df


# ============================================================
# DUPLICATE SIGNATURE
# ============================================================

def create_duplicate_signature(df):
    """
    Transaction-level duplicate signature.

    Customer ID is used only for the unsupervised duplicate
    DQ signal and NEVER becomes a model feature.
    """

    fields = [
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
        c for c in fields
        if c in df.columns
    ]

    if not available:
        return pd.Series(
            "",
            index=df.index
        )

    temp = df[available].copy()

    for col in available:

        if pd.api.types.is_datetime64_any_dtype(
            temp[col]
        ):
            temp[col] = (
                temp[col]
                .astype(str)
            )
        else:
            temp[col] = (
                temp[col]
                .fillna("<NA>")
                .astype(str)
            )

    return (
        temp.astype(str)
        .agg("||".join, axis=1)
    )


def add_train_only_duplicate_evidence(
    df,
    train_indices
):
    """
    Strict duplicate evidence.

    TRAIN:
        Leave-one-out duplicate frequency.

    VALIDATION / TEST:
        Frequency based ONLY on training signatures.

    Therefore validation/test rows never contribute to
    their own duplicate evidence.
    """

    signatures = df[
        "_V56_DUP_SIGNATURE"
    ].astype(str)

    train_signatures = signatures.loc[
        train_indices
    ]

    train_counts = (
        train_signatures
        .value_counts()
        .astype(float)
    )

    evidence = (
        signatures
        .map(train_counts)
        .fillna(0)
        .astype(float)
    )

    # Leave-one-out for training rows.
    train_evidence = (
        evidence.loc[train_indices]
        .subtract(1)
        .clip(lower=0)
    )

    evidence.loc[
        train_indices
    ] = train_evidence

    train_max = float(
        train_evidence.max()
    )

    if train_max > 0:
        normalized = (
            evidence / train_max
        ).clip(0, 1)
    else:
        normalized = pd.Series(
            0.0,
            index=df.index
        )

    df[
        "V56_TrainOnly_Duplicate_Evidence"
    ] = normalized

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 72)
    print(
        "TYPE-AWARE HYBRID V5.6"
    )
    print(
        "STRICT LEAKAGE-SAFE RANDOM FOREST"
    )
    print(
        "TRAIN-ONLY CONTEXT + SPECIALIZED MISSING/DUPLICATE"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DEFENSIVE CLEANUP
    # --------------------------------------------------------

    raw = clean_after_merge(
        raw,
        "raw loading"
    )

    v2_results = clean_after_merge(
        v2_results,
        "V2 results loading"
    )

    v2_features = clean_after_merge(
        v2_features,
        "V2 features loading"
    )

    ground_truth = clean_after_merge(
        ground_truth,
        "ground truth loading"
    )

    # --------------------------------------------------------
    # CORRUPTED ROW ID
    # --------------------------------------------------------

    if "Corrupted_Row_ID" not in v2_features.columns:

        print()
        print(
            "V2 features do not contain "
            "Corrupted_Row_ID."
        )

        v2_features = v2_features.copy()

        v2_features[
            "Corrupted_Row_ID"
        ] = np.arange(
            len(v2_features)
        )

    if "Corrupted_Row_ID" not in v2_results.columns:

        v2_results = v2_results.copy()

        v2_results[
            "Corrupted_Row_ID"
        ] = np.arange(
            len(v2_results)
        )

    if "Corrupted_Row_ID" not in raw.columns:

        print()
        print(
            "Raw CSV does not contain Corrupted_Row_ID."
        )

        raw = raw.copy()

        raw[
            "Corrupted_Row_ID"
        ] = np.arange(
            len(raw)
        )

        print(
            "Creating from raw row order..."
        )

    # --------------------------------------------------------
    # INTERNAL STABLE INDEX
    # --------------------------------------------------------

    raw[
        "_V56_INTERNAL_INDEX"
    ] = np.arange(
        len(raw)
    )

    v2_features[
        "_V56_INTERNAL_INDEX"
    ] = np.arange(
        len(v2_features)
    )

    v2_results[
        "_V56_INTERNAL_INDEX"
    ] = np.arange(
        len(v2_results)
    )

    # --------------------------------------------------------
    # ENSURE UNIQUE FEATURE ROWS
    # --------------------------------------------------------

    print(
        "Unique feature rows:",
        v2_features[
            "Corrupted_Row_ID"
        ].nunique()
    )

    # --------------------------------------------------------
    # MERGE V2 FEATURES + RESULTS
    # --------------------------------------------------------

    # Only take V2 result columns that are not duplicated
    # elsewhere.

    result_columns = [
        c
        for c in v2_results.columns
        if c not in {
            "Corrupted_Row_ID",
            "_V56_INTERNAL_INDEX",
        }
    ]

    result_columns = [
        c
        for c in result_columns
        if c not in v2_features.columns
    ]

    v2_result_subset = v2_results[
        ["Corrupted_Row_ID"]
        + result_columns
    ].copy()

    df = v2_features.merge(
        v2_result_subset,
        on="Corrupted_Row_ID",
        how="left",
        validate="one_to_one",
    )

    df = clean_after_merge(
        df,
        "V2 feature/result merge"
    )

    # --------------------------------------------------------
    # GROUND TRUTH
    # --------------------------------------------------------

    if "Original_Row_ID" not in df.columns:

        if "Original_Row_ID" in raw.columns:

            id_map = raw[
                [
                    "Corrupted_Row_ID",
                    "Original_Row_ID",
                ]
            ].copy()

            df = df.merge(
                id_map,
                on="Corrupted_Row_ID",
                how="left",
                validate="one_to_one",
            )

        else:
            raise RuntimeError(
                "Original_Row_ID is missing from both "
                "V2 features and raw data."
            )

    required_gt_columns = [
        "Original_Row_ID",
        "Missing_Value",
        "Invalid_Value",
        "Outlier",
        "Inconsistency",
        "Duplicate",
    ]

    missing_gt_columns = [
        c
        for c in required_gt_columns
        if c not in ground_truth.columns
    ]

    if missing_gt_columns:
        raise RuntimeError(
            "Ground truth is missing columns: "
            f"{missing_gt_columns}"
        )

    gt_subset = ground_truth[
        required_gt_columns
    ].copy()

    # Prevent duplicate GT keys.
    gt_subset = (
        gt_subset
        .drop_duplicates(
            subset=["Original_Row_ID"]
        )
    )

    df = df.merge(
        gt_subset,
        on="Original_Row_ID",
        how="left",
        validate="many_to_one",
    )

    df = clean_after_merge(
        df,
        "ground-truth merge"
    )

    print(
        f"After ground-truth merge: {len(df)}"
    )

    # --------------------------------------------------------
    # RAW CONTEXT
    # --------------------------------------------------------

    raw_context_columns = [
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

    raw_context_columns = [
        c
        for c in raw_context_columns
        if c in raw.columns
    ]

    raw_context = raw[
        raw_context_columns
    ].copy()

    # Remove columns already present in df except key.
    merge_key = "Corrupted_Row_ID"

    context_columns_to_add = [
        c
        for c in raw_context.columns
        if c == merge_key
        or c not in df.columns
    ]

    raw_context = raw_context[
        context_columns_to_add
    ]

    print(
        "Raw context columns prepared."
    )

    df = df.merge(
        raw_context,
        on=merge_key,
        how="left",
        validate="one_to_one",
    )

    df = clean_after_merge(
        df,
        "raw context merge"
    )

    print(
        f"After raw context merge: {len(df)}"
    )

    # --------------------------------------------------------
    # FINAL MERGE INTEGRITY
    # --------------------------------------------------------

    if len(df) != len(v2_features):

        raise RuntimeError(
            "Row count changed unexpectedly."
        )

    if df[
        "Corrupted_Row_ID"
    ].duplicated().any():

        raise RuntimeError(
            "Corrupted_Row_ID is no longer unique."
        )

    print(
        "Merge integrity verified."
    )

    # --------------------------------------------------------
    # FINAL COLUMN SAFETY
    # --------------------------------------------------------

    df = clean_after_merge(
        df,
        "final pre-processing"
    )

    # --------------------------------------------------------
    # RAW NUMERIC CONVERSION
    # --------------------------------------------------------

    for column in [
        "Quantity",
        "Price",
        "Customer ID",
    ]:

        if column in df.columns:

            df[column] = safe_numeric(
                df[column]
            )

    # --------------------------------------------------------
    # DATETIME FEATURES
    # --------------------------------------------------------

    if "InvoiceDate" in df.columns:

        dt = pd.to_datetime(
            df["InvoiceDate"],
            errors="coerce"
        )

        df["V56_Invoice_Year"] = (
            dt.dt.year.fillna(0)
        )

        df["V56_Invoice_Month"] = (
            dt.dt.month.fillna(0)
        )

        df["V56_Invoice_Day"] = (
            dt.dt.day.fillna(0)
        )

        df["V56_Invoice_Weekday"] = (
            dt.dt.weekday.fillna(-1)
        )

        df["V56_Invoice_Hour"] = (
            dt.dt.hour.fillna(0)
        )

    # --------------------------------------------------------
    # BASIC RAW FEATURES
    # --------------------------------------------------------

    if "Quantity" in df.columns:

        quantity = safe_numeric(
            df["Quantity"]
        )

        df["V56_Quantity_Abs"] = (
            quantity.abs()
        )

        df["V56_Quantity_LogAbs"] = np.log1p(
            quantity.abs()
        )

        df["V56_Quantity_IsZero"] = (
            quantity == 0
        ).astype(int)

        df["V56_Quantity_IsNegative"] = (
            quantity < 0
        ).astype(int)

    if "Price" in df.columns:

        price = safe_numeric(
            df["Price"]
        )

        df["V56_Price_Abs"] = (
            price.abs()
        )

        df["V56_Price_LogAbs"] = np.log1p(
            price.abs()
        )

        df["V56_Price_IsZero"] = (
            price == 0
        ).astype(int)

        df["V56_Price_IsNegative"] = (
            price < 0
        ).astype(int)

    if (
        "Quantity" in df.columns
        and "Price" in df.columns
    ):

        quantity = safe_numeric(
            df["Quantity"]
        )

        price = safe_numeric(
            df["Price"]
        )

        df["V56_Transaction_Value"] = (
            quantity * price
        )

        df[
            "V56_Transaction_Value_Abs"
        ] = (
            quantity.abs()
            * price.abs()
        )

        df[
            "V56_Transaction_Value_LogAbs"
        ] = np.log1p(
            (
                quantity.abs()
                * price.abs()
            )
        )

    if "Description" in df.columns:

        description = (
            df["Description"]
            .fillna("")
            .astype(str)
        )

        df["V56_Description_Length"] = (
            description.str.len()
        )

        df[
            "V56_Description_Word_Count"
        ] = (
            description
            .str.split()
            .str.len()
        )

        df[
            "V56_Description_Missing"
        ] = (
            description.str.len() == 0
        ).astype(int)

    if "StockCode" in df.columns:

        stock = (
            df["StockCode"]
            .fillna("")
            .astype(str)
        )

        df["V56_StockCode_Length"] = (
            stock.str.len()
        )

    if "Country" in df.columns:

        country = (
            df["Country"]
            .fillna("")
            .astype(str)
        )

        df["V56_Country_Length"] = (
            country.str.len()
        )

    # --------------------------------------------------------
    # MISSING PATTERN
    # --------------------------------------------------------

    raw_columns_for_pattern = [
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

    if raw_columns_for_pattern:

        missing_matrix = (
            df[
                raw_columns_for_pattern
            ].isna().astype(int)
        )

        df[
            "V56_Missing_Pattern"
        ] = missing_matrix.astype(str).agg(
            "".join,
            axis=1
        )

    else:

        df[
            "V56_Missing_Pattern"
        ] = "00000000"

    # --------------------------------------------------------
    # JOINT ERROR PATTERN
    # --------------------------------------------------------

    error_columns = [
        "Missing_Value",
        "Invalid_Value",
        "Outlier",
        "Inconsistency",
        "Duplicate",
    ]

    for c in error_columns:

        df[c] = make_binary_label(
            safe_series(df, c)
        )

    df[
        "V56_Error_Pattern"
    ] = (
        df[
            error_columns
        ]
        .astype(int)
        .astype(str)
        .agg("".join, axis=1)
    )

    # --------------------------------------------------------
    # STRATIFIED SPLIT
    # --------------------------------------------------------

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
        "V56_Error_Pattern",
        sort=False
    ):

        indices = group.index.to_numpy().copy()

        rng.shuffle(indices)

        n = len(indices)

        n_train = int(
            round(n * 0.60)
        )

        n_val = int(
            round(n * 0.20)
        )

        # Ensure all remaining rows go to test.
        train_part = indices[
            :n_train
        ]

        val_part = indices[
            n_train:n_train + n_val
        ]

        test_part = indices[
            n_train + n_val:
        ]

        train_indices.extend(
            train_part.tolist()
        )

        val_indices.extend(
            val_part.tolist()
        )

        test_indices.extend(
            test_part.tolist()
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

    # --------------------------------------------------------
    # SPLIT DISTRIBUTION
    # --------------------------------------------------------

    split_distribution = []

    for split_name, indices in [
        ("Train", train_indices),
        ("Validation", val_indices),
        ("Test", test_indices),
    ]:

        subset = df.loc[indices]

        for error_type in error_columns:

            count = int(
                subset[
                    error_type
                ].sum()
            )

            percentage = (
                count / len(subset) * 100
            )

            split_distribution.append(
                {
                    "Split": split_name,
                    "Error_Type": error_type,
                    "Count": count,
                    "Percentage": percentage,
                }
            )

    pd.DataFrame(
        split_distribution
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "v5_6_split_distribution.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # TRAIN-ONLY CONTEXT FEATURES
    # --------------------------------------------------------

    print()
    print(
        "Creating TRAIN-ONLY contextual features..."
    )

    frequency_columns = [
        ("StockCode", "V56_StockCode_Train_Frequency"),
        (
            "Description",
            "V56_Description_Train_Frequency",
        ),
        (
            "Country",
            "V56_Country_Train_Frequency",
        ),
        (
            "Invoice",
            "V56_Invoice_Train_Frequency",
        ),
    ]

    for source_column, output_column in frequency_columns:

        if source_column in df.columns:

            train_values = (
                df.loc[
                    train_indices,
                    source_column
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
                df[
                    source_column
                ]
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

                df[output_column] = (
                    mapped / train_max
                ).clip(0, 1)

            else:

                df[output_column] = 0.0

    # --------------------------------------------------------
    # TRAIN-ONLY MISSING PATTERN FREQUENCY
    # --------------------------------------------------------

    train_pattern_counts = (
        df.loc[
            train_indices,
            "V56_Missing_Pattern"
        ]
        .value_counts()
        .astype(float)
    )

    df[
        "V56_Missing_Pattern_Frequency"
    ] = (
        df[
            "V56_Missing_Pattern"
        ]
        .map(train_pattern_counts)
        .fillna(0)
        .astype(float)
    )

    train_missing_max = float(
        train_pattern_counts.max()
    ) if len(train_pattern_counts) else 1.0

    if train_missing_max > 0:

        df[
            "V56_Missing_Pattern_Frequency"
        ] = (
            df[
                "V56_Missing_Pattern_Frequency"
            ]
            / train_missing_max
        ).clip(0, 1)

    # --------------------------------------------------------
    # MISSING PATTERN RARITY
    # --------------------------------------------------------

    df[
        "V56_Missing_Pattern_Rarity"
    ] = 1.0 / (
        df[
            "V56_Missing_Pattern_Frequency"
        ]
        + 1.0
    )

    # --------------------------------------------------------
    # DUPLICATE SIGNATURE
    # --------------------------------------------------------

    print(
        "Creating TRAIN-ONLY duplicate evidence..."
    )

    df[
        "_V56_DUP_SIGNATURE"
    ] = create_duplicate_signature(
        df
    )

    df = add_train_only_duplicate_evidence(
        df,
        train_indices
    )

    # --------------------------------------------------------
    # DUPLICATE OCCURRENCE / EXACT SIGNAL
    # --------------------------------------------------------

    duplicate_evidence = (
        df[
            "V56_TrainOnly_Duplicate_Evidence"
        ]
    )

    df[
        "V56_Duplicate_Strong_Evidence"
    ] = (
        duplicate_evidence >= 0.5
    ).astype(int)

    df[
        "V56_Duplicate_Weak_Evidence"
    ] = (
        duplicate_evidence > 0
    ).astype(int)

    # --------------------------------------------------------
    # REMOVE RAW CATEGORICAL / IDENTIFIER COLUMNS
    # --------------------------------------------------------

    forbidden_columns = {
        # identifiers
        "Corrupted_Row_ID",
        "Original_Row_ID",
        "_V56_INTERNAL_INDEX",

        # customer identifier
        "Customer ID",

        # raw text/categorical
        "Invoice",
        "StockCode",
        "Description",
        "InvoiceDate",
        "Country",

        # ground truth labels
        "Missing_Value",
        "Invalid_Value",
        "Outlier",
        "Inconsistency",
        "Duplicate",

        # internal helper
        "V56_Error_Pattern",
        "V56_Missing_Pattern",
        "_V56_DUP_SIGNATURE",

        # synthetic artifact protection
        "V51_Incorrect_Description_Flag",
        "V55_Incorrect_Description_Flag",
        "V56_Incorrect_Description_Flag",
    }

    # --------------------------------------------------------
    # BUILD NUMERIC MODEL FEATURES
    # --------------------------------------------------------

    candidate_features = []

    for column in df.columns:

        if column in forbidden_columns:
            continue

        if column.startswith("_"):
            continue

        series = df[column]

        # Never allow objects/text into RF.
        if pd.api.types.is_numeric_dtype(
            series
        ):
            candidate_features.append(
                column
            )

    # Convert all model features to numeric.
    for column in candidate_features:

        df[column] = safe_numeric(
            df[column]
        )

    model_features = []

    for column in candidate_features:

        if (
            df[column].notna().sum() == 0
        ):
            continue

        df[column] = (
            df[column]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(0)
        )

        model_features.append(
            column
        )

    # Remove constant features.
    variable_features = []

    for column in model_features:

        if df[column].nunique(dropna=False) > 1:

            variable_features.append(
                column
            )

    model_features = variable_features

    print()
    print(
        f"Clean model features: "
        f"{len(model_features)}"
    )

    if len(model_features) < 5:

        raise RuntimeError(
            "Too few model features were created."
        )

    # --------------------------------------------------------
    # SAVE FEATURE LIST
    # --------------------------------------------------------

    pd.DataFrame(
        {
            "Feature": model_features
        }
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "v5_6_model_features.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # TRAIN / VALIDATION / TEST MATRICES
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # TRAIN RANDOM FOREST FOR EACH ERROR TYPE
    # --------------------------------------------------------

    print()
    print(
        "=" * 72
    )
    print(
        "TRAINING TYPE-SPECIFIC RANDOM FORESTS"
    )
    print(
        "=" * 72
    )

    error_results = []
    threshold_results = []
    test_predictions = pd.DataFrame(
        index=test_indices
    )

    feature_importance_all = []

    test_any_probability = np.zeros(
        len(test_indices)
    )

    test_any_prediction = np.zeros(
        len(test_indices),
        dtype=int
    )

    specialized_probabilities = {}

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
        # SPECIALIZED FUSION
        # ----------------------------------------------------

        if error_type == "Missing_Value":

            # Missingness is strongly represented by
            # actual missingness and contextual rarity.
            missing_signal = np.zeros(
                len(test_indices)
            )

            if (
                "V56_Missing_Pattern_Rarity"
                in X_test.columns
            ):

                missing_signal = (
                    X_test[
                        "V56_Missing_Pattern_Rarity"
                    ]
                    .to_numpy()
                )

            test_probability_final = (
                0.85 * test_probability
                + 0.15 * missing_signal
            )

            # Keep validation model threshold as the
            # decision threshold.
            test_probability = (
                test_probability_final
            )

        elif error_type == "Duplicate":

            duplicate_signal = (
                X_test[
                    "V56_TrainOnly_Duplicate_Evidence"
                ]
                .to_numpy()
                if (
                    "V56_TrainOnly_Duplicate_Evidence"
                    in X_test.columns
                )
                else np.zeros(
                    len(test_indices)
                )
            )

            # Specialized duplicate fusion.
            test_probability = (
                0.70 * test_probability
                + 0.30 * duplicate_signal
            )

        specialized_probabilities[
            error_type
        ] = test_probability

        # ----------------------------------------------------
        # FINAL TEST EVALUATION
        # ----------------------------------------------------

        result = evaluate_binary(
            y_test,
            test_probability,
            threshold
        )

        error_results.append(
            {
                "Error_Type": error_type,
                "Threshold": threshold,
                "Accuracy": result["Accuracy"],
                "Precision": result["Precision"],
                "Recall": result["Recall"],
                "F1_Score": result["F1_Score"],
                "TP": result["TP"],
                "FP": result["FP"],
                "TN": result["TN"],
                "FN": result["FN"],
            }
        )

        threshold_results.append(
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
        ] = result["Predictions"]

        # ----------------------------------------------------
        # FEATURE IMPORTANCE
        # ----------------------------------------------------

        importances = model.feature_importances_

        for feature, importance in zip(
            model_features,
            importances
        ):

            feature_importance_all.append(
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
            f"{result['Precision'] * 100:.2f}%"
        )

        print(
            f"Test Recall: "
            f"{result['Recall'] * 100:.2f}%"
        )

        print(
            f"Test F1: "
            f"{result['F1_Score'] * 100:.2f}%"
        )

    # --------------------------------------------------------
    # OVERALL ANY-ERROR PREDICTION
    # --------------------------------------------------------

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

    overall_prediction_matrix = (
        test_predictions[
            prediction_columns
        ]
        .to_numpy()
    )

    overall_prediction = (
        overall_prediction_matrix
        .max(axis=1)
        .astype(int)
    )

    overall_accuracy = accuracy_score(
        y_test_any,
        overall_prediction
    )

    overall_precision = precision_score(
        y_test_any,
        overall_prediction,
        zero_division=0
    )

    overall_recall = recall_score(
        y_test_any,
        overall_prediction,
        zero_division=0
    )

    overall_f1 = f1_score(
        y_test_any,
        overall_prediction,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test_any,
        overall_prediction,
        labels=[0, 1]
    ).ravel()

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        error_results
    )

    thresholds_df = pd.DataFrame(
        threshold_results
    )

    feature_importance_df = pd.DataFrame(
        feature_importance_all
    )

    overall_df = pd.DataFrame(
        [
            {
                "Model": "TYPE-AWARE HYBRID V5.6",
                "Accuracy": overall_accuracy,
                "Precision": overall_precision,
                "Recall": overall_recall,
                "F1_Score": overall_f1,
                "TP": int(tp),
                "FP": int(fp),
                "TN": int(tn),
                "FN": int(fn),
            }
        ]
    )

    results_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "type_aware_hybrid_v5_6_results.csv"
        ),
        index=False
    )

    thresholds_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "v5_6_threshold_optimization.csv"
        ),
        index=False
    )

    feature_importance_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "v5_6_random_forest_feature_importance.csv"
        ),
        index=False
    )

    overall_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "v5_6_overall_performance.csv"
        ),
        index=False
    )

    # Add stable test row identifiers.
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
    ] = overall_prediction

    test_predictions.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "v5_6_test_predictions.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # AUDIT FILE
    # --------------------------------------------------------

    audit = pd.DataFrame(
        {
            "Audit_Item": [
                "Split Strategy",
                "Context Statistics",
                "Missing Pattern Statistics",
                "Duplicate Evidence",
                "Threshold Optimization",
                "Test Set Usage",
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
                "TRAIN ONLY",
                "VALIDATION ONLY",
                "FINAL EVALUATION ONLY",
                "REMOVED",
                "REMOVED",
                "REMOVED",
                "ENABLED",
                "ENABLED",
                str(len(model_features)),
            ],
        }
    )

    audit.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "v5_6_methodology_audit.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "TYPE-AWARE HYBRID V5.6 RESULTS"
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
                "Precision": lambda x:
                    f"{x * 100:.2f}%",
                "Recall": lambda x:
                    f"{x * 100:.2f}%",
                "F1_Score": lambda x:
                    f"{x * 100:.2f}%",
            },
        )
    )

    print()
    print(
        "=" * 72
    )

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

    print(
        "=" * 72
    )

    print()
    print(
        "METHODOLOGY AUDIT"
    )
    print(
        "-" * 72
    )

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
        "Duplicate evidence             : "
        "TRAIN ONLY"
    )

    print(
        "Threshold optimization         : "
        "VALIDATION ONLY"
    )

    print(
        "Test set usage                 : "
        "FINAL EVALUATION ONLY"
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
        "V5.6 execution completed successfully."
    )