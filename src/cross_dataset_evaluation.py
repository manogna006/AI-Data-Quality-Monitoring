# ============================================================
# CROSS-DATASET EVALUATION V2.3
# ============================================================
#
# Generalized evaluation of the AI Data Quality Monitoring
# framework on heterogeneous tabular datasets.
#
# Datasets:
#   1. Bank Marketing
#   2. Telco Customer Churn
#
# Methodology:
#   - Controlled synthetic corruption
#   - Missing / Invalid / Outlier / Inconsistency / Duplicate
#   - Joint error-pattern stratified split
#   - 60% train / 20% validation / 20% test
#   - Identifier columns removed
#   - Target columns removed
#   - Numeric-like columns automatically converted
#   - Train-only contextual statistics
#   - Dataset-level unsupervised duplicate signal
#   - Isolation Forest trained on TRAIN only
#   - Type-specific Random Forest
#   - Validation-only threshold optimization
#   - Test used only for final evaluation
#
# V2.3 fixes:
#   - Correct duplicate ground truth
#   - Robust confusion matrix
#   - Fixed n_original scope bug
#   - Better contextual inconsistency corruption
#   - Correct output generation
#   - Per-dataset failure isolation
#   - Final summary always generated for completed datasets
#
# ============================================================

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    IsolationForest,
    RandomForestClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20
TEST_RATIO = 0.20

CORRUPTION_RATE = 0.01

ERROR_TYPES = [
    "Missing",
    "Invalid",
    "Outlier",
    "Inconsistency",
    "Duplicate"
]

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "cross_dataset_v2_3"
)

DATASETS = {

    "bank_marketing": {

        "path":
            BASE_DIR
            / "datasets"
            / "raw"
            / "bank+marketing"
            / "bank"
            / "bank-full.csv",

        "separator": ";",

        "target": "y"
    },

    "telco_customer_churn": {

        "path":
            BASE_DIR
            / "datasets"
            / "raw"
            / "telco_customer_churn"
            / "WA_Fn-UseC_-Telco-Customer-Churn.csv",

        "separator": ",",

        "target": "Churn"
    }
}


# ============================================================
# DIRECTORY
# ============================================================

def ensure_directory(path):

    path.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# NUMERIC-LIKE DETECTION
# ============================================================

def is_numeric_like(
    series,
    threshold=0.95
):

    if pd.api.types.is_numeric_dtype(series):
        return True

    converted = pd.to_numeric(
        series,
        errors="coerce"
    )

    non_null = series.notna().sum()

    if non_null == 0:
        return False

    ratio = (
        converted.notna().sum()
        /
        non_null
    )

    return ratio >= threshold


def convert_numeric_like_columns(df):

    df = df.copy()

    for col in df.columns:

        if is_numeric_like(df[col]):

            converted = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            if converted.notna().sum() > 0:

                df[col] = converted

    return df


# ============================================================
# IDENTIFIER DETECTION
# ============================================================

def detect_identifier_columns(
    df,
    target_column
):

    identifiers = []

    keywords = [
        "customerid",
        "customer_id",
        "accountid",
        "account_id",
        "userid",
        "user_id",
        "transactionid",
        "transaction_id",
        "recordid",
        "record_id",
        "invoiceid",
        "invoice_id",
        "identifier",
        "uuid"
    ]

    for col in df.columns:

        if col == target_column:
            continue

        normalized = (
            str(col)
            .lower()
            .replace(" ", "")
        )

        if any(
            key.replace("_", "")
            in normalized.replace("_", "")
            for key in keywords
        ):

            identifiers.append(col)
            continue

        unique_ratio = (
            df[col].nunique(
                dropna=False
            )
            /
            max(len(df), 1)
        )

        if (
            unique_ratio >= 0.995
            and
            not pd.api.types.is_numeric_dtype(
                df[col]
            )
        ):

            identifiers.append(col)

    return sorted(
        set(identifiers)
    )


# ============================================================
# ROW SELECTION
# ============================================================

def select_rows(
    available_indices,
    n_rows,
    rng,
    excluded_indices=None
):

    available = np.asarray(
        list(available_indices),
        dtype=int
    ).copy()

    if excluded_indices is None:
        excluded_indices = set()

    excluded = {
        int(x)
        for x in excluded_indices
    }

    if excluded:

        available = np.asarray(
            [
                x
                for x in available
                if int(x) not in excluded
            ],
            dtype=int
        )

    if len(available) < n_rows:

        raise ValueError(
            "Not enough unused rows for "
            "controlled corruption."
        )

    return np.asarray(
        rng.choice(
            available,
            size=n_rows,
            replace=False
        ),
        dtype=int
    )


# ============================================================
# COLUMN SELECTION
# ============================================================

def choose_numeric_column(
    df,
    feature_columns
):

    candidates = [
        col
        for col in feature_columns
        if pd.api.types.is_numeric_dtype(
            df[col]
        )
    ]

    if not candidates:
        return None

    useful = [
        col
        for col in candidates
        if df[col].nunique(
            dropna=True
        ) >= 5
    ]

    if useful:
        return useful[0]

    return candidates[0]


def choose_categorical_columns(
    df,
    feature_columns
):

    return [
        col
        for col in feature_columns
        if not pd.api.types.is_numeric_dtype(
            df[col]
        )
    ]


# ============================================================
# CONTEXTUAL PAIR SELECTION
# ============================================================

def choose_inconsistency_pair(
    df,
    feature_columns
):

    categorical_columns = (
        choose_categorical_columns(
            df,
            feature_columns
        )
    )

    if len(categorical_columns) < 2:

        return None, None

    best_pair = None
    best_score = -1

    for context_col in categorical_columns:

        for value_col in categorical_columns:

            if context_col == value_col:
                continue

            temp = df[
                [
                    context_col,
                    value_col
                ]
            ].dropna()

            if len(temp) < 100:
                continue

            # Number of distinct values of value_col
            # appearing within each context.
            diversity = (
                temp
                .groupby(context_col)[value_col]
                .nunique()
            )

            if len(diversity) == 0:
                continue

            # Prefer pairs with enough contextual structure.
            score = (
                diversity.mean()
                *
                np.log1p(
                    len(diversity)
                )
            )

            if score > best_score:

                best_score = score

                best_pair = (
                    context_col,
                    value_col
                )

    if best_pair is not None:

        return best_pair

    return (
        categorical_columns[0],
        categorical_columns[1]
    )


# ============================================================
# CORRUPTION GENERATION
# ============================================================

def generate_corruption(
    original_df,
    feature_columns,
    corruption_rate,
    random_state
):

    rng = np.random.default_rng(
        random_state
    )

    original = (
        original_df
        .copy()
        .reset_index(drop=True)
    )

    n_original = len(original)

    n_corrupt = max(
        1,
        int(
            round(
                n_original
                *
                corruption_rate
            )
        )
    )

    corrupted = original.copy()

    labels = pd.DataFrame(
        0,
        index=np.arange(n_original),
        columns=ERROR_TYPES,
        dtype=int
    )

    used_rows = set()

    corruption_info = {}

    # --------------------------------------------------------
    # MISSING
    # --------------------------------------------------------

    missing_candidates = [
        col
        for col in feature_columns
        if original[col].notna().sum() > 0
    ]

    missing_rows = select_rows(
        range(n_original),
        n_corrupt,
        rng,
        used_rows
    )

    used_rows.update(
        int(x)
        for x in missing_rows
    )

    if missing_candidates:

        missing_col = missing_candidates[0]

        corrupted.loc[
            missing_rows,
            missing_col
        ] = np.nan

        labels.loc[
            missing_rows,
            "Missing"
        ] = 1

        corruption_info[
            "missing_column"
        ] = missing_col

    else:

        corruption_info[
            "missing_column"
        ] = None

    # --------------------------------------------------------
    # INVALID
    # --------------------------------------------------------

    invalid_col = choose_numeric_column(
        original,
        feature_columns
    )

    invalid_rows = select_rows(
        range(n_original),
        n_corrupt,
        rng,
        used_rows
    )

    used_rows.update(
        int(x)
        for x in invalid_rows
    )

    if invalid_col is not None:

        values = pd.to_numeric(
            original[invalid_col],
            errors="coerce"
        )

        finite = values[
            np.isfinite(values)
        ]

        if len(finite) > 0:

            magnitude = float(
                finite.abs().quantile(
                    0.99
                )
            )

            if (
                not np.isfinite(magnitude)
                or magnitude <= 0
            ):

                magnitude = 1.0

            invalid_value = (
                magnitude
                *
                1000.0
                +
                1.0
            )

        else:

            invalid_value = 999999999.0

        corrupted.loc[
            invalid_rows,
            invalid_col
        ] = invalid_value

        labels.loc[
            invalid_rows,
            "Invalid"
        ] = 1

    corruption_info[
        "invalid_column"
    ] = invalid_col

    # --------------------------------------------------------
    # OUTLIER
    # --------------------------------------------------------

    outlier_col = choose_numeric_column(
        original,
        feature_columns
    )

    outlier_rows = select_rows(
        range(n_original),
        n_corrupt,
        rng,
        used_rows
    )

    used_rows.update(
        int(x)
        for x in outlier_rows
    )

    if outlier_col is not None:

        values = pd.to_numeric(
            original[outlier_col],
            errors="coerce"
        )

        median = values.median()

        if pd.isna(median):
            median = 1.0

        outlier_value = (
            median
            *
            100.0
        )

        if outlier_value == 0:
            outlier_value = 1000.0

        corrupted.loc[
            outlier_rows,
            outlier_col
        ] = outlier_value

        labels.loc[
            outlier_rows,
            "Outlier"
        ] = 1

    corruption_info[
        "outlier_column"
    ] = outlier_col

    # --------------------------------------------------------
    # INCONSISTENCY
    # --------------------------------------------------------

    context_col, value_col = (
        choose_inconsistency_pair(
            original,
            feature_columns
        )
    )

    inconsistency_rows = select_rows(
        range(n_original),
        n_corrupt,
        rng,
        used_rows
    )

    used_rows.update(
        int(x)
        for x in inconsistency_rows
    )

    corruption_info[
        "inconsistency_context"
    ] = context_col

    corruption_info[
        "inconsistency_value"
    ] = value_col

    if (
        context_col is not None
        and
        value_col is not None
    ):

        working = original[
            [
                context_col,
                value_col
            ]
        ].copy()

        working["_context"] = (
            working[context_col]
            .astype(str)
            .fillna("<NA>")
        )

        working["_value"] = (
            working[value_col]
            .astype(str)
            .fillna("<NA>")
        )

        global_values = (
            working["_value"]
            .unique()
            .tolist()
        )

        # Context -> value frequency.
        context_value_counts = (
            working
            .groupby(
                [
                    "_context",
                    "_value"
                ]
            )
            .size()
        )

        context_totals = (
            working
            .groupby("_context")
            .size()
        )

        for row_idx in inconsistency_rows:

            context_value = str(
                corrupted.at[
                    row_idx,
                    context_col
                ]
            )

            current_value = str(
                corrupted.at[
                    row_idx,
                    value_col
                ]
            )

            context_total = context_totals.get(
                context_value,
                0
            )

            candidate_values = []

            for candidate in global_values:

                candidate = str(candidate)

                if candidate == current_value:
                    continue

                count = context_value_counts.get(
                    (
                        context_value,
                        candidate
                    ),
                    0
                )

                if context_total > 0:

                    conditional_frequency = (
                        count
                        /
                        context_total
                    )

                else:

                    conditional_frequency = 0.0

                candidate_values.append(
                    (
                        candidate,
                        conditional_frequency
                    )
                )

            if candidate_values:

                # Prefer a value that is rare or absent
                # for the selected context.
                candidate_values.sort(
                    key=lambda x: x[1]
                )

                lowest_frequency = (
                    candidate_values[0][1]
                )

                low_candidates = [
                    value
                    for value, freq
                    in candidate_values
                    if freq
                    <=
                    max(
                        lowest_frequency,
                        0.001
                    )
                ]

                replacement = rng.choice(
                    low_candidates
                    if low_candidates
                    else [
                        x[0]
                        for x in candidate_values
                    ]
                )

                corrupted.at[
                    row_idx,
                    value_col
                ] = replacement

                labels.at[
                    row_idx,
                    "Inconsistency"
                ] = 1

    # --------------------------------------------------------
    # DUPLICATE
    # --------------------------------------------------------
    #
    # Exact copies are appended to the dataset.
    # Their ground-truth Duplicate label is explicitly 1.
    #
    # --------------------------------------------------------

    duplicate_source_indices = select_rows(
        range(n_original),
        n_corrupt,
        rng,
        used_rows
    )

    duplicate_rows = corrupted.iloc[
        duplicate_source_indices
    ].copy()

    duplicate_rows = duplicate_rows.reset_index(
        drop=True
    )

    corrupted = pd.concat(
        [
            corrupted,
            duplicate_rows
        ],
        ignore_index=True
    )

    duplicate_labels = pd.DataFrame(
        0,
        index=np.arange(
            len(duplicate_rows)
        ),
        columns=ERROR_TYPES,
        dtype=int
    )

    duplicate_labels[
        "Duplicate"
    ] = 1

    labels = pd.concat(
        [
            labels,
            duplicate_labels
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # INTERNAL ROW IDs
    # --------------------------------------------------------

    corrupted[
        "Corrupted_Row_ID"
    ] = np.arange(
        len(corrupted)
    )

    labels[
        "Corrupted_Row_ID"
    ] = np.arange(
        len(labels)
    )

    labels[
        "Original_Row_ID"
    ] = np.concatenate(
        [
            np.arange(
                n_original
            ),
            np.asarray(
                duplicate_source_indices,
                dtype=int
            )
        ]
    )

    return (
        corrupted,
        labels,
        corruption_info
    )


# ============================================================
# ERROR PATTERN
# ============================================================

def create_error_pattern_key(
    labels
):

    keys = []

    for _, row in labels.iterrows():

        active = [
            error
            for error in ERROR_TYPES
            if int(row[error]) == 1
        ]

        if active:

            keys.append(
                "+".join(active)
            )

        else:

            keys.append("Clean")

    return pd.Series(
        keys,
        index=labels.index
    )


# ============================================================
# STRATIFIED SPLIT
# ============================================================

def joint_stratified_split(
    labels,
    random_state
):

    pattern = create_error_pattern_key(
        labels
    )

    counts = pattern.value_counts()

    rare = set(
        counts[
            counts < 5
        ].index
    )

    pattern = pattern.map(
        lambda x:
            "__RARE__"
            if x in rare
            else x
    )

    indices = np.arange(
        len(labels)
    )

    train_idx, temp_idx = (
        train_test_split(
            indices,
            test_size=0.40,
            random_state=random_state,
            stratify=pattern
        )
    )

    temp_pattern = pattern.iloc[
        temp_idx
    ]

    val_idx, test_idx = (
        train_test_split(
            temp_idx,
            test_size=0.50,
            random_state=random_state,
            stratify=temp_pattern
        )
    )

    return (
        np.asarray(
            train_idx,
            dtype=int
        ),
        np.asarray(
            val_idx,
            dtype=int
        ),
        np.asarray(
            test_idx,
            dtype=int
        )
    )


# ============================================================
# TRAIN STATISTICS
# ============================================================

def build_train_statistics(
    train_df,
    feature_columns
):

    statistics = {}

    for col in feature_columns:

        series = train_df[col]

        if pd.api.types.is_numeric_dtype(
            series
        ):

            values = pd.to_numeric(
                series,
                errors="coerce"
            )

            statistics[col] = {

                "type":
                    "numeric",

                "median":
                    float(
                        values.median()
                    )
                    if values.notna().any()
                    else 0.0,

                "mean":
                    float(
                        values.mean()
                    )
                    if values.notna().any()
                    else 0.0,

                "std":
                    float(
                        values.std()
                    )
                    if values.notna().any()
                    else 1.0,

                "q1":
                    float(
                        values.quantile(
                            0.25
                        )
                    )
                    if values.notna().any()
                    else 0.0,

                "q3":
                    float(
                        values.quantile(
                            0.75
                        )
                    )
                    if values.notna().any()
                    else 0.0
            }

        else:

            values = (
                series
                .astype(str)
                .fillna("<NA>")
            )

            counts = values.value_counts()

            total = max(
                counts.sum(),
                1
            )

            frequencies = (
                counts
                /
                total
            ).to_dict()

            statistics[col] = {

                "type":
                    "categorical",

                "frequencies":
                    frequencies
            }

    return statistics


# ============================================================
# CONTEXT FEATURES
# ============================================================

def apply_train_statistics(
    df,
    feature_columns,
    statistics
):

    output = pd.DataFrame(
        index=df.index
    )

    for col in feature_columns:

        stats = statistics[col]

        if stats["type"] == "numeric":

            values = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            median = stats["median"]

            q1 = stats["q1"]
            q3 = stats["q3"]

            iqr = q3 - q1

            if iqr <= 0:

                iqr = max(
                    abs(
                        stats["std"]
                    ),
                    1.0
                )

            output[
                f"Context_{col}_Missing"
            ] = (
                values.isna()
                .astype(float)
            )

            output[
                f"Context_{col}_Deviation"
            ] = (
                (
                    values
                    -
                    median
                ).abs()
                /
                max(
                    abs(
                        stats["std"]
                    ),
                    1.0
                )
            ).fillna(0.0)

            output[
                f"Context_{col}_IQR_Outside"
            ] = (
                (
                    values
                    <
                    q1 - 1.5 * iqr
                )
                |
                (
                    values
                    >
                    q3 + 1.5 * iqr
                )
            ).astype(float)

        else:

            values = (
                df[col]
                .astype(str)
                .fillna("<NA>")
            )

            frequencies = (
                stats["frequencies"]
            )

            frequency = values.map(
                frequencies
            ).fillna(0.0)

            output[
                f"Context_{col}_Missing"
            ] = (
                df[col]
                .isna()
                .astype(float)
            )

            output[
                f"Context_{col}_Frequency"
            ] = frequency

            output[
                f"Context_{col}_Rare"
            ] = (
                frequency <= 0.01
            ).astype(float)

    return output


# ============================================================
# CONTEXTUAL PAIR STATISTICS
# ============================================================

def build_pair_statistics(
    train_df,
    context_col,
    value_col
):

    if (
        context_col is None
        or
        value_col is None
    ):

        return {}

    temp = train_df[
        [
            context_col,
            value_col
        ]
    ].copy()

    temp["_context"] = (
        temp[context_col]
        .astype(str)
        .fillna("<NA>")
    )

    temp["_value"] = (
        temp[value_col]
        .astype(str)
        .fillna("<NA>")
    )

    pair_counts = (
        temp
        .groupby(
            [
                "_context",
                "_value"
            ]
        )
        .size()
    )

    context_counts = (
        temp
        .groupby("_context")
        .size()
    )

    result = {}

    for key, count in pair_counts.items():

        context = key[0]

        total = context_counts.get(
            context,
            1
        )

        result[key] = (
            float(count)
            /
            max(
                float(total),
                1.0
            )
        )

    return result


def apply_pair_statistics(
    df,
    context_col,
    value_col,
    statistics
):

    if (
        context_col is None
        or
        value_col is None
    ):

        return pd.DataFrame(
            {
                "Contextual_Pair_Probability":
                    np.zeros(len(df)),

                "Contextual_Inconsistency":
                    np.zeros(len(df))
            },
            index=df.index
        )

    contexts = (
        df[context_col]
        .astype(str)
        .fillna("<NA>")
    )

    values = (
        df[value_col]
        .astype(str)
        .fillna("<NA>")
    )

    probabilities = []

    for context, value in zip(
        contexts,
        values
    ):

        probabilities.append(
            statistics.get(
                (
                    context,
                    value
                ),
                0.0
            )
        )

    probabilities = np.asarray(
        probabilities,
        dtype=float
    )

    return pd.DataFrame(
        {
            "Contextual_Pair_Probability":
                probabilities,

            "Contextual_Inconsistency":
                1.0
                -
                np.clip(
                    probabilities,
                    0.0,
                    1.0
                )
        },
        index=df.index
    )


# ============================================================
# BASE FEATURES
# ============================================================

def build_base_features(
    df,
    feature_columns
):

    output = pd.DataFrame(
        index=df.index
    )

    for col in feature_columns:

        if pd.api.types.is_numeric_dtype(
            df[col]
        ):

            values = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            median = values.median()

            if pd.isna(median):
                median = 0.0

            output[
                f"Num_{col}"
            ] = values.fillna(
                median
            )

            output[
                f"Num_{col}_Missing"
            ] = values.isna().astype(float)

            output[
                f"Num_{col}_Abs"
            ] = values.abs().fillna(
                abs(median)
            )

            output[
                f"Num_{col}_LogAbs"
            ] = np.log1p(
                values.abs()
            ).fillna(0.0)

        else:

            values = (
                df[col]
                .astype(str)
                .fillna("<NA>")
            )

            frequencies = (
                values.value_counts(
                    normalize=True
                )
            )

            output[
                f"Cat_{col}_Frequency"
            ] = values.map(
                frequencies
            ).fillna(0.0)

            output[
                f"Cat_{col}_Rare"
            ] = (
                output[
                    f"Cat_{col}_Frequency"
                ]
                <= 0.01
            ).astype(float)

            output[
                f"Cat_{col}_Missing"
            ] = df[col].isna().astype(float)

    return output


# ============================================================
# RULE FEATURES
# ============================================================

def build_rule_features(
    df,
    feature_columns,
    train_statistics
):

    output = pd.DataFrame(
        index=df.index
    )

    missing_count = (
        df[feature_columns]
        .isna()
        .sum(axis=1)
    )

    output[
        "Rule_Missing_Count"
    ] = missing_count.astype(float)

    output[
        "Rule_Missing_Flag"
    ] = (
        missing_count > 0
    ).astype(float)

    numeric_columns = [
        col
        for col in feature_columns
        if pd.api.types.is_numeric_dtype(
            df[col]
        )
    ]

    invalid_scores = np.zeros(
        len(df),
        dtype=float
    )

    outlier_scores = np.zeros(
        len(df),
        dtype=float
    )

    for col in numeric_columns:

        values = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        stats = train_statistics[col]

        q1 = stats["q1"]
        q3 = stats["q3"]

        iqr = q3 - q1

        if iqr <= 0:

            iqr = max(
                abs(
                    stats["std"]
                ),
                1.0
            )

        extreme_limit = (
            max(
                abs(q1),
                abs(q3),
                abs(stats["median"]),
                1.0
            )
            *
            100.0
        )

        invalid_scores += (
            values.abs()
            >
            extreme_limit
        ).astype(float).to_numpy()

        outlier_scores += (
            (
                values
                <
                q1 - 1.5 * iqr
            )
            |
            (
                values
                >
                q3 + 1.5 * iqr
            )
        ).astype(float).to_numpy()

    if numeric_columns:

        invalid_scores /= len(
            numeric_columns
        )

        outlier_scores /= len(
            numeric_columns
        )

    output[
        "Rule_Invalid_Score"
    ] = invalid_scores

    output[
        "Rule_Outlier_Score"
    ] = outlier_scores

    return output


# ============================================================
# DUPLICATE FEATURES
# ============================================================

def build_duplicate_signature(
    df,
    feature_columns
):

    if not feature_columns:

        return pd.Series(
            np.arange(
                len(df)
            ).astype(str),
            index=df.index
        )

    temp = df[
        feature_columns
    ].copy()

    for col in feature_columns:

        if not pd.api.types.is_numeric_dtype(
            temp[col]
        ):

            temp[col] = (
                temp[col]
                .astype(str)
                .fillna("<NA>")
                .str.strip()
            )

    return (
        temp
        .astype(str)
        .fillna("<NA>")
        .agg(
            "||".join,
            axis=1
        )
    )


def build_duplicate_features(
    df,
    feature_columns
):

    signature = build_duplicate_signature(
        df,
        feature_columns
    )

    counts = signature.value_counts()

    group_size = (
        signature
        .map(counts)
        .astype(float)
    )

    other_count = (
        group_size - 1.0
    ).clip(
        lower=0
    )

    exact_flag = (
        group_size > 1
    ).astype(float)

    log_size = np.log1p(
        group_size
    )

    strength = (
        other_count
        /
        np.maximum(
            group_size,
            1.0
        )
    )

    features = pd.DataFrame(
        {
            "Duplicate_Group_Size":
                group_size,

            "Duplicate_Other_Count":
                other_count,

            "Duplicate_Exact_Flag":
                exact_flag,

            "Duplicate_Log_Group_Size":
                log_size,

            "Duplicate_Strength":
                strength
        },
        index=df.index
    )

    return features


# ============================================================
# ENCODING
# ============================================================

def encode_features(
    train_features,
    validation_features,
    test_features
):

    train_encoded = pd.get_dummies(
        train_features,
        dummy_na=True
    )

    validation_encoded = pd.get_dummies(
        validation_features,
        dummy_na=True
    )

    test_encoded = pd.get_dummies(
        test_features,
        dummy_na=True
    )

    validation_encoded = (
        validation_encoded.reindex(
            columns=train_encoded.columns,
            fill_value=0
        )
    )

    test_encoded = (
        test_encoded.reindex(
            columns=train_encoded.columns,
            fill_value=0
        )
    )

    train_encoded = train_encoded.astype(
        np.float32
    )

    validation_encoded = (
        validation_encoded.astype(
            np.float32
        )
    )

    test_encoded = test_encoded.astype(
        np.float32
    )

    return (
        train_encoded,
        validation_encoded,
        test_encoded
    )


# ============================================================
# PROBABILITY
# ============================================================

def positive_probability(
    model,
    X
):

    probabilities = model.predict_proba(
        X
    )

    classes = list(
        model.classes_
    )

    if 1 not in classes:

        return np.zeros(
            len(X),
            dtype=float
        )

    positive_index = classes.index(
        1
    )

    return probabilities[
        :,
        positive_index
    ]


# ============================================================
# ISOLATION FOREST NORMALIZATION
# ============================================================

def normalize_if_scores(
    train_scores,
    other_scores
):

    low = np.percentile(
        train_scores,
        1
    )

    high = np.percentile(
        train_scores,
        99
    )

    denominator = max(
        high - low,
        1e-9
    )

    return np.clip(
        (
            other_scores
            -
            low
        )
        /
        denominator,
        0.0,
        1.0
    )


# ============================================================
# THRESHOLD
# ============================================================

def optimize_threshold(
    y_true,
    probabilities
):

    y_true = np.asarray(
        y_true,
        dtype=int
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float
    )

    if y_true.sum() == 0:

        return (
            0.50,
            0.0
        )

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in np.linspace(
        0.01,
        0.99,
        99
    ):

        predictions = (
            probabilities
            >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = float(
                threshold
            )

    return (
        best_threshold,
        best_f1
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    y_true = np.asarray(
        y_true,
        dtype=int
    )

    y_pred = np.asarray(
        y_pred,
        dtype=int
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    return {

        "Accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred
                )
            ),

        "Precision":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0
                )
            ),

        "Recall":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0
                )
            ),

        "F1_Score":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0
                )
            ),

        "True_Positive":
            int(tp),

        "False_Positive":
            int(fp),

        "True_Negative":
            int(tn),

        "False_Negative":
            int(fn)
    }


# ============================================================
# DATASET EVALUATION
# ============================================================

def evaluate_dataset(
    dataset_name,
    config
):

    print()
    print("=" * 70)
    print(
        f"DATASET: {dataset_name.upper()}"
    )
    print("=" * 70)
    print()

    dataset_output = (
        OUTPUT_DIR
        /
        dataset_name
    )

    ensure_directory(
        dataset_output
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print(
        "Loading dataset..."
    )

    original_df = pd.read_csv(
        config["path"],
        sep=config["separator"]
    )

    original_df.columns = [
        str(col).strip()
        for col in original_df.columns
    ]

    n_original = len(
        original_df
    )

    print(
        f"Original shape: "
        f"{original_df.shape}"
    )

    target_column = (
        config["target"]
    )

    print()
    print(
        f"Target column: "
        f"{target_column}"
    )

    # --------------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------------

    original_df = (
        convert_numeric_like_columns(
            original_df
        )
    )

    # --------------------------------------------------------
    # IDENTIFIERS
    # --------------------------------------------------------

    identifier_columns = (
        detect_identifier_columns(
            original_df,
            target_column
        )
    )

    print()
    print(
        "Excluded identifier columns:"
    )

    if identifier_columns:

        for col in identifier_columns:

            print(
                f"  - {col}"
            )

    else:

        print(
            "  None detected"
        )

    # --------------------------------------------------------
    # MODEL FEATURES
    # --------------------------------------------------------

    feature_columns = [
        col
        for col in original_df.columns
        if col != target_column
        and col not in identifier_columns
    ]

    numeric_columns = [
        col
        for col in feature_columns
        if pd.api.types.is_numeric_dtype(
            original_df[col]
        )
    ]

    categorical_columns = [
        col
        for col in feature_columns
        if col not in numeric_columns
    ]

    print()
    print(
        f"Model feature columns: "
        f"{len(feature_columns)}"
    )

    print(
        f"Numeric: "
        f"{len(numeric_columns)}"
    )

    print(
        f"Categorical: "
        f"{len(categorical_columns)}"
    )

    # --------------------------------------------------------
    # CORRUPTION
    # --------------------------------------------------------

    print()
    print(
        "Generating controlled corruption..."
    )

    (
        corrupted,
        labels,
        corruption_info
    ) = generate_corruption(
        original_df,
        feature_columns,
        CORRUPTION_RATE,
        RANDOM_STATE
    )

    print(
        f"Corrupted shape: "
        f"{corrupted.shape}"
    )

    print()
    print(
        "Synthetic error distribution:"
    )

    distribution_records = []

    for error_type in ERROR_TYPES:

        count = int(
            labels[
                error_type
            ].sum()
        )

        percentage = (
            count
            /
            len(labels)
            *
            100
        )

        print(
            f"  {error_type:<16}: "
            f"{count:6d} "
            f"({percentage:.3f}%)"
        )

        distribution_records.append(
            {
                "Dataset":
                    dataset_name,

                "Error_Type":
                    error_type,

                "Count":
                    count,

                "Percentage":
                    percentage
            }
        )

    pd.DataFrame(
        distribution_records
    ).to_csv(
        dataset_output
        /
        "corruption_distribution.csv",
        index=False
    )

    corrupted.to_csv(
        dataset_output
        /
        "corrupted.csv",
        index=False
    )

    labels.to_csv(
        dataset_output
        /
        "ground_truth.csv",
        index=False
    )

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    print()
    print(
        "Creating joint error-pattern "
        "stratified split..."
    )

    (
        train_idx,
        validation_idx,
        test_idx
    ) = joint_stratified_split(
        labels,
        RANDOM_STATE
    )

    print(
        f"Train: "
        f"{len(train_idx)}"
    )

    print(
        f"Validation: "
        f"{len(validation_idx)}"
    )

    print(
        f"Test: "
        f"{len(test_idx)}"
    )

    split_records = []

    for idx in train_idx:

        split_records.append(
            {
                "Corrupted_Row_ID":
                    int(idx),

                "Split":
                    "Train"
            }
        )

    for idx in validation_idx:

        split_records.append(
            {
                "Corrupted_Row_ID":
                    int(idx),

                "Split":
                    "Validation"
            }
        )

    for idx in test_idx:

        split_records.append(
            {
                "Corrupted_Row_ID":
                    int(idx),

                "Split":
                    "Test"
            }
        )

    pd.DataFrame(
        split_records
    ).to_csv(
        dataset_output
        /
        "split.csv",
        index=False
    )

    # --------------------------------------------------------
    # SPLIT DATA
    # --------------------------------------------------------

    train_df = corrupted.iloc[
        train_idx
    ].copy()

    validation_df = corrupted.iloc[
        validation_idx
    ].copy()

    test_df = corrupted.iloc[
        test_idx
    ].copy()

    train_labels = labels.iloc[
        train_idx
    ].copy()

    validation_labels = labels.iloc[
        validation_idx
    ].copy()

    test_labels = labels.iloc[
        test_idx
    ].copy()

    # --------------------------------------------------------
    # CONTEXTUAL PAIR
    # --------------------------------------------------------

    context_col = (
        corruption_info[
            "inconsistency_context"
        ]
    )

    value_col = (
        corruption_info[
            "inconsistency_value"
        ]
    )

    print()
    print(
        "Contextual pair:"
    )

    print(
        f"  Context: "
        f"{context_col}"
    )

    print(
        f"  Value:   "
        f"{value_col}"
    )

    pair_statistics = (
        build_pair_statistics(
            train_df,
            context_col,
            value_col
        )
    )

    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    print()
    print(
        "Duplicate signature columns:"
    )

    print(
        "  "
        +
        ", ".join(
            feature_columns
        )
    )

    duplicate_features = (
        build_duplicate_features(
            corrupted,
            feature_columns
        )
    )

    # --------------------------------------------------------
    # BASE FEATURES
    # --------------------------------------------------------

    train_base = (
        build_base_features(
            train_df,
            feature_columns
        )
    )

    validation_base = (
        build_base_features(
            validation_df,
            feature_columns
        )
    )

    test_base = (
        build_base_features(
            test_df,
            feature_columns
        )
    )

    # --------------------------------------------------------
    # TRAIN STATISTICS
    # --------------------------------------------------------

    train_statistics = (
        build_train_statistics(
            train_df,
            feature_columns
        )
    )

    train_context = (
        apply_train_statistics(
            train_df,
            feature_columns,
            train_statistics
        )
    )

    validation_context = (
        apply_train_statistics(
            validation_df,
            feature_columns,
            train_statistics
        )
    )

    test_context = (
        apply_train_statistics(
            test_df,
            feature_columns,
            train_statistics
        )
    )

    # --------------------------------------------------------
    # RULE FEATURES
    # --------------------------------------------------------

    train_rules = (
        build_rule_features(
            train_df,
            feature_columns,
            train_statistics
        )
    )

    validation_rules = (
        build_rule_features(
            validation_df,
            feature_columns,
            train_statistics
        )
    )

    test_rules = (
        build_rule_features(
            test_df,
            feature_columns,
            train_statistics
        )
    )

    # --------------------------------------------------------
    # PAIR FEATURES
    # --------------------------------------------------------

    train_pair = (
        apply_pair_statistics(
            train_df,
            context_col,
            value_col,
            pair_statistics
        )
    )

    validation_pair = (
        apply_pair_statistics(
            validation_df,
            context_col,
            value_col,
            pair_statistics
        )
    )

    test_pair = (
        apply_pair_statistics(
            test_df,
            context_col,
            value_col,
            pair_statistics
        )
    )

    # --------------------------------------------------------
    # DUPLICATE FEATURES
    # --------------------------------------------------------

    train_duplicate = (
        duplicate_features.iloc[
            train_idx
        ].copy()
    )

    validation_duplicate = (
        duplicate_features.iloc[
            validation_idx
        ].copy()
    )

    test_duplicate = (
        duplicate_features.iloc[
            test_idx
        ].copy()
    )

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    train_features = pd.concat(
        [
            train_base,
            train_context,
            train_rules,
            train_pair,
            train_duplicate
        ],
        axis=1
    )

    validation_features = pd.concat(
        [
            validation_base,
            validation_context,
            validation_rules,
            validation_pair,
            validation_duplicate
        ],
        axis=1
    )

    test_features = pd.concat(
        [
            test_base,
            test_context,
            test_rules,
            test_pair,
            test_duplicate
        ],
        axis=1
    )

    train_features = (
        train_features.loc[
            :,
            ~train_features.columns.duplicated()
        ]
    )

    validation_features = (
        validation_features.loc[
            :,
            ~validation_features.columns.duplicated()
        ]
    )

    test_features = (
        test_features.loc[
            :,
            ~test_features.columns.duplicated()
        ]
    )

    (
        X_train,
        X_validation,
        X_test
    ) = encode_features(
        train_features,
        validation_features,
        test_features
    )

    print()
    print(
        f"Feature matrix: "
        f"({len(corrupted)}, "
        f"{X_train.shape[1]})"
    )

    print(
        f"Encoded features: "
        f"{X_train.shape[1]}"
    )

    pd.DataFrame(
        {
            "Feature":
                X_train.columns
        }
    ).to_csv(
        dataset_output
        /
        "model_features.csv",
        index=False
    )

    # --------------------------------------------------------
    # ISOLATION FOREST
    # --------------------------------------------------------

    print()
    print(
        "Training Isolation Forest "
        "on TRAIN only..."
    )

    isolation_forest = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    isolation_forest.fit(
        X_train
    )

    train_if_raw = (
        -isolation_forest.score_samples(
            X_train
        )
    )

    validation_if_raw = (
        -isolation_forest.score_samples(
            X_validation
        )
    )

    test_if_raw = (
        -isolation_forest.score_samples(
            X_test
        )
    )

    validation_if = (
        normalize_if_scores(
            train_if_raw,
            validation_if_raw
        )
    )

    test_if = (
        normalize_if_scores(
            train_if_raw,
            test_if_raw
        )
    )

    # --------------------------------------------------------
    # ERROR MODELS
    # --------------------------------------------------------

    print()
    print(
        "Training type-specific models..."
    )

    probability_results = {}

    threshold_records = []

    error_records = []

    importance_records = []

    validation_probability_results = {}

    for error_type in ERROR_TYPES:

        print()
        print(
            f"--- {error_type} ---"
        )

        y_train = (
            train_labels[
                error_type
            ].to_numpy()
        )

        y_validation = (
            validation_labels[
                error_type
            ].to_numpy()
        )

        y_test = (
            test_labels[
                error_type
            ].to_numpy()
        )

        rf = RandomForestClassifier(
            n_estimators=300,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        rf.fit(
            X_train,
            y_train
        )

        rf_validation = (
            positive_probability(
                rf,
                X_validation
            )
        )

        rf_test = (
            positive_probability(
                rf,
                X_test
            )
        )

        # ----------------------------------------------------
        # RULE SIGNAL
        # ----------------------------------------------------

        if error_type == "Missing":

            validation_rule = (
                validation_rules[
                    "Rule_Missing_Flag"
                ].to_numpy()
            )

            test_rule = (
                test_rules[
                    "Rule_Missing_Flag"
                ].to_numpy()
            )

        elif error_type == "Invalid":

            validation_rule = (
                validation_rules[
                    "Rule_Invalid_Score"
                ].to_numpy()
            )

            test_rule = (
                test_rules[
                    "Rule_Invalid_Score"
                ].to_numpy()
            )

        elif error_type == "Outlier":

            validation_rule = (
                validation_rules[
                    "Rule_Outlier_Score"
                ].to_numpy()
            )

            test_rule = (
                test_rules[
                    "Rule_Outlier_Score"
                ].to_numpy()
            )

        elif error_type == "Inconsistency":

            validation_rule = (
                validation_pair[
                    "Contextual_Inconsistency"
                ].to_numpy()
            )

            test_rule = (
                test_pair[
                    "Contextual_Inconsistency"
                ].to_numpy()
            )

        else:

            validation_rule = (
                validation_duplicate[
                    "Duplicate_Exact_Flag"
                ].to_numpy()
            )

            test_rule = (
                test_duplicate[
                    "Duplicate_Exact_Flag"
                ].to_numpy()
            )

        # ----------------------------------------------------
        # HYBRID
        # ----------------------------------------------------

        if error_type == "Duplicate":

            validation_strength = (
                validation_duplicate[
                    "Duplicate_Strength"
                ].to_numpy()
            )

            test_strength = (
                test_duplicate[
                    "Duplicate_Strength"
                ].to_numpy()
            )

            validation_log = np.clip(
                validation_duplicate[
                    "Duplicate_Log_Group_Size"
                ].to_numpy()
                /
                np.log1p(10.0),
                0.0,
                1.0
            )

            test_log = np.clip(
                test_duplicate[
                    "Duplicate_Log_Group_Size"
                ].to_numpy()
                /
                np.log1p(10.0),
                0.0,
                1.0
            )

            validation_probability = (
                0.50 * rf_validation
                +
                0.35 * validation_rule
                +
                0.10 * validation_strength
                +
                0.05 * validation_log
            )

            test_probability = (
                0.50 * rf_test
                +
                0.35 * test_rule
                +
                0.10 * test_strength
                +
                0.05 * test_log
            )

        else:

            validation_probability = (
                0.70 * rf_validation
                +
                0.20 * validation_rule
                +
                0.10 * validation_if
            )

            test_probability = (
                0.70 * rf_test
                +
                0.20 * test_rule
                +
                0.10 * test_if
            )

        # ----------------------------------------------------
        # VALIDATION THRESHOLD
        # ----------------------------------------------------

        (
            threshold,
            validation_f1
        ) = optimize_threshold(
            y_validation,
            validation_probability
        )

        test_prediction = (
            test_probability
            >= threshold
        ).astype(int)

        metrics = calculate_metrics(
            y_test,
            test_prediction
        )

        print(
            f"Threshold: "
            f"{threshold:.2f}"
        )

        print(
            f"Validation F1: "
            f"{validation_f1 * 100:.2f}%"
        )

        print(
            f"Test Accuracy: "
            f"{metrics['Accuracy'] * 100:.2f}%"
        )

        print(
            f"Test Precision: "
            f"{metrics['Precision'] * 100:.2f}%"
        )

        print(
            f"Test Recall: "
            f"{metrics['Recall'] * 100:.2f}%"
        )

        print(
            f"Test F1: "
            f"{metrics['F1_Score'] * 100:.2f}%"
        )

        validation_probability_results[
            error_type
        ] = validation_probability

        probability_results[
            error_type
        ] = test_probability

        threshold_records.append(
            {
                "Dataset":
                    dataset_name,

                "Error_Type":
                    error_type,

                "Threshold":
                    threshold,

                "Validation_F1":
                    validation_f1
            }
        )

        error_records.append(
            {
                "Dataset":
                    dataset_name,

                "Error_Type":
                    error_type,

                **metrics,

                "Threshold":
                    threshold,

                "Validation_F1":
                    validation_f1
            }
        )

        for feature, importance in zip(
            X_train.columns,
            rf.feature_importances_
        ):

            importance_records.append(
                {
                    "Dataset":
                        dataset_name,

                    "Error_Type":
                        error_type,

                    "Feature":
                        feature,

                    "Importance":
                        float(importance)
                }
            )

    # --------------------------------------------------------
    # OVERALL
    # --------------------------------------------------------

    validation_probability_matrix = (
        np.column_stack(
            [
                validation_probability_results[
                    error_type
                ]
                for error_type in ERROR_TYPES
            ]
        )
    )

    test_probability_matrix = (
        np.column_stack(
            [
                probability_results[
                    error_type
                ]
                for error_type in ERROR_TYPES
            ]
        )
    )

    validation_overall_probability = (
        np.max(
            validation_probability_matrix,
            axis=1
        )
    )

    test_overall_probability = (
        np.max(
            test_probability_matrix,
            axis=1
        )
    )

    validation_any_true = (
        validation_labels[
            ERROR_TYPES
        ]
        .sum(axis=1)
        > 0
    ).astype(int).to_numpy()

    test_any_true = (
        test_labels[
            ERROR_TYPES
        ]
        .sum(axis=1)
        > 0
    ).astype(int).to_numpy()

    # --------------------------------------------------------
    # Overall threshold is optimized on validation only.
    # --------------------------------------------------------

    overall_threshold, overall_validation_f1 = (
        optimize_threshold(
            validation_any_true,
            validation_overall_probability
        )
    )

    overall_prediction = (
        test_overall_probability
        >= overall_threshold
    ).astype(int)

    overall_metrics = calculate_metrics(
        test_any_true,
        overall_prediction
    )

    print()
    print("=" * 70)
    print(
        f"OVERALL RESULT: "
        f"{dataset_name.upper()}"
    )
    print("=" * 70)

    print(
        f"Overall threshold: "
        f"{overall_threshold:.2f}"
    )

    print(
        f"Validation F1: "
        f"{overall_validation_f1 * 100:.2f}%"
    )

    print(
        f"Accuracy : "
        f"{overall_metrics['Accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{overall_metrics['Precision'] * 100:.2f}%"
    )

    print(
        f"Recall   : "
        f"{overall_metrics['Recall'] * 100:.2f}%"
    )

    print(
        f"F1 Score : "
        f"{overall_metrics['F1_Score'] * 100:.2f}%"
    )

    print(
        f"TP={overall_metrics['True_Positive']} "
        f"FP={overall_metrics['False_Positive']} "
        f"TN={overall_metrics['True_Negative']} "
        f"FN={overall_metrics['False_Negative']}"
    )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    prediction_output = pd.DataFrame(
        {
            "Corrupted_Row_ID":
                test_labels[
                    "Corrupted_Row_ID"
                ].to_numpy(),

            "Original_Row_ID":
                test_labels[
                    "Original_Row_ID"
                ].to_numpy()
        }
    )

    for error_type in ERROR_TYPES:

        threshold = next(
            record["Threshold"]
            for record in threshold_records
            if record["Error_Type"]
            == error_type
        )

        prediction_output[
            f"{error_type}_Probability"
        ] = probability_results[
            error_type
        ]

        prediction_output[
            f"{error_type}_Prediction"
        ] = (
            probability_results[
                error_type
            ]
            >= threshold
        ).astype(int)

        prediction_output[
            f"{error_type}_True"
        ] = test_labels[
            error_type
        ].to_numpy()

    prediction_output[
        "Any_Error_Probability"
    ] = test_overall_probability

    prediction_output[
        "Any_Error_Prediction"
    ] = overall_prediction

    prediction_output[
        "Any_Error_True"
    ] = test_any_true

    prediction_output.to_csv(
        dataset_output
        /
        "predictions.csv",
        index=False
    )

    # --------------------------------------------------------
    # OUTPUT TABLES
    # --------------------------------------------------------

    pd.DataFrame(
        error_records
    ).to_csv(
        dataset_output
        /
        "error_type_performance.csv",
        index=False
    )

    pd.DataFrame(
        threshold_records
    ).to_csv(
        dataset_output
        /
        "threshold_optimization.csv",
        index=False
    )

    pd.DataFrame(
        importance_records
    ).to_csv(
        dataset_output
        /
        "random_forest_feature_importance.csv",
        index=False
    )

    overall_record = {
        "Dataset":
            dataset_name,

        "Accuracy":
            overall_metrics["Accuracy"],

        "Precision":
            overall_metrics["Precision"],

        "Recall":
            overall_metrics["Recall"],

        "F1_Score":
            overall_metrics["F1_Score"],

        "True_Positive":
            overall_metrics["True_Positive"],

        "False_Positive":
            overall_metrics["False_Positive"],

        "True_Negative":
            overall_metrics["True_Negative"],

        "False_Negative":
            overall_metrics["False_Negative"],

        "Overall_Threshold":
            overall_threshold,

        "Validation_F1":
            overall_validation_f1
    }

    pd.DataFrame(
        [overall_record]
    ).to_csv(
        dataset_output
        /
        "overall_performance.csv",
        index=False
    )

    # --------------------------------------------------------
    # SCHEMA REPORT
    # --------------------------------------------------------

    schema_report = {

        "Dataset":
            dataset_name,

        "Original_Rows":
            int(n_original),

        "Corrupted_Rows":
            int(len(corrupted)),

        "Target_Column":
            target_column,

        "Identifier_Columns":
            identifier_columns,

        "Model_Feature_Columns":
            feature_columns,

        "Numeric_Columns":
            numeric_columns,

        "Categorical_Columns":
            categorical_columns,

        "Context_Column":
            context_col,

        "Context_Value_Column":
            value_col,

        "Duplicate_Signature_Columns":
            feature_columns
    }

    with open(
        dataset_output
        /
        "schema_report.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            schema_report,
            file,
            indent=4,
            default=str
        )

    # --------------------------------------------------------
    # METHODOLOGY AUDIT
    # --------------------------------------------------------

    audit_records = [

        {
            "Item":
                "Split Strategy",

            "Value":
                "JOINT ERROR-PATTERN STRATIFIED"
        },

        {
            "Item":
                "Train Split",

            "Value":
                "60%"
        },

        {
            "Item":
                "Validation Split",

            "Value":
                "20%"
        },

        {
            "Item":
                "Test Split",

            "Value":
                "20%"
        },

        {
            "Item":
                "Context Statistics",

            "Value":
                "TRAIN ONLY"
        },

        {
            "Item":
                "Contextual Pair Statistics",

            "Value":
                "TRAIN ONLY"
        },

        {
            "Item":
                "Isolation Forest",

            "Value":
                "TRAIN ONLY"
        },

        {
            "Item":
                "Random Forest",

            "Value":
                "TRAIN ONLY"
        },

        {
            "Item":
                "Error Thresholds",

            "Value":
                "VALIDATION ONLY"
        },

        {
            "Item":
                "Overall Threshold",

            "Value":
                "VALIDATION ONLY"
        },

        {
            "Item":
                "Test Usage",

            "Value":
                "FINAL EVALUATION ONLY"
        },

        {
            "Item":
                "Identifier Features",

            "Value":
                "REMOVED"
        },

        {
            "Item":
                "Target Feature",

            "Value":
                "REMOVED"
        },

        {
            "Item":
                "Synthetic Artifact Flag",

            "Value":
                "NOT USED"
        },

        {
            "Item":
                "Duplicate Signal",

            "Value":
                "UNSUPERVISED DATASET-LEVEL"
        },

        {
            "Item":
                "Duplicate Ground Truth Used By Model",

            "Value":
                "NO"
        },

        {
            "Item":
                "Duplicate Labels",

            "Value":
                "EXPLICITLY GENERATED FOR APPENDED DUPLICATES"
        },

        {
            "Item":
                "Controlled Corruption Rate",

            "Value":
                "1% PER ERROR TYPE"
        },

        {
            "Item":
                "Error Types",

            "Value":
                ", ".join(ERROR_TYPES)
        }
    ]

    pd.DataFrame(
        audit_records
    ).to_csv(
        dataset_output
        /
        "methodology_audit.csv",
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY JSON
    # --------------------------------------------------------

    result = {

        "dataset":
            dataset_name,

        "original_rows":
            int(n_original),

        "corrupted_rows":
            int(len(corrupted)),

        "train_rows":
            int(len(train_idx)),

        "validation_rows":
            int(len(validation_idx)),

        "test_rows":
            int(len(test_idx)),

        "feature_count":
            int(
                X_train.shape[1]
            ),

        "overall":
            overall_record,

        "error_type_results":
            error_records,

        "thresholds":
            threshold_records,

        "identifier_columns":
            identifier_columns,

        "corruption_info":
            corruption_info
    }

    with open(
        dataset_output
        /
        "summary.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=4,
            default=str
        )

    print()
    print(
        "Saved results to:"
    )

    print(
        f"  {dataset_output}"
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "CROSS-DATASET EVALUATION V2.3"
    )
    print("=" * 70)

    print()
    print(
        "Methodology:"
    )

    print(
        "  • Controlled synthetic corruption"
    )

    print(
        "  • Joint error-pattern stratified split"
    )

    print(
        "  • 60% train / 20% validation / 20% test"
    )

    print(
        "  • Identifier columns removed"
    )

    print(
        "  • Target columns removed"
    )

    print(
        "  • Numeric-like columns automatically converted"
    )

    print(
        "  • Train-only contextual statistics"
    )

    print(
        "  • Dataset-level unsupervised duplicate signal"
    )

    print(
        "  • Isolation Forest trained on train only"
    )

    print(
        "  • Type-specific Random Forest"
    )

    print(
        "  • Validation-only threshold optimization"
    )

    print(
        "  • Test used only for final evaluation"
    )

    print(
        "  • Duplicate ground truth correctly assigned"
    )

    print(
        "  • Overall threshold optimized on validation"
    )

    ensure_directory(
        OUTPUT_DIR
    )

    completed_results = []

    failed_datasets = []

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    for dataset_name, config in DATASETS.items():

        try:

            result = evaluate_dataset(
                dataset_name,
                config
            )

            completed_results.append(
                result
            )

        except Exception as error:

            failed_datasets.append(
                {
                    "Dataset":
                        dataset_name,

                    "Error_Type":
                        type(error).__name__,

                    "Error":
                        str(error)
                }
            )

            print()
            print("=" * 70)
            print(
                f"ERROR IN DATASET: "
                f"{dataset_name}"
            )
            print("=" * 70)

            print(
                f"{type(error).__name__}: "
                f"{error}"
            )

            print()
            print(
                "The remaining datasets "
                "will still be attempted."
            )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    if completed_results:

        summary_records = []

        for result in completed_results:

            overall = result[
                "overall"
            ]

            summary_records.append(
                {
                    "Dataset":
                        result["dataset"],

                    "Accuracy":
                        overall["Accuracy"],

                    "Precision":
                        overall["Precision"],

                    "Recall":
                        overall["Recall"],

                    "F1_Score":
                        overall["F1_Score"],

                    "TP":
                        overall[
                            "True_Positive"
                        ],

                    "FP":
                        overall[
                            "False_Positive"
                        ],

                    "TN":
                        overall[
                            "True_Negative"
                        ],

                    "FN":
                        overall[
                            "False_Negative"
                        ],

                    "Overall_Threshold":
                        overall[
                            "Overall_Threshold"
                        ],

                    "Validation_F1":
                        overall[
                            "Validation_F1"
                        ]
                }
            )

        summary_df = pd.DataFrame(
            summary_records
        )

        summary_df.to_csv(
            OUTPUT_DIR
            /
            "cross_dataset_overall_summary.csv",
            index=False
        )

        all_error_records = []

        for result in completed_results:

            all_error_records.extend(
                result[
                    "error_type_results"
                ]
            )

        pd.DataFrame(
            all_error_records
        ).to_csv(
            OUTPUT_DIR
            /
            "cross_dataset_error_type_summary.csv",
            index=False
        )

    else:

        summary_df = pd.DataFrame()

    # --------------------------------------------------------
    # FAILURE REPORT
    # --------------------------------------------------------

    pd.DataFrame(
        failed_datasets
    ).to_csv(
        OUTPUT_DIR
        /
        "failed_datasets.csv",
        index=False
    )

    # --------------------------------------------------------
    # COMPLETE JSON
    # --------------------------------------------------------

    final_summary = {

        "completed_datasets":
            [
                result["dataset"]
                for result in completed_results
            ],

        "failed_datasets":
            failed_datasets,

        "results":
            completed_results
    }

    with open(
        OUTPUT_DIR
        /
        "cross_dataset_summary.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            final_summary,
            file,
            indent=4,
            default=str
        )

    # --------------------------------------------------------
    # PRINT FINAL RESULTS
    # --------------------------------------------------------

    print()
    print()
    print("=" * 70)
    print(
        "FINAL CROSS-DATASET SUMMARY"
    )
    print("=" * 70)

    if len(summary_df) == 0:

        print()
        print(
            "No datasets completed successfully."
        )

    else:

        for _, row in summary_df.iterrows():

            print()
            print(
                row["Dataset"].upper()
            )

            print(
                f"  Accuracy : "
                f"{row['Accuracy'] * 100:.2f}%"
            )

            print(
                f"  Precision: "
                f"{row['Precision'] * 100:.2f}%"
            )

            print(
                f"  Recall   : "
                f"{row['Recall'] * 100:.2f}%"
            )

            print(
                f"  F1 Score : "
                f"{row['F1_Score'] * 100:.2f}%"
            )

            print(
                f"  TP={int(row['TP'])} "
                f"FP={int(row['FP'])} "
                f"TN={int(row['TN'])} "
                f"FN={int(row['FN'])}"
            )

    # --------------------------------------------------------
    # FAILED DATASETS
    # --------------------------------------------------------

    if failed_datasets:

        print()
        print(
            "Failed datasets:"
        )

        for failure in failed_datasets:

            print(
                f"  - "
                f"{failure['Dataset']}: "
                f"{failure['Error_Type']} - "
                f"{failure['Error']}"
            )

    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "CROSS-DATASET EVALUATION COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        "Results directory:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()