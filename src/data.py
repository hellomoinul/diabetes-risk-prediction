"""Dataset loading, cleaning, and leakage-safe train/validation/test splitting.

The dataset is the Kaggle "Diabetes prediction dataset"
(``iammustafatz/diabetes-prediction-dataset``): 100,000 patient records with a
mixture of numeric and categorical clinical features and a binary ``diabetes``
target that is positive for only 8.5% of patients.

Design decisions made during profiling (see the notebook for the evidence):

* **Duplicate rows are dropped** -- 3,854 full-row duplicates (3.85%), and only
  one duplicate pair disagrees on the label, so removing them loses no
  information.
* **``smoking_history == "No Info"`` is kept as its own category**, not imputed.
  It covers 35.8% of rows but has a 4.06% diabetes rate against 9.5-17.0% for
  every real smoking category, so it is systematic rather than random
  missingness and imputing it would destroy real signal.
* **No age cleaning.** The data contains implausible ages (a minimum of 0.08
  years, and 17,219 rows under 18), which indicates a synthetic generator. We
  document this as a limitation instead of silently manipulating 17% of the
  rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .fetch_data import (
    CATEGORICAL_FEATURES,
    COLUMNS,
    DATA_FILE,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
)

PROJECT_ROOT = DATA_FILE.parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42

__all__ = [
    "TARGET",
    "FEATURES",
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
    "DATA_FILE",
    "RANDOM_STATE",
    "MODELS_DIR",
    "load_data",
    "drop_duplicates",
    "summary_statistics",
    "missing_value_report",
    "class_balance",
    "category_breakdown",
    "duplicate_report",
    "data_quality_report",
    "make_splits",
    "build_preprocessor",
    "build_pipeline",
    "majority_baseline",
    "Splits",
]


def load_data(path=DATA_FILE) -> pd.DataFrame:
    """Load the raw dataset exactly as downloaded (no cleaning applied)."""
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"{path} is missing the '{TARGET}' column")
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove full-row duplicate records."""
    return df.drop_duplicates().reset_index(drop=True)


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics for the numeric clinical features."""
    return df[NUMERIC_FEATURES].describe().T


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Count missingness, including the ``No Info`` sentinel category.

    A naive null check reports nothing here: the file has zero NaN cells.
    All the missingness is encoded as the string ``"No Info"`` inside
    ``smoking_history``, which is why a plain ``isnull().sum()`` is misleading
    on this dataset.
    """
    rows = []
    for col in COLUMNS:
        n_nan = int(df[col].isna().sum())
        sentinel = (
            int((df[col] == "No Info").sum()) if col in CATEGORICAL_FEATURES else 0
        )
        rows.append(
            {
                "Feature": col,
                "NaN cells": n_nan,
                "'No Info' sentinel": sentinel,
                "Total missing": n_nan + sentinel,
            }
        )
    return pd.DataFrame(rows)


def class_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Counts, proportions, and the resulting majority-class baseline."""
    counts = df[TARGET].value_counts().sort_index()
    out = pd.DataFrame(
        {
            "Diagnosis": ["Negative (no diabetes)", "Positive (diabetes)"],
            "Count": counts.values,
        }
    )
    out["Percent"] = (100 * out["Count"] / len(df)).round(2)
    out["Ratio"] = counts.max() / counts.min()
    return out


def majority_baseline(y) -> float:
    """Accuracy of the trivial 'always predict the majority class' rule.

    With an 8.5% positive rate this sits near 0.915, so accuracy on its own is
    close to meaningless and every model must be read against this number.
    """
    y = np.asarray(y)
    return float(np.bincount(y).max() / len(y))


def category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Diabetes rate and counts for every categorical and binary feature.

    This is what reveals that ``"No Info"`` is systematic rather than random:
    its diabetes rate is far below every genuine smoking category.
    """
    rows = []
    for col in CATEGORICAL_FEATURES + ["hypertension", "heart_disease"]:
        grouped = df.groupby(col, observed=True)[TARGET].agg(["size", "mean"])
        for value, row in grouped.iterrows():
            rows.append(
                {
                    "Feature": col,
                    "Value": value,
                    "Count": int(row["size"]),
                    "Diabetes %": round(100 * row["mean"], 2),
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["Feature", "Diabetes %"], ascending=[True, False])


def duplicate_report(df: pd.DataFrame) -> pd.DataFrame:
    """Quantify exact and feature-level duplicate records."""
    feature_cols = [c for c in COLUMNS if c != TARGET]
    feature_dupes = int(df.duplicated(subset=feature_cols).sum())

    conflicting = 0
    dup_mask = df.duplicated(keep=False)
    if dup_mask.any():
        groups = df[dup_mask].groupby(feature_cols)[TARGET].nunique()
        conflicting = int((groups > 1).sum())

    return pd.DataFrame(
        {
            "Check": [
                "Full-row duplicates",
                "Duplicates ignoring target",
                "Rows in a duplicate group",
                "Groups with conflicting labels",
            ],
            "Count": [
                int(df.duplicated().sum()),
                feature_dupes,
                int(dup_mask.sum()),
                conflicting,
            ],
        }
    )


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Flag physiologically suspect values found during profiling.

    None of these are corrected. They are reported so the limitation is visible
    in the write-up rather than hidden.
    """
    checks = [
        ("age below 18", int((df["age"] < 18).sum())),
        ("age below 1 year", int((df["age"] < 1).sum())),
        ("age above 75", int((df["age"] > 75).sum())),
        ("bmi below 15", int((df["bmi"] < 15).sum())),
        ("bmi above 60", int((df["bmi"] > 60).sum())),
        ("bmi == 0", int((df["bmi"] == 0).sum())),
        ("HbA1c above 10", int((df["HbA1c_level"] > 10).sum())),
        ("blood glucose below 70", int((df["blood_glucose_level"] < 70).sum())),
        ("gender == 'Other'", int((df["gender"] == "Other").sum())),
    ]
    out = pd.DataFrame(checks, columns=["Check", "Rows"])
    out["Percent of data"] = (100 * out["Rows"] / len(df)).round(2)
    return out


@dataclass
class Splits:
    """Container for the three partitions, all still untransformed."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def make_splits(
    df: pd.DataFrame,
    test_size: float = 0.30,
    val_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> Splits:
    """Split into 70/15/15 train/validation/test, stratified on the target.

    ``test_size`` is the fraction of the full dataset held out as a temporary
    pool; ``val_size`` is carved out of that pool, so the final test fraction
    is ``test_size - val_size``. The defaults give 70 / 15 / 15.

    With ~96,000 rows this leaves roughly 1,280 positive patients in the test
    set, which is enough for model differences to be statistically meaningful.
    """
    if not 0 < val_size <= test_size < 1:
        raise ValueError(
            f"need 0 < val_size <= test_size < 1, got val_size={val_size}, "
            f"test_size={test_size}"
        )

    X, y = df[FEATURES], df[TARGET]

    X_train, X_pool, y_train, y_pool = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_pool, y_pool, test_size=val_size / test_size, stratify=y_pool,
        random_state=random_state,
    )

    return Splits(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
    )


def build_preprocessor(
    numeric: list[str] | None = None,
    categorical: list[str] | None = None,
) -> ColumnTransformer:
    """Assemble the two-path cleaning + encoding + scaling transformer.

    * **Numeric path** -- median imputation then standardisation. Median is used
      rather than mean because ``bmi`` and ``blood_glucose_level`` are
      right-skewed. Scaling is required by Logistic Regression and SVM and
      harmless for the tree models; applying it uniformly keeps the comparison
      fair.
    * **Categorical path** -- most-frequent imputation then one-hot encoding.
      One-hot columns are deliberately *not* scaled, since a 0/1 indicator
      should stay on its natural scale.

    Every step is a fitted sklearn estimator, so calling ``fit`` on the
    training split only means no statistic from the validation or test set can
    leak backwards into training.
    """
    numeric = list(numeric if numeric is not None else NUMERIC_FEATURES)
    categorical = list(categorical if categorical is not None else CATEGORICAL_FEATURES)

    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop=None)),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(model, features: list[str] | None = None) -> Pipeline:
    """Wrap a classifier so it receives preprocessed features."""
    if features is None:
        return Pipeline(
            [("preprocess", build_preprocessor()), ("clf", model)]
        )

    numeric = [c for c in features if c in NUMERIC_FEATURES]
    categorical = [c for c in features if c in CATEGORICAL_FEATURES]
    return Pipeline(
        [("preprocess", build_preprocessor(numeric, categorical)), ("clf", model)]
    )
