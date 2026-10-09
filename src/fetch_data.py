"""Download the Kaggle "Diabetes prediction dataset" into ``data/``.

Source: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset
Owner: Mohammed Mustafa. Distributed by Kaggle.

The download uses the official Kaggle API, authenticated from the credentials
in ``~/.kaggle/``. Whatever comes back is validated before it is accepted, so a
truncated or unexpected file fails loudly instead of silently poisoning the
pipeline.

Run directly::

    python -m src.fetch_data          # download if missing
    python -m src.fetch_data --force  # re-download regardless
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_FILE = DATA_DIR / "diabetes_prediction_dataset.csv"

KAGGLE_SLUG = "iammustafatz/diabetes-prediction-dataset"
KAGGLE_URL = f"https://www.kaggle.com/datasets/{KAGGLE_SLUG}"

TARGET = "diabetes"

# Order matters: it fixes the column order used everywhere downstream.
COLUMNS = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
    TARGET,
]

NUMERIC_FEATURES = [
    "age",
    "hypertension",
    "heart_disease",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]

CATEGORICAL_FEATURES = ["gender", "smoking_history"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

EXPECTED_ROWS = 100_000
EXPECTED_CATEGORIES = {
    "gender": {"Female", "Male", "Other"},
    "smoking_history": {
        "never", "former", "current", "not current", "ever", "No Info",
    },
}


def _validate(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Reject anything that does not match the documented Kaggle schema."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{source}: missing columns {missing}")

    df = df[COLUMNS].copy()

    for col in NUMERIC_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"{source}: expected {EXPECTED_ROWS} rows, got {len(df)}")

    if df[NUMERIC_FEATURES + [TARGET]].isna().any().any():
        raise ValueError(f"{source}: numeric columns contain unparseable values")

    labels = set(df[TARGET].unique())
    if not labels.issubset({0, 1}):
        raise ValueError(f"{source}: target must be binary 0/1, found {labels}")

    for col, expected in EXPECTED_CATEGORIES.items():
        found = set(df[col].dropna().unique())
        if not found.issubset(expected):
            raise ValueError(
                f"{source}: unexpected {col} values "
                f"{sorted(found - expected)} (expected {sorted(expected)})"
            )

    return df


def _read(path: Path) -> pd.DataFrame:
    return _validate(pd.read_csv(path), str(path))


def fetch(force: bool = False) -> pd.DataFrame:
    """Return the dataset, downloading it only if it is not already on disk.

    The local file is preferred, so a machine (or a Colab runtime) that already
    has ``data/diabetes_prediction_dataset.csv`` never needs Kaggle credentials
    or network access.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_FILE.exists() and not force:
        df = _read(DATA_FILE)
        print(f"Using existing {DATA_FILE.relative_to(PROJECT_ROOT)} "
              f"({len(df):,} rows)")
        return df

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError(
            f"{DATA_FILE.name} is not present and the kaggle package is "
            "required to download it. Install it with: pip install kaggle"
        ) from exc

    # Fail with an actionable message rather than an opaque auth error, which
    # is the common failure when the project is uploaded without the data file.
    if not (Path.home() / ".kaggle" / "kaggle.json").exists() and not (
        os.environ.get("KAGGLE_USERNAME")
    ):
        raise RuntimeError(
            f"{DATA_FILE.name} is missing and no Kaggle credentials were found.\n"
            "Either place the CSV at that path, or set up credentials:\n"
            "  - download kaggle.json from kaggle.com/settings and save it to\n"
            "    ~/.kaggle/kaggle.json, or\n"
            "  - export KAGGLE_USERNAME and KAGGLE_KEY."
        )

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading {KAGGLE_SLUG} from the Kaggle API ...")
    api.dataset_download_files(KAGGLE_SLUG, path=str(DATA_DIR), unzip=True)

    if not DATA_FILE.exists():
        candidates = [p for p in DATA_DIR.glob("*.csv")]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected {DATA_FILE.name} after unzip; found {candidates}"
            )
        candidates[0].rename(DATA_FILE)

    df = _read(DATA_FILE)
    print(f"Validated {len(df):,} rows x {df.shape[1]} columns from {KAGGLE_URL}")
    print(f"Saved to {DATA_FILE.relative_to(PROJECT_ROOT)}")
    return df


if __name__ == "__main__":
    data = fetch(force="--force" in sys.argv)
    print()
    print(f"Shape        : {data.shape}")
    print(f"Positive rate: {data[TARGET].mean():.4f}")
    print(f"Duplicates   : {int(data.duplicated().sum()):,}")
