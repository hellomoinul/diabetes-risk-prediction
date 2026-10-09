"""Score a new dataset with a model trained by the pipeline.

The saved artifact is the *whole* fitted pipeline -- the preprocessing plus the
classifier -- so the CSV you pass in must contain the raw feature columns (not
pre-encoded ones). Any of the trained models can be used::

    python predict.py newdata.csv                      # XGBoost (best AUC)
    python predict.py newdata.csv --model decision_tree
    python predict.py newdata.csv --out scored.csv
    python predict.py exam.csv --models-dir results_exam/models

The feature columns are read from the model itself, so this works for both the
diabetes models in ``models/`` and the ones written by
``run_generic_pipeline.py``.

The output is the input rows with two extra columns: ``prediction`` (the 0/1
label) and ``probability`` (positive-class score; for SVM, which has no
``predict_proba``, the column is ``decision_score``). If the CSV also carries
the target column, the metrics are printed as a bonus.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import FEATURES, MODELS_DIR, TARGET
from src.models import model_slug

DEFAULT_MODEL = "xgboost"


def _as_labels(series: pd.Series) -> pd.Series | None:
    """Return a 0/1 int target, or None if it is not a clean binary column."""
    values = list(pd.unique(series.dropna()))
    if set(values) <= {0, 1}:
        return series.astype(int)
    if len(values) == 2:
        positive = series.value_counts().idxmin()
        negative = next(v for v in values if v != positive)
        return series.map({positive: 1, negative: 0}).astype(int)
    return None


def _available_models(models_dir: Path) -> list[str]:
    return sorted(p.stem for p in models_dir.glob("*.joblib"))


def _resolve(model_name: str, models_dir: Path) -> Path:
    path = models_dir / f"{model_slug(model_name)}.joblib"
    if not path.exists():
        available = _available_models(models_dir)
        raise SystemExit(
            f"No trained model at {path}.\n"
            f"Available: {', '.join(available) if available else '(none)'}\n"
            "Run `python run_pipeline.py` (or run_generic_pipeline.py) first."
        )
    return path


def _score(pipeline, X: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Positive-class score and the column name to store it under."""
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1], "probability"
    return pipeline.decision_function(X), "decision_score"


def _report(target, y_pred: np.ndarray, y_score: np.ndarray) -> None:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    tn, fp, fn, tp = confusion_matrix(target, y_pred, labels=[0, 1]).ravel()
    print("\nMetrics on the supplied labels:")
    print(f"  ROC-AUC   {roc_auc_score(target, y_score):.4f}")
    print(f"  Accuracy  {accuracy_score(target, y_pred):.4f}")
    print(f"  Precision {precision_score(target, y_pred, zero_division=0):.4f}")
    print(f"  Recall    {recall_score(target, y_pred, zero_division=0):.4f}")
    print(f"  F1-Score  {f1_score(target, y_pred, zero_division=0):.4f}")
    print(f"  TP {tp}  FP {fp}  TN {tn}  FN {fn}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a CSV with a trained model.")
    parser.add_argument("input", type=Path, help="CSV of raw patient rows to score")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"model to use (default: {DEFAULT_MODEL}); "
             f"available: {_available_models(MODELS_DIR)}",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=None,
        help="directory holding the *.joblib models (default: models/)",
    )
    parser.add_argument(
        "--target", default=None,
        help="target column for optional metrics "
             f"(default: '{TARGET}' if present)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="where to write predictions (default: <input>_predictions.csv)",
    )
    args = parser.parse_args(argv)

    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input}")

    models_dir = args.models_dir or MODELS_DIR
    model_path = _resolve(args.model, models_dir)
    pipeline = joblib.load(model_path)

    df = pd.read_csv(args.input)
    required = list(getattr(pipeline, "feature_names_in_", FEATURES))
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{args.input} is missing required columns: {missing}\n"
            f"Expected feature columns: {required}"
        )

    print(f"Model : {model_path.name}")
    print(f"Rows  : {len(df):,}")

    X = df[required]
    y_pred = pipeline.predict(X)
    y_score, score_col = _score(pipeline, X)

    out = df.copy()
    out["prediction"] = y_pred
    out[score_col] = np.round(y_score, 6)

    out_path = args.out or args.input.with_name(f"{args.input.stem}_predictions.csv")
    out.to_csv(out_path, index=False)

    flagged = int(np.asarray(y_pred).sum())
    print(f"Positive predictions: {flagged:,} of {len(df):,} "
          f"({100 * flagged / len(df):.2f}%)")
    print(f"Written: {out_path}")

    target = args.target or (TARGET if TARGET in df.columns else None)
    if target and target in df.columns:
        labels = _as_labels(df[target])
        if labels is not None:
            _report(labels, y_pred, y_score)
        else:
            print(f"\nSkipped metrics: '{target}' is not a clean binary column.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
