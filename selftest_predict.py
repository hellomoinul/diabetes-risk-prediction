"""Self-test for predict.py: proves the saved models load and score new rows.

This is a FUNCTIONAL check, not a performance claim. It scores a slice of the
training CSV (downloaded from Kaggle if missing), prints ROC-AUC, and asserts the saved
artifact is usable and sensible (AUC > 0.95, recall > 0.85, all 4 models load).

It does NOT reproduce the 0.9799 test-set figure from the report. That number
comes from the held-out test split inside run_pipeline.py. This script scores
rows the model may have seen during training (in-sample), so a high number here
is expected and proves nothing about generalization. The point is only: the
.joblib loads, --map renaming gives identical predictions, and the outputs are
sane.

Run:  python selftest_predict.py          # ~30 seconds

Exits 0 on pass, 1 on any failure. Uses only temporary files; the repo is
left untouched.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import predict as predict_mod  # noqa: E402
from src.data import MODELS_DIR, TARGET  # noqa: E402
from src.fetch_data import fetch  # noqa: E402

EXPECTED_MODELS = ["logistic_regression", "decision_tree", "svm", "xgboost"]
N_ROWS = 5_000
MIN_AUC = 0.95
MIN_RECALL = 0.85


def check_all_models_load() -> None:
    print("1. Loading all saved models...")
    for slug in EXPECTED_MODELS:
        path = MODELS_DIR / f"{slug}.joblib"
        if not path.is_file():
            raise SystemExit(f"FAIL: missing model artifact {path} (run run_pipeline.py first)")
        pipeline = joblib.load(path)
        print(f"   {slug}: loaded ({type(pipeline).__name__})")
    print("   OK")


def score_slice(tmp: Path, df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n2. Scoring a {N_ROWS:,}-row slice with xgboost...")
    slice_path = tmp / "slice.csv"
    df.to_csv(slice_path, index=False)
    out_path = tmp / "slice_predictions.csv"
    code = predict_mod.main([
        str(slice_path), "--model", "xgboost", "--out", str(out_path),
    ])
    if code != 0:
        raise SystemExit("FAIL: predict.py returned non-zero")
    scored = pd.read_csv(out_path)
    if "prediction" not in scored.columns or "probability" not in scored.columns:
        raise SystemExit(f"FAIL: missing output columns: {list(scored.columns)}")

    labels = scored[TARGET].astype(int).to_numpy()
    y_pred = scored["prediction"].to_numpy()
    y_score = scored["probability"].to_numpy()
    auc = roc_auc_score(labels, y_score)
    recall = recall_score(labels, y_pred, zero_division=0)
    print(f"   ROC-AUC {auc:.4f}  Recall {recall:.4f}")
    if auc < MIN_AUC or recall < MIN_RECALL:
        raise SystemExit(
            f"FAIL: metrics below sanity floor (AUC {auc:.4f} < {MIN_AUC}, "
            f"recall {recall:.4f} < {MIN_RECALL})"
        )
    print("   OK")
    return scored


def check_map_equivalence(tmp: Path, reference: pd.DataFrame, df: pd.DataFrame) -> None:
    print("\n3. Verifying --map gives identical predictions...")
    renamed = df.rename(columns={"blood_glucose_level": "glucose"})
    renamed_path = tmp / "renamed.csv"
    renamed.to_csv(renamed_path, index=False)
    out_path = tmp / "renamed_predictions.csv"
    code = predict_mod.main([
        str(renamed_path), "--model", "xgboost",
        "--map", "glucose:blood_glucose_level",
        "--out", str(out_path),
    ])
    if code != 0:
        raise SystemExit("FAIL: predict.py --map returned non-zero")
    mapped = pd.read_csv(out_path)
    same_pred = np.array_equal(mapped["prediction"].to_numpy(), reference["prediction"].to_numpy())
    same_prob = np.allclose(mapped["probability"].to_numpy(), reference["probability"].to_numpy())
    if not (same_pred and same_prob):
        raise SystemExit("FAIL: --map predictions differ from direct predictions")
    print("   identical predictions and probabilities. OK")


def main() -> int:
    print("=" * 78)
    print("SELF-TEST: predict.py on a labeled slice (functional check only)")
    print("=" * 78)
    print("Loading training data (downloads from Kaggle if missing)...")
    df_full = fetch().head(N_ROWS)
    check_all_models_load()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        reference = score_slice(tmp, df_full)
        check_map_equivalence(tmp, reference, df_full)
    print("\nSELF-TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
