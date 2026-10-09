"""Handle a brand-new dataset with one command: CSV *or* Kaggle link.

This is the exam-time entry point. It accepts a dataset either as a local CSV or
as a Kaggle URL / ``owner/slug`` and then decides, automatically, which of the
project's two workflows to run:

* **same features as the diabetes model** -> score the rows with the saved model
  (``predict.py``), no retraining;
* **different features** -> retrain the same four-model workflow on the new
  dataset (``run_generic_pipeline.py``).

    python run_on_new_data.py --csv /content/exam.csv
    python run_on_new_data.py --kaggle https://www.kaggle.com/datasets/owner/slug
    python run_on_new_data.py --csv exam.csv --target churn --mode auto
    python run_on_new_data.py --csv exam.csv --mode retrain --outdir results_exam

``--mode`` overrides the automatic decision (``auto`` | ``predict`` | ``retrain``).
Everything is written under ``--outdir`` (default ``results_new/``) together with
a ``run_on_new_data.json`` that records the path taken and why.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.data import FEATURES, MODELS_DIR
from src.kaggle import download_dataset, parse_slug

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTDIR = PROJECT_ROOT / "results_new"


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def diabetes_features() -> list[str]:
    """Feature columns the saved diabetes model expects, from the model itself."""
    models = sorted(MODELS_DIR.glob("*.joblib"))
    if not models:
        return list(FEATURES)
    try:
        pipeline = joblib.load(models[0])
        return list(getattr(pipeline, "feature_names_in_", FEATURES))
    except Exception:
        return list(FEATURES)


def resolve_target(columns: list[str], required: list[str], given: str | None):
    """Pick the target column, or None when the CSV carries only features."""
    if given:
        return given
    last = columns[-1]
    return None if last in required else last


def decide_mode(
    columns: list[str], required: list[str], target: str | None
) -> tuple[str, str]:
    """Return ("predict"|"retrain", reason) by comparing columns to the model."""
    cols = set(columns)
    missing = [c for c in required if c not in cols]
    extra = [c for c in columns if c not in required and c != target]

    if missing:
        return "retrain", f"missing trained-model features: {missing}"
    if extra:
        return "retrain", f"extra feature columns not seen in training: {extra}"
    return "predict", f"columns match the trained model ({len(required)} features)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the right workflow on a new CSV or Kaggle dataset."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="local CSV dataset")
    source.add_argument("--kaggle", help="Kaggle dataset URL or 'owner/slug'")
    parser.add_argument("--target", default=None,
                        help="target column (default: last column, if it is not a feature)")
    parser.add_argument("--positive-label", default=None,
                        help="positive class value for the retrain path (default: minority)")
    parser.add_argument("--mode", choices=["auto", "predict", "retrain"], default="auto",
                        help="force a workflow instead of auto-detecting")
    parser.add_argument("--model", default=None,
                        help="model for the predict path (default: xgboost)")
    parser.add_argument("--models-dir", type=Path, default=None,
                        help="where the predict path reads *.joblib from (default: models/)")
    parser.add_argument("--outdir", type=Path, default=None,
                        help="output directory (default: results_new/)")
    args = parser.parse_args(argv)

    outdir = args.outdir or DEFAULT_OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ get the data
    section("STEP 0  Obtain the dataset")
    source_kind = "csv"
    kaggle_url = None
    if args.kaggle:
        source_kind = "kaggle"
        kaggle_url = args.kaggle
        slug = parse_slug(args.kaggle)
        csv_path = download_dataset(args.kaggle, outdir / "data" / slug.split("/")[-1])
    else:
        csv_path = args.csv
        if not csv_path.is_file():
            raise SystemExit(f"Input file not found: {csv_path}")

    columns = list(pd.read_csv(csv_path, nrows=0).columns)
    print(f"CSV   : {csv_path}")
    print(f"Rows  : {sum(1 for _ in open(csv_path, encoding='utf-8', errors='ignore')) - 1:,}")
    print(f"Columns ({len(columns)}): {columns}")

    # --------------------------------------------------------- choose workflow
    section("STEP 1  Choose the workflow")
    required = diabetes_features()
    target = resolve_target(columns, required, args.target)
    if target and target not in columns:
        raise SystemExit(f"--target {target!r} not in columns {columns}")

    if args.mode == "auto":
        mode, reason = decide_mode(columns, required, target)
    else:
        mode = args.mode
        reason = "forced by --mode"

    models_dir = args.models_dir or MODELS_DIR
    if mode == "predict" and not sorted(models_dir.glob("*.joblib")):
        reason = f"no trained model in {models_dir}; falling back to retrain"
        mode = "retrain"

    print(f"Trained-model features : {required}")
    print(f"Target column          : {target}")
    print(f"Decision               : {mode.upper()}  ({reason})")

    summary = {
        "source": source_kind,
        "kaggle": kaggle_url,
        "csv": str(csv_path),
        "columns": columns,
        "target": target,
        "mode": mode,
        "reason": reason,
        "outdir": str(outdir),
    }

    # ------------------------------------------------------------------ run it
    if mode == "predict":
        import predict as predict_mod

        section("STEP 2  Score with the saved model (predict.py)")
        argv_run = [str(csv_path), "--out", str(outdir / "predictions.csv"),
                    "--models-dir", str(models_dir)]
        if args.model:
            argv_run += ["--model", args.model]
        if target:
            argv_run += ["--target", target]
        code = predict_mod.main(argv_run)
        summary["outputs"] = {"predictions": str(outdir / "predictions.csv")}
    else:
        import run_generic_pipeline as generic_mod

        section("STEP 2  Retrain the 4-model workflow (run_generic_pipeline.py)")
        argv_run = [str(csv_path), "--outdir", str(outdir)]
        if target:
            argv_run += ["--target", target]
        if args.positive_label:
            argv_run += ["--positive-label", args.positive_label]
        code = generic_mod.main(argv_run)
        summary["outputs"] = {
            "result_table": str(outdir / "final_result_table.csv"),
            "figures": str(outdir / "figures"),
            "models": str(outdir / "models"),
        }

    with open(outdir / "run_on_new_data.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    section("DONE")
    print(f"Workflow : {mode}")
    print(f"Outputs  : {outdir}")
    return code


if __name__ == "__main__":
    sys.exit(main())
