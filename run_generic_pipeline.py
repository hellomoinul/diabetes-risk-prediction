"""Run the same 4-model workflow on ANY binary-classification CSV.

This is the dataset-agnostic sibling of ``run_pipeline.py``. Instead of the
hardcoded diabetes columns it *infers* the schema, so it works on the exam
dataset (different features, same binary-classification task)::

    python run_generic_pipeline.py path/to/data.csv
    python run_generic_pipeline.py data.csv --target outcome --positive-label 1
    python run_generic_pipeline.py data.csv --outdir results_exam

What it does, matching the project's methodology:

* profiles the data (balance, missingness, duplicates, summary stats),
* infers numeric vs. categorical features from the dtypes,
* encodes the target (minority class = positive, unless ``--positive-label``),
* stratified 70/15/15 split,
* tunes the same four models (Logistic Regression, Decision Tree, SVM, XGBoost)
  with the same 5-fold ROC-AUC grids and imbalance handling,
* evaluates on the held-out test set and writes the required result table,
* draws ROC / PR / confusion / metric / importance figures,
* saves the tuned pipelines to ``<outdir>/models/``.

Every output goes under ``--outdir`` (default ``results_<csv-stem>``) so it never
touches the diabetes results in ``results/`` and ``figures/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src import plots
from src.evaluation import (
    baseline_evaluation,
    baseline_reference_line,
    evaluate_pipeline,
    final_result_table,
    metrics_table,
)
from src.models import (
    _cv,
    get_specs,
    imbalance_ratio,
    model_slug,
    stratified_subsample,
)

PROJECT_ROOT = Path(__file__).resolve().parent
RANDOM_STATE = 42
SVM_SEARCH_ROWS = 20_000
SCORING = "roc_auc"


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------- schema
def infer_schema(df: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    features = [c for c in df.columns if c != target]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in features if c not in numeric]
    return numeric, categorical


def match_label(values: list, positive_label: str):
    for v in values:
        if str(v) == str(positive_label):
            return v
    try:
        numeric = float(positive_label)
        for v in values:
            if isinstance(v, (int, float, np.integer, np.floating)) and float(v) == numeric:
                return v
    except (TypeError, ValueError):
        pass
    return None


def encode_target(y: pd.Series, positive_label: str | None) -> tuple[pd.Series, object]:
    values = list(pd.unique(y))
    if len(values) != 2:
        raise SystemExit(
            f"Target must be binary (2 classes); found {len(values)}: {values}"
        )

    if positive_label is None:
        positive = y.value_counts().idxmin()
    else:
        positive = match_label(values, positive_label)
        if positive is None:
            raise SystemExit(
                f"--positive-label {positive_label!r} is not one of the target "
                f"values {values}"
            )

    negative = next(v for v in values if v != positive)
    encoded = y.map({positive: 1, negative: 0}).astype(int)
    return encoded, positive


def build_preprocessor(numeric: list[str], categorical: list[str]):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    transformers = []
    if numeric:
        transformers.append((
            "num",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
        ))
    if not transformers:
        raise SystemExit("No usable feature columns found.")
    return ColumnTransformer(transformers, verbose_feature_names_out=True)


def split_data(X, y, test_size=0.30, val_size=0.15):
    X_train, X_pool, y_train, y_pool = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_pool, y_pool, test_size=val_size / test_size, stratify=y_pool,
        random_state=RANDOM_STATE,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ------------------------------------------------------------------ profiling
def class_balance(y: pd.Series) -> pd.DataFrame:
    counts = y.value_counts().sort_index()
    out = pd.DataFrame({"Class": counts.index.astype(str), "Count": counts.values})
    out["Percent"] = (100 * out["Count"] / len(y)).round(2)
    out["Ratio"] = (counts.max() / counts.min())
    return out


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"Feature": c, "Missing": int(df[c].isna().sum()),
         "Percent": round(100 * df[c].isna().mean(), 2)}
        for c in df.columns
    ]
    return pd.DataFrame(rows).sort_values("Missing", ascending=False)


def duplicate_report(df: pd.DataFrame, target: str) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c != target]
    dup_mask = df.duplicated(keep=False)
    conflicting = 0
    if dup_mask.any():
        groups = df[dup_mask].groupby(feature_cols, dropna=False)[target].nunique()
        conflicting = int((groups > 1).sum())
    return pd.DataFrame({
        "Check": [
            "Full-row duplicates",
            "Duplicates ignoring target",
            "Rows in a duplicate group",
            "Groups with conflicting labels",
        ],
        "Count": [
            int(df.duplicated().sum()),
            int(df.duplicated(subset=feature_cols).sum()),
            int(dup_mask.sum()),
            conflicting,
        ],
    })


# -------------------------------------------------------------------- helpers
def generic_importance(
    pipeline, numeric: list[str], categorical: list[str],
    X: pd.DataFrame, y: pd.Series,
) -> pd.DataFrame:
    clf = pipeline.named_steps["clf"]

    if hasattr(clf, "feature_importances_") or hasattr(clf, "coef_"):
        names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
        if hasattr(clf, "feature_importances_"):
            values = np.asarray(clf.feature_importances_, dtype=float)
            measure = "Gain-based importance"
        else:
            values = np.abs(np.asarray(clf.coef_, dtype=float).ravel())
            values = values / (values.sum() or 1.0)
            measure = "|Standardised coefficient|"

        totals: dict[str, float] = {}
        for name, value in zip(names, values):
            block, _, rest = name.partition("__")
            if block == "cat":
                matches = [c for c in categorical if rest == c or rest.startswith(f"{c}_")]
                parent = max(matches, key=len) if matches else rest
            else:
                parent = rest
            totals[parent] = totals.get(parent, 0.0) + abs(float(value))

        total = sum(totals.values()) or 1.0
        out = pd.DataFrame(
            {"Feature": list(totals), "Importance": [v / total for v in totals.values()]}
        )
        out = out.sort_values("Importance", ascending=False).reset_index(drop=True)
        out["Measure"] = measure
        return out

    # SVM (RBF) exposes neither: fall back to permutation importance.
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        pipeline, X, y, scoring=SCORING, n_repeats=5,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    out = pd.DataFrame(
        {"Feature": list(X.columns), "Importance": np.clip(result.importances_mean, 0, None)}
    )
    total = out["Importance"].sum() or 1.0
    out["Importance"] = out["Importance"] / total
    out = out.sort_values("Importance", ascending=False).reset_index(drop=True)
    out["Measure"] = "Permutation importance (ROC-AUC drop)"
    return out


def overfit_row(name, clf, preprocessor, X, y, cv) -> dict:
    pipe = Pipeline([("preprocess", clone(preprocessor)), ("clf", clone(clf))])
    scores = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, return_train_score=True)
    return {
        "Model": name,
        "Train AUC": scores["train_score"].mean(),
        "Validation AUC": scores["test_score"].mean(),
        "Gap (over-fit)": scores["train_score"].mean() - scores["test_score"].mean(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the 4-model workflow on any CSV.")
    parser.add_argument("input", type=Path, help="CSV dataset (binary classification)")
    parser.add_argument("--target", default=None,
                        help="target column name (default: the last column)")
    parser.add_argument("--positive-label", default=None,
                        help="which target value is the positive class (default: minority)")
    parser.add_argument("--outdir", type=Path, default=None,
                        help="output directory (default: results_<csv-stem>)")
    args = parser.parse_args(argv)

    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input}")

    started = time.time()
    df = pd.read_csv(args.input)
    target = args.target or df.columns[-1]
    if target not in df.columns:
        raise SystemExit(f"--target {target!r} not in columns {list(df.columns)}")

    outdir = args.outdir or PROJECT_ROOT / f"results_{args.input.stem}"
    fig_dir = outdir / "figures"
    models_dir = outdir / "models"
    fig_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    plots.FIG_DIR = fig_dir

    summary: dict = {"input": str(args.input), "target": target,
                     "outdir": str(outdir), "random_state": RANDOM_STATE}

    # ---------------------------------------------------------------- profile
    section("STEP 1  Load and profile")
    raw = df.dropna(subset=[target]).reset_index(drop=True)
    dropped = len(df) - len(raw)
    numeric, categorical = infer_schema(raw, target)

    y_all, positive = encode_target(raw[target], args.positive_label)
    raw = raw.assign(**{target: y_all})

    print(f"Rows (target present): {len(raw):,}  (dropped {dropped} without target)")
    print(f"Target column        : {target}  (positive = {positive!r})")
    print(f"Numeric features     : {numeric}")
    print(f"Categorical features : {categorical}")

    balance = class_balance(y_all)
    missing = missing_report(df)
    dupes = duplicate_report(raw, target)
    stats = raw[numeric].describe().T if numeric else pd.DataFrame()

    print()
    print(balance.to_string(index=False))
    print(f"\nFull-row duplicates: {int(raw.duplicated().sum()):,}")

    # ---------------------------------------------------------------- splitting
    section("STEP 2  Stratified 70 / 15 / 15 split")
    X = raw[numeric + categorical]
    y = raw[target]
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    for name, Xs, ys in [
        ("train", X_train, y_train), ("val  ", X_val, y_val), ("test ", X_test, y_test),
    ]:
        print(f"{name}  {len(Xs):7,} rows   positive {int(ys.sum()):5,}  "
              f"({100 * ys.mean():.2f}%)")

    baseline = baseline_evaluation("Majority Baseline", y_test)
    summary["majority_baseline_accuracy"] = round(baseline.metrics["Accuracy"], 4)
    summary["rows_used"] = len(raw)
    summary["positive_label"] = str(positive)
    summary["duplicates_removed"] = 0

    print(f"\nMajority-class baseline accuracy: {baseline.metrics['Accuracy']:.4f}")

    # ---------------------------------------------------------------- overfit
    section("STEP 3  Over-fitting control on unrestricted models")
    cv = _cv()
    overfit = pd.DataFrame([
        overfit_row(
            "Decision Tree (unrestricted)",
            DecisionTreeClassifier(max_depth=None, min_samples_leaf=1,
                                   class_weight="balanced", random_state=RANDOM_STATE),
            build_preprocessor(numeric, categorical), X_train, y_train, cv),
        overfit_row(
            "XGBoost (deep, 2000 trees)",
            XGBClassifier(n_estimators=2000, max_depth=12, learning_rate=0.1,
                          eval_metric="logloss", verbosity=0, random_state=RANDOM_STATE),
            build_preprocessor(numeric, categorical), X_train, y_train, cv),
    ])
    print(overfit.round(4).to_string(index=False))

    # ---------------------------------------------------------------- tuning
    section("STEP 4  Hyperparameter tuning (5-fold CV, ROC-AUC)")
    ratio = imbalance_ratio(y_train)
    print(f"Negative-to-positive ratio in train: {ratio:.2f}\n")
    specs = get_specs(scale_pos_weight=ratio)
    fitted = {}
    for name, spec in specs.items():
        t0 = time.time()
        search_X, search_y = X_train, y_train
        if name == "SVM" and len(X_train) > SVM_SEARCH_ROWS:
            search_X, search_y = stratified_subsample(X_train, y_train, SVM_SEARCH_ROWS)
            print(f"Tuning {name} on a stratified {len(search_X):,}-row subsample ...",
                  flush=True)
        else:
            print(f"Tuning {name} ...", flush=True)

        pipe = Pipeline([("preprocess", build_preprocessor(numeric, categorical)),
                         ("clf", clone(spec.model))])
        search = GridSearchCV(pipe, param_grid=spec.grid, scoring=SCORING, cv=cv,
                              refit=True, n_jobs=-1, return_train_score=True)
        search.fit(search_X, search_y)
        fitted[name] = search
        params = {k.removeprefix("clf__"): v for k, v in search.best_params_.items()}
        print(f"  best CV ROC-AUC {search.best_score_:.4f}  params={params}  "
              f"({time.time() - t0:.1f}s)")
        plots.plot_cv_results(
            pd.DataFrame(search.cv_results_), name,
            f"cv_{model_slug(name)}.png",
        )
        summary.setdefault("best_params", {})[name] = params
        summary.setdefault("best_cv_auc", {})[name] = round(float(search.best_score_), 4)

    # ---------------------------------------------------------------- evaluate
    section("STEP 5  Final evaluation on the held-out test set")
    evals = [baseline]
    for name, search in fitted.items():
        evals.append(evaluate_pipeline(name, search.best_estimator_, X_test, y_test))

    table = metrics_table(evals)
    print(table.to_string())
    result_table = final_result_table(evals)
    print("\n--- Final result table (project brief format) ---")
    print(result_table.to_string())
    print()
    print(baseline_reference_line(evals))

    best_model = max((e for e in evals if e.name != "Majority Baseline"),
                     key=lambda e: e.metrics["ROC-AUC"])
    summary["best_model"] = best_model.name
    print(f"\nBest model by ROC-AUC: {best_model.name} "
          f"({best_model.metrics['ROC-AUC']:.4f})")

    section("STEP 6  Figures")
    plots.plot_roc_curves(evals)
    plots.plot_pr_curves(evals, float(y_test.mean()))
    plots.plot_confusion_matrices(evals)
    plots.plot_metric_comparison(table)
    for name, search in fitted.items():
        imp = generic_importance(search.best_estimator_, numeric, categorical,
                                 X_test, y_test)
        plots.plot_feature_importance(
            imp, imp["Measure"].iloc[0],
            f"importance_{model_slug(name)}.png",
        )
    print(f"Wrote figures to {fig_dir}")

    # ---------------------------------------------------------------- save
    section("STEP 7  Save tables and models")
    for filename, frame in {
        "class_balance.csv": balance,
        "missing_values.csv": missing,
        "duplicate_report.csv": dupes,
        "summary_statistics.csv": stats,
        "overfitting_check.csv": overfit,
        "model_comparison.csv": table,
        "final_result_table.csv": result_table,
    }.items():
        frame.to_csv(outdir / filename)
        print(f"  {outdir.name}/{filename}")

    for name, search in fitted.items():
        joblib.dump(search.best_estimator_, models_dir / f"{model_slug(name)}.joblib")
    with open(outdir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  {outdir.name}/summary.json")
    print(f"  {outdir.name}/models/*.joblib")

    section("DONE")
    print(f"Total runtime: {time.time() - started:.1f}s")
    print(f"Score another CSV with: "
          f"python predict.py <csv> --model {model_slug(best_model.name)} "
          f"--models-dir {outdir.name}/models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
