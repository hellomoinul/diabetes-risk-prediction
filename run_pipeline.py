"""End-to-end pipeline: fetch -> clean -> split -> tune -> evaluate -> save.

Run with::

    python run_pipeline.py

Writes every figure to ``figures/`` and every table to ``results/``. The
notebook walks through the same steps interactively; this script is the
one-command reproduction of the whole thing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src import plots
from src.data import (
    MODELS_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    TARGET,
    FEATURES,
    build_pipeline,
    category_breakdown,
    class_balance,
    data_quality_report,
    drop_duplicates,
    duplicate_report,
    load_data,
    majority_baseline,
    make_splits,
    missing_value_report,
    summary_statistics,
)
from src.evaluation import (
    baseline_evaluation,
    baseline_reference_line,
    evaluate_pipeline,
    final_result_table,
    metrics_table,
)
from src.fetch_data import KAGGLE_URL, fetch
from src.models import (
    _cv,
    feature_importance,
    get_specs,
    imbalance_ratio,
    model_slug,
    overfitting_check,
    stratified_subsample,
    tune_model,
)

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_REG_NAME = "Logistic Regression"
TREE_NAME = "Decision Tree"
SVM_SEARCH_ROWS = 20_000


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> dict:
    started = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    # ---------------------------------------------------------------- step 1
    section("STEP 1  Fetch and validate the dataset (official Kaggle API)")
    raw = fetch()
    print(f"Source: {KAGGLE_URL}")
    print(f"Shape : {raw.shape}")

    # ---------------------------------------------------------------- step 2
    section("STEP 2  Profile")
    balance = class_balance(raw)
    missing = missing_value_report(raw)
    dupes = duplicate_report(raw)
    stats = summary_statistics(raw)
    cats = category_breakdown(raw)
    quality = data_quality_report(raw)

    print(balance.to_string(index=False))
    print()
    print(missing.to_string(index=False))
    print()
    print(dupes.to_string(index=False))

    # ---------------------------------------------------------------- step 3
    section("STEP 3  Clean: drop full-row duplicates")
    df = drop_duplicates(raw)
    removed = len(raw) - len(df)
    print(f"Removed {removed:,} duplicate rows -> {len(df):,} rows "
          f"({100 * removed / len(raw):.2f}% of the raw data)")
    print(f"Positive rate after cleaning: {100 * df[TARGET].mean():.2f}%")

    # ---------------------------------------------------------------- step 4
    section("STEP 4  Stratified 70 / 15 / 15 split")
    splits = make_splits(df)
    for name, X, y in [
        ("train", splits.X_train, splits.y_train),
        ("val  ", splits.X_val, splits.y_val),
        ("test ", splits.X_test, splits.y_test),
    ]:
        print(f"{name}  {len(X):7,} rows   positive {int(y.sum()):5,}  "
              f"({100 * y.mean():.2f}%)")

    baseline = majority_baseline(splits.y_test)
    print(f"\nMajority-class baseline (always predict 'no diabetes'): {baseline:.4f} accuracy")
    summary["majority_baseline_accuracy"] = round(baseline, 4)
    summary["rows_after_cleaning"] = len(df)
    summary["duplicates_removed"] = removed

    # ---------------------------------------------------------------- step 5
    section("STEP 5  Exploratory analysis figures")
    plots.plot_numeric_distributions(df)
    plots.plot_numeric_boxplots(df)
    plots.plot_class_balance(df)
    plots.plot_correlation_heatmap(df)
    plots.plot_category_diabetes_rates(df)
    plots.plot_missingness(df)
    print("Wrote figures 01, 01b, 02-05")

    # ---------------------------------------------------------------- step 6
    section("STEP 6  Over-fitting control on an unrestricted model")
    ratio = imbalance_ratio(splits.y_train)
    print(f"Negative-to-positive ratio in train: {ratio:.2f}")
    print("This is passed to XGBoost as scale_pos_weight.\n")

    cv = _cv()
    overfit = []
    for label, model in [
        ("Decision Tree (unrestricted)",
         DecisionTreeClassifier(max_depth=None, min_samples_leaf=1,
                                class_weight="balanced", random_state=RANDOM_STATE)),
        ("XGBoost (deep, 2000 trees)",
         XGBClassifier(n_estimators=2000, max_depth=12, learning_rate=0.1,
                       eval_metric="logloss", verbosity=0,
                       random_state=RANDOM_STATE)),
    ]:
        overfit.append(
            overfitting_check(label, model, splits.X_train, splits.y_train, cv=cv)
        )
    overfit = pd.concat(overfit, ignore_index=True)
    print(overfit.round(4).to_string(index=False))

    # ---------------------------------------------------------------- step 7
    section("STEP 7  Hyperparameter tuning (5-fold CV, ROC-AUC)")
    specs = get_specs(scale_pos_weight=ratio)
    fitted = {}
    for name, spec in specs.items():
        t0 = time.time()
        search_X, search_y = splits.X_train, splits.y_train

        if name == "SVM":
            search_X, search_y = stratified_subsample(
                splits.X_train, splits.y_train, SVM_SEARCH_ROWS
            )
            print(f"\nTuning and refitting {name} on a stratified "
                  f"{len(search_X):,}-row subsample (RBF cost is ~quadratic), "
                  f"instead of all {len(splits.X_train):,} rows ...", flush=True)
        else:
            print(f"\nTuning {name} ...", flush=True)

        fitted[name] = tune_model(spec, search_X, search_y, cv=cv)
        params = {k.removeprefix("clf__"): v
                  for k, v in fitted[name].best_params.items()}
        print(f"  best CV ROC-AUC {fitted[name].best_cv_score:.4f}  "
              f"params={params}  ({time.time() - t0:.1f}s)")
        plots.plot_cv_results(
            fitted[name].cv_results, name, f"cv_{name.split()[0].lower()}.png"
        )
        summary.setdefault("best_params", {})[name] = params
        summary.setdefault("best_cv_auc", {})[name] = round(fitted[name].best_cv_score, 4)

    # ---------------------------------------------------------------- step 8
    section("STEP 8  Final evaluation on the held-out test set")
    evals = [baseline_evaluation("Majority Baseline", splits.y_test)]
    for name, model in fitted.items():
        evals.append(evaluate_pipeline(name, model.best_estimator,
                                       splits.X_test, splits.y_test))
    table = metrics_table(evals)
    print(table.to_string())
    table.to_csv(RESULTS_DIR / "model_comparison.csv")

    # The table in the exact form the project brief specifies:
    # 1. Model list | 2. AUC | 3. Accuracy | 4. Precision | 5. Recall | 6. F1-Score
    result_table = final_result_table(evals)
    print("\n--- Final result table (project brief format) ---")
    print(result_table.to_string())
    result_table.to_csv(RESULTS_DIR / "final_result_table.csv")
    print()
    print(baseline_reference_line(evals))

    best_model = max(
        (e for e in evals if e.name != "Majority Baseline"),
        key=lambda e: e.metrics["ROC-AUC"],
    )
    print(f"\nBest model by ROC-AUC: {best_model.name} "
          f"({best_model.metrics['ROC-AUC']:.4f})")
    summary["best_model"] = best_model.name

    # ---------------------------------------------------------------- step 9
    section("STEP 9  Evaluation figures")
    plots.plot_roc_curves(evals)
    plots.plot_pr_curves(evals, float(splits.y_test.mean()))
    plots.plot_confusion_matrices(evals)
    plots.plot_metric_comparison(table)
    print("Wrote figures 06-09")

    # ---------------------------------------------------------------- step 10
    section("STEP 10  Feature importance")
    for name, filename in [
        (LOG_REG_NAME, "10_feature_importance_logistic_regression.png"),
        (TREE_NAME, "11_feature_importance_decision_tree.png"),
    ]:
        importance = feature_importance(fitted[name])
        measure = importance["Measure"].iloc[0]
        plots.plot_feature_importance(importance, measure, filename)
        print(f"\n{name} ({measure}):")
        print(importance[["Feature", "Importance"]].round(4).to_string(index=False))
        importance.drop(columns="Measure").to_csv(
            RESULTS_DIR / f"feature_importance_{name.split()[0].lower()}.csv",
            index=False,
        )

    # ---------------------------------------------------------------- step 11
    section("STEP 11  Save tables")
    saved = {
        "class_balance.csv": balance,
        "missing_values.csv": missing,
        "duplicate_report.csv": dupes,
        "summary_statistics.csv": stats,
        "category_diabetes_rates.csv": cats,
        "data_quality_flags.csv": quality,
        "overfitting_check.csv": overfit,
        "model_comparison.csv": table,
        "final_result_table.csv": result_table,
    }
    for filename, frame in saved.items():
        frame.to_csv(RESULTS_DIR / filename)
        print(f"  results/{filename}")

    with open(RESULTS_DIR / "pipeline_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("  results/pipeline_summary.json")

    # ---------------------------------------------------------------- step 12
    section("STEP 12  Save fitted models for reuse (predict on new data)")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_index: dict = {}
    for name, model in fitted.items():
        slug = model_slug(name)
        joblib.dump(model.best_estimator, MODELS_DIR / f"{slug}.joblib")
        model_index[name] = {
            "slug": slug,
            "file": f"models/{slug}.joblib",
            "best_cv_auc": round(model.best_cv_score, 4),
            "features": FEATURES,
            "target": TARGET,
        }
        print(f"  models/{slug}.joblib")
    with open(MODELS_DIR / "models.json", "w", encoding="utf-8") as fh:
        json.dump(model_index, fh, indent=2)
    print("  models/models.json")
    print("\nScore a new CSV with the trained model, e.g.:")
    print("  python predict.py newdata.csv --model xgboost")

    section("DONE")
    print(f"Total runtime: {time.time() - started:.1f}s")
    return summary


if __name__ == "__main__":
    main()
