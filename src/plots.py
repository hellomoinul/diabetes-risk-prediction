"""Figure helpers. Every function both draws and saves, returning the path."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
)
from .evaluation import Evaluation, pr_points, roc_points

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "figures"

PALETTE = ["#2E86AB", "#F18F01", "#C73E1D", "#3B8C5A", "#6B4E9B"]
GREY = "#9AA5B1"


def _setup() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str) -> Path:
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_numeric_distributions(df: pd.DataFrame):
    """Histogram per numeric feature, split by diagnosis."""
    _setup()
    n = len(NUMERIC_FEATURES)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, NUMERIC_FEATURES):
        for label, color in zip([0, 1], PALETTE[:2]):
            ax.hist(
                df.loc[df[TARGET] == label, col],
                bins=40, alpha=0.55, density=True, color=color,
                label=f"diabetes={label}",
            )
        ax.set_title(col, fontsize=10)
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)

    for ax in axes[n:]:
        ax.axis("off")

    return axes, _save(fig, "01_numeric_distributions.png")


def plot_numeric_boxplots(df: pd.DataFrame):
    """Boxplots per numeric feature, split by diagnosis, to spot outliers.

    Part of the exploration required by the project brief. A boxplot shows
    spread and tail behaviour that a histogram hides, which is how the
    right-skew of ``bmi`` and ``blood_glucose_level`` -- and the long low-age
    tail -- become visible.
    """
    _setup()
    n = len(NUMERIC_FEATURES)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for ax, col in zip(axes.ravel(), NUMERIC_FEATURES):
        sns.boxplot(
            data=df, x=TARGET, y=col, hue=TARGET, order=[0, 1], hue_order=[0, 1],
            palette=PALETTE[:2], width=0.55, fliersize=1.2,
            ax=ax, linewidth=1.0, legend=False,
        )
        ax.set_xlabel("diabetes (0 = no, 1 = yes)", fontsize=9)
        ax.set_title(col, fontsize=10)
        for i, group in enumerate(df[TARGET].value_counts().sort_index().index):
            sub = df.loc[df[TARGET] == group, col]
            ax.text(i, sub.max() * 1.02, f"n={len(sub):,}", ha="center",
                    va="bottom", fontsize=7, color=GREY)

    fig.suptitle(
        "Numeric Feature Distributions by Diagnosis -- spread and outliers",
        fontsize=13, y=1.01,
    )
    return axes, _save(fig, "01b_numeric_boxplots.png")


def plot_class_balance(df: pd.DataFrame):
    """Bar chart of the 91.5 / 8.5 split, annotated with the baseline."""
    _setup()
    counts = df[TARGET].value_counts().sort_index()
    labels = ["Negative\n(no diabetes)", "Positive\n(diabetes)"]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(labels, counts.values, color=PALETTE[:2], width=0.55)
    for bar, count in zip(bars, counts.values):
        pct = 100 * count / counts.sum()
        ax.text(
            bar.get_x() + bar.get_width() / 2, count + counts.max() * 0.02,
            f"{count:,}\n({pct:.2f}%)", ha="center", fontsize=10,
        )
    ax.set_ylabel("Number of patients")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_title(
        "Severe Class Imbalance -- 8.8% positive\n"
        f"(always-negative baseline scores {counts.max() / counts.sum():.4f} accuracy)",
        fontsize=12, pad=12,
    )
    return ax, _save(fig, "02_class_imbalance.png")


def plot_correlation_heatmap(df: pd.DataFrame):
    """Correlation matrix across the encoded frame and the target."""
    _setup()
    encoded = pd.get_dummies(
        df[FEATURES + [TARGET]], columns=CATEGORICAL_FEATURES, drop_first=False
    )
    corr = encoded.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr.to_numpy(), dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
        square=True, linewidths=0.4, ax=ax, cbar_kws={"shrink": 0.7},
    )
    ax.set_title("Feature Correlation with the Target", fontsize=13, pad=12)
    return ax, _save(fig, "03_correlation_heatmap.png")


def plot_category_diabetes_rates(df: pd.DataFrame):
    """Diabetes rate per category -- the plot that shows 'No Info' is signal.

    ``smoking_history == 'No Info'`` has a far lower diabetes rate than any
    genuine smoking category, which is why it is kept as its own level rather
    than being imputed away.
    """
    _setup()
    groups = ["gender", "smoking_history", "hypertension", "heart_disease"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, col in zip(axes.ravel(), groups):
        rate = df.groupby(col)[TARGET].mean() * 100
        rate = rate.sort_values()
        colors = [PALETTE[2] if str(i) == "No Info" else PALETTE[0] for i in rate.index]
        bars = ax.barh([str(i) for i in rate.index], rate.values, color=colors)
        for bar, v in zip(bars, rate.values):
            ax.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}%", va="center", fontsize=9)
        overall = 100 * df[TARGET].mean()
        ax.axvline(overall, color=PALETTE[1], linestyle="--", linewidth=1.2,
                   label=f"overall {overall:.2f}%")
        ax.set_title(f"Diabetes rate by {col}", fontsize=11)
        ax.set_xlabel("Diabetes rate (%)")
        ax.set_xlim(0, max(rate.max() * 1.2, overall * 1.2))
        ax.legend(fontsize=8)

    return axes, _save(fig, "04_category_diabetes_rates.png")


def plot_missingness(df: pd.DataFrame):
    """Missingness: zero NaN cells, but a large 'No Info' sentinel category.

    Sentinels are looked up in ``CATEGORICAL_FEATURES`` rather than by sniffing
    dtypes: under pandas 3 a text column has dtype ``str``, not ``object``, so
    a dtype check silently finds nothing.
    """
    _setup()
    counts: dict[str, int] = {c: int(df[c].isna().sum()) for c in df.columns}
    for col in CATEGORICAL_FEATURES:
        counts[col] = counts.get(col, 0) + int((df[col] == "No Info").sum())

    sentinels = pd.Series(counts)
    sentinels = sentinels[sentinels > 0].sort_values(ascending=False)

    if sentinels.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.text(0.5, 0.5, "No missing values or sentinel categories",
                ha="center", va="center", fontsize=13)
        ax.axis("off")
        return ax, _save(fig, "05_missingness_sentinel.png")

    peak = float(sentinels.max())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(
        [f"{c}\n('No Info' sentinel)" for c in sentinels.index],
        sentinels.values,
        color=[PALETTE[2] if c == "smoking_history" else GREY for c in sentinels.index],
    )
    for bar, v in zip(bars, sentinels.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, v + peak * 0.02,
            f"{v:,}\n({100 * v / len(df):.1f}%)", ha="center", fontsize=9,
        )
    ax.set_ylabel("Rows")
    ax.set_ylim(0, peak * 1.22)
    ax.set_title(
        "Missing Data: isnull() reports 0, but 35.8% of rows are\n"
        "encoded as the string 'No Info' in smoking_history",
        fontsize=12, pad=12,
    )
    return ax, _save(fig, "05_missingness_sentinel.png")


def plot_roc_curves(evaluations: list[Evaluation]):
    """Overlaid ROC curves for all models on one chart."""
    _setup()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    for color, ev in zip(PALETTE, evaluations):
        fpr, tpr, _ = roc_points(ev)
        auc = ev.metrics["ROC-AUC"]
        if np.isnan(auc):
            continue
        ax.plot(fpr, tpr, linewidth=2, color=color,
                label=f"{ev.name} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Chance (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC Curves -- All Models on the Test Set", fontsize=13, pad=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    return ax, _save(fig, "06_roc_curves.png")


def plot_pr_curves(evaluations: list[Evaluation], positive_rate: float):
    """Overlaid Precision-Recall curves -- the informative view at 8.8% positive."""
    _setup()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    for color, ev in zip(PALETTE, evaluations):
        recall, precision, _ = pr_points(ev)
        pr_auc = ev.metrics["PR-AUC"]
        if np.isnan(pr_auc):
            continue
        ax.plot(recall, precision, linewidth=2, color=color,
                label=f"{ev.name} (PR-AUC = {pr_auc:.4f})")

    ax.axhline(positive_rate, color="k", linestyle="--", linewidth=1.2,
               label=f"No-skill baseline ({positive_rate:.4f})")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_title(
        "Precision-Recall Curves -- the right view for an 8.8% positive class",
        fontsize=12.5, pad=12,
    )
    ax.legend(loc="lower left", fontsize=8.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    return ax, _save(fig, "07_precision_recall_curves.png")


def plot_confusion_matrices(evaluations: list[Evaluation]):
    """One confusion-matrix heatmap per model, in a single row."""
    _setup()
    n = len(evaluations)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.9))
    axes = np.atleast_1d(axes)

    for ax, ev in zip(axes, evaluations):
        sns.heatmap(
            ev.confusion, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Pred 0", "Pred 1"], yticklabels=["Actual 0", "Actual 1"],
            ax=ax, linewidths=1, linecolor="white",
        )
        ax.set_title(
            f"{ev.name}\nRec {ev.metrics['Recall']:.3f} | "
            f"Prec {ev.metrics['Precision']:.3f}",
            fontsize=9.5,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.suptitle("Confusion Matrices on the Test Set", fontsize=13, y=1.05)
    return axes, _save(fig, "08_confusion_matrices.png")


def plot_metric_comparison(table: pd.DataFrame):
    """Grouped bars for the metrics that matter under imbalance."""
    _setup()
    metrics = ["ROC-AUC", "PR-AUC", "Recall", "F1-Score"]
    models = list(table.index)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(metrics))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        vals = [float(table.loc[model, m]) for m in metrics]
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                      label=model, color=PALETTE[i % len(PALETTE)])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                    f"{v:.3f}", ha="center", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison on the Test Set", fontsize=13, pad=12)
    ax.legend(fontsize=9, ncol=len(models))
    return ax, _save(fig, "09_metric_comparison.png")


def plot_feature_importance(importance: pd.DataFrame, title: str, filename: str):
    """Horizontal bar chart of feature relevance for one model."""
    _setup()
    data = importance.sort_values("Importance")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars = ax.barh(data["Feature"], data["Importance"], color=PALETTE[0])
    for bar, value in zip(bars, data["Importance"]):
        ax.text(value + data["Importance"].max() * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}", va="center", fontsize=9)
    ax.set_xlabel(title)
    ax.set_xlim(0, data["Importance"].max() * 1.18)
    ax.set_title("Feature Relevance (one-hot columns grouped by parent)",
                 fontsize=12, pad=12)
    return ax, _save(fig, filename)


def plot_cv_results(cv_results: pd.DataFrame, name: str, filename: str):
    """Training vs. validation AUC across the hyperparameter grid."""
    _setup()
    data = cv_results.copy()

    param_cols = [c for c in data.columns if c.startswith("param_")]
    labels = []
    for _, row in data.iterrows():
        labels.append(" | ".join(
            f"{c.removeprefix('clf__').removeprefix('param_')}={row[c]}"
            for c in param_cols
        ))
    data["label"] = labels

    fig, ax = plt.subplots(figsize=(max(9.0, 0.62 * len(data)), 5))
    x = np.arange(len(data))
    width = 0.4

    ax.bar(x - width / 2, data["mean_train_score"], width,
           label="Train AUC", color=PALETTE[0], alpha=0.75)
    ax.bar(x + width / 2, data["mean_test_score"], width,
           label="Validation AUC", color=PALETTE[1], alpha=0.9)

    best = int(data["mean_test_score"].idxmax())
    best_pos = list(data.index).index(best)
    top = min(1.02, data["mean_train_score"].max() + 0.1)
    ax.set_ylim(min(0.4, data["mean_test_score"].min() - 0.05), top)
    ax.annotate(
        "selected",
        xy=(best_pos + width / 2, data.loc[best, "mean_test_score"]),
        xytext=(best_pos + width / 2,
                min(top - 0.02, data["mean_test_score"].max() + 0.04)),
        ha="center", fontsize=9, color=PALETTE[2], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=PALETTE[2]),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(data["label"], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("ROC-AUC (5-fold mean)")
    ax.set_title(f"{name}: Over-fitting Check Across Hyperparameter Grid",
                 fontsize=13, pad=12)
    ax.legend(fontsize=9)
    return ax, _save(fig, filename)
