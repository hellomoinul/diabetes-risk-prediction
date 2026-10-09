"""Model evaluation: baseline, confusion matrices, metrics, ROC and PR curves.

With a positive rate of 8.8%, accuracy is close to meaningless on its own -- a
model that always answers "no diabetes" already scores 91.2%. Every table here
therefore carries the majority-class baseline alongside the models, and
Precision-Recall curves sit next to the ROC curves, because PR curves are the
informative view when the positive class is rare.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

METRIC_ORDER = [
    "ROC-AUC", "PR-AUC", "Accuracy", "Precision", "Recall", "F1-Score",
]

# Row labels and column order for the final result table, exactly as specified
# by the project brief: "1. Model list / 2. AUC / 3. Accuracy / 4. Precision /
# 5. Recall / 6. F1-Score" with rows "Log St / DT / SVM / XGBoost".
RESULT_LABELS = {
    "Logistic Regression": "Log St",
    "Decision Tree": "DT",
    "SVM": "SVM",
    "XGBoost": "XGBoost",
}
RESULT_ROW_ORDER = ["Log St", "DT", "SVM", "XGBoost"]
RESULT_COLUMN_ORDER = [
    "2. AUC", "3. Accuracy", "4. Precision", "5. Recall", "6. F1-Score",
]


@dataclass
class Evaluation:
    """All test-set results for one model."""

    name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    y_score: np.ndarray
    metrics: dict[str, float]
    confusion: np.ndarray

    def __post_init__(self) -> None:
        """Accept either a probability or a raw decision score.

        ROC-AUC and PR-AUC only depend on how scores are *ranked*, so an SVM
        decision function is a valid substitute for a probability. Precision,
        Recall and F1 come from ``y_pred``, the hard 0/1 decision, and never
        touch ``y_score``.
        """

    @property
    def tn(self) -> int:
        return int(self.confusion[0, 0])

    @property
    def fp(self) -> int:
        return int(self.confusion[0, 1])

    @property
    def fn(self) -> int:
        return int(self.confusion[1, 0])

    @property
    def tp(self) -> int:
        return int(self.confusion[1, 1])


def evaluate(name: str, y_true, y_pred, y_score) -> Evaluation:
    """Score one model's predictions on the held-out test set."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)

    return Evaluation(
        name=name,
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        metrics={
            "ROC-AUC": float(roc_auc_score(y_true, y_score)),
            "PR-AUC": float(average_precision_score(y_true, y_score)),
            "Accuracy": float(accuracy_score(y_true, y_pred)),
            "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
        },
        confusion=confusion_matrix(y_true, y_pred, labels=[0, 1]),
    )


def evaluate_pipeline(name: str, pipeline, X_test, y_test) -> Evaluation:
    """Predict and score a fitted sklearn pipeline on the test split.

    Works for models that expose ``predict_proba`` and for SVM, which only
    exposes ``decision_function``.
    """
    y_pred = pipeline.predict(X_test)
    if hasattr(pipeline, "predict_proba"):
        y_score = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_score = pipeline.decision_function(X_test)
    return evaluate(name, y_test, y_pred, y_score)


def baseline_evaluation(name: str, y_true, y_score=None) -> Evaluation:
    """The trivial 'always predict the majority class' reference model.

    ``ROC-AUC`` and ``PR-AUC`` are left as NaN because a constant classifier has
    no meaningful ranking ability. ``PR-AUC`` is set to the positive rate,
    which is the correct no-skill reference for average precision.
    """
    y_true = np.asarray(y_true).astype(int)
    # The *most frequent* class, not ``y_true.max()``: with an 8.8% positive
    # rate the majority class is 0, and ``max()`` would return 1 and predict
    # every patient as diabetic.
    majority = int(np.bincount(y_true).argmax())
    y_pred = np.full_like(y_true, fill_value=majority)

    if y_score is None:
        y_score = np.full(len(y_true), float(y_true.mean()))

    return Evaluation(
        name=name,
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        metrics={
            "ROC-AUC": float("nan"),
            "PR-AUC": float(y_true.mean()),
            "Accuracy": float(accuracy_score(y_true, y_pred)),
            "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
        },
        confusion=confusion_matrix(y_true, y_pred, labels=[0, 1]),
    )


def metrics_table(
    evaluations: list[Evaluation], round_to: int = 4
) -> pd.DataFrame:
    """Build the final comparison table.

    A false negative -- telling a diabetic patient they are fine -- is the
    costly error in screening, so Recall, PR-AUC and F1 are the columns to read
    first.
    """
    rows = []
    for ev in evaluations:
        row = {"Model": ev.name}
        row.update({k: round(v, round_to) for k, v in ev.metrics.items()})
        row.update({"TP": ev.tp, "FP": ev.fp, "TN": ev.tn, "FN": ev.fn})
        rows.append(row)

    table = pd.DataFrame(rows).set_index("Model")
    ordered = METRIC_ORDER + ["TP", "FP", "TN", "FN"]
    return table[ordered]


def final_result_table(
    evaluations: list[Evaluation], round_to: int = 4
) -> pd.DataFrame:
    """Build the result table in the exact form the project brief requires.

    Six columns -- ``1. Model list``, ``2. AUC``, ``3. Accuracy``,
    ``4. Precision``, ``5. Recall``, ``6. F1-Score`` -- and four rows in the
    required order: Log St, DT, SVM, XGBoost. The majority-class baseline is
    deliberately *excluded* because the brief specifies four models; it is
    reported separately as a reference line so the accuracy column is not read
    in isolation.
    """
    mapping = {
        "ROC-AUC": "2. AUC",
        "Accuracy": "3. Accuracy",
        "Precision": "4. Precision",
        "Recall": "5. Recall",
        "F1-Score": "6. F1-Score",
    }

    rows = []
    for ev in evaluations:
        if ev.name not in RESULT_LABELS:
            continue
        row = {"1. Model list": RESULT_LABELS[ev.name]}
        for key, column in mapping.items():
            row[column] = round(ev.metrics[key], round_to)
        rows.append(row)

    table = pd.DataFrame(rows).set_index("1. Model list")
    missing = [r for r in RESULT_ROW_ORDER if r not in table.index]
    if missing:
        raise ValueError(f"final result table is missing rows: {missing}")
    return table.loc[RESULT_ROW_ORDER, RESULT_COLUMN_ORDER]


def baseline_reference_line(evaluations: list[Evaluation]) -> str:
    """One-line text reference for the majority-class baseline.

    Rendered beneath the required four-row table so the accuracy column is
    always read against the number it must beat, without adding a fifth row
    that the brief does not ask for.
    """
    for ev in evaluations:
        if ev.name == "Majority Baseline":
            return (
                f"Majority-class baseline (always predict the most frequent class): "
                f"Accuracy {ev.metrics['Accuracy']:.4f}, "
                f"Recall {ev.metrics['Recall']:.4f}, "
                f"PR-AUC {ev.metrics['PR-AUC']:.4f}"
            )
    return "Majority-class baseline: not computed"


def roc_points(ev: Evaluation):
    """Return (fpr, tpr, thresholds) for plotting one ROC curve."""
    return roc_curve(ev.y_true, ev.y_score)


def pr_points(ev: Evaluation):
    """Return (recall, precision, thresholds) for plotting one PR curve."""
    precision, recall, thresholds = precision_recall_curve(ev.y_true, ev.y_score)
    return recall, precision, thresholds


def confusion_frame(ev: Evaluation) -> pd.DataFrame:
    """Confusion matrix as a labelled DataFrame, indexed by actual class."""
    return pd.DataFrame(
        ev.confusion,
        index=["Negative (actual)", "Positive (actual)"],
        columns=["Predicted Negative", "Predicted Positive"],
    )
