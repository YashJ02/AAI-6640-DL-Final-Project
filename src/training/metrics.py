"""Training and evaluation metrics, including statistical significance tests."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

try:
    from scipy.stats import chi2

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Compute aggregate and per-class classification metrics."""
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=["down", "neutral", "up"],
        output_dict=True,
        zero_division=0,
    )
    metrics["classification_report"] = report

    return metrics


def majority_class_baseline(y_true: np.ndarray, y_train: np.ndarray) -> dict[str, float]:
    """Score a naive majority-class predictor for context."""
    if y_train.size == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0}

    majority = int(np.bincount(y_train).argmax())
    preds = np.full_like(y_true, fill_value=majority)

    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
    }


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> dict[str, float]:
    """Compute McNemar's chi-squared test from two paired classifiers."""
    if not (len(y_true) == len(y_pred_a) == len(y_pred_b)):
        raise ValueError("Input arrays must have equal length")

    a_correct = y_pred_a == y_true
    b_correct = y_pred_b == y_true

    n01 = int(np.sum(a_correct & ~b_correct))
    n10 = int(np.sum(~a_correct & b_correct))

    denominator = n01 + n10
    if denominator == 0:
        return {"n01": float(n01), "n10": float(n10), "chi2": 0.0, "p_value": 1.0}

    chi2_stat = ((n01 - n10) ** 2) / denominator

    if HAS_SCIPY:
        p_value = float(1.0 - chi2.cdf(chi2_stat, df=1))
    else:
        # Conservative fallback if scipy is unavailable.
        p_value = float("nan")

    return {
        "n01": float(n01),
        "n10": float(n10),
        "chi2": float(chi2_stat),
        "p_value": p_value,
    }


def summarize_fold_metrics(fold_metrics: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Aggregate scalar metrics across folds as mean and standard deviation."""
    if not fold_metrics:
        return {}

    scalar_keys: list[str] = []
    for key, value in fold_metrics[0].items():
        if isinstance(value, (float, int)):
            scalar_keys.append(key)

    summary: dict[str, dict[str, float]] = {}
    for key in scalar_keys:
        values = np.array([float(metrics[key]) for metrics in fold_metrics], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }

    return summary
