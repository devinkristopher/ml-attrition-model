"""Model evaluation utilities."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(
    y_true: Any,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
) -> dict[str, float]:
    """Calculate classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, y_probability)
        ),
    }


def thresholds_passed(
    metrics: dict[str, float],
    thresholds: dict[str, float],
) -> bool:
    """Return whether all configured thresholds were met."""
    passed = True

    for metric_name, threshold in thresholds.items():
        actual = metrics[metric_name]

        if actual < threshold:
            print(
                f"FAILED: {metric_name}={actual:.4f} "
                f"is below {threshold:.4f}"
            )
            passed = False

    return passed