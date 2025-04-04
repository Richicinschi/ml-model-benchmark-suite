"""Metrics computation for classification and regression tasks."""

from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
)


def compute_classification_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    y_proba: Optional[Union[List, np.ndarray]] = None,
    average: str = "weighted",
    multi_class: str = "ovr",
) -> Dict[str, float]:
    """Compute standard classification metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        try:
            if len(np.unique(y_true)) > 2:
                metrics["roc_auc"] = roc_auc_score(
                    y_true, y_proba, multi_class=multi_class, average=average
                )
            else:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
        except Exception:
            metrics["roc_auc"] = None

    return metrics


def compute_regression_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
) -> Dict[str, float]:
    """Compute standard regression metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def compute_confusion_matrix(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
) -> List[List[int]]:
    """Compute confusion matrix as a nested list."""
    return confusion_matrix(y_true, y_pred).tolist()


def compute_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    task_type: str,
    y_proba: Optional[Union[List, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Dispatch to classification or regression metrics based on task type."""
    if task_type == "classification":
        metrics = compute_classification_metrics(y_true, y_pred, y_proba)
        metrics["confusion_matrix"] = compute_confusion_matrix(y_true, y_pred)
        return metrics
    elif task_type == "regression":
        return compute_regression_metrics(y_true, y_pred)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
