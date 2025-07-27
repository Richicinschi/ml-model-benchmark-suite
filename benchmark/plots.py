"""Visualization utilities for benchmark results and model comparisons."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import label_binarize


def plot_multi_metric_comparison(
    model_metrics: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Grouped bar plot comparing multiple metrics across models."""
    models = list(model_metrics.keys())
    if not models:
        return None

    metric_names = list(next(iter(model_metrics.values())).keys())
    x = np.arange(len(metric_names))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(max(7, len(metric_names) * 1.5), 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(models)))

    for i, (model, color) in enumerate(zip(models, colors)):
        values = [model_metrics[model].get(m, 0.0) for m in metric_names]
        offset = width * (i - len(models) / 2 + 0.5)
        bars = ax.bar(x + offset, values, width, label=model, color=color)
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                f"{val:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90 if len(metric_names) > 4 else 0,
            )

    ax.set_ylabel("Score")
    ax.set_title(title or "Multi-Metric Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=30, ha="right")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def plot_model_comparison(
    metric_values: Dict[str, float],
    metric_name: str,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Bar plot comparing a single metric across multiple models."""
    models = list(metric_values.keys())
    values = list(metric_values.values())

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 0.8), 5))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
    bars = ax.bar(models, values, color=colors)
    ax.set_ylabel(metric_name)
    ax.set_title(title or f"Model Comparison: {metric_name}")
    ax.set_ylim(0, max(values) * 1.15 if values else 1)
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f"{val:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def plot_precision_recall_curve(
    y_true: List[int],
    y_proba: np.ndarray,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plot precision-recall curve(s) for binary or multi-class classification."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    fig, ax = plt.subplots(figsize=(6, 6))
    classes = np.unique(y_true)

    if len(classes) == 2 and y_proba.ndim > 1:
        y_proba = y_proba[:, 1]

    if y_proba.ndim == 1:
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        ax.plot(recall, precision, label=f"PR curve (AP = {ap:.3f})")
    else:
        y_true_bin = label_binarize(y_true, classes=classes)
        if y_true_bin.shape[1] == 2:
            precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
            ap = average_precision_score(y_true, y_proba[:, 1])
            ax.plot(recall, precision, label=f"PR curve (AP = {ap:.3f})")
        else:
            for i, cls in enumerate(classes):
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
                ap = average_precision_score(y_true_bin[:, i], y_proba[:, i])
                ax.plot(recall, precision, label=f"Class {cls} (AP = {ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title or "Precision-Recall Curve")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def plot_train_val_curves(
    fold_results: List[Dict[str, Any]],
    metric_name: str,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plot train vs validation metric across CV folds to detect overfitting."""
    folds = [f["fold"] for f in fold_results]
    train_scores = [f["train_metrics"][metric_name] for f in fold_results]
    val_scores = [f["val_metrics"][metric_name] for f in fold_results]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(folds, train_scores, marker="o", label="Train", color="tab:blue")
    ax.plot(folds, val_scores, marker="s", label="Validation", color="tab:orange")
    ax.set_xlabel("Fold")
    ax.set_ylabel(metric_name)
    ax.set_title(title or f"Train vs Validation: {metric_name}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def plot_calibration_curve(
    y_true: List[float],
    y_prob: List[List[float]],
    n_bins: int = 10,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plot a simple calibration curve for binary classification probabilities."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if y_prob.ndim > 1 and y_prob.shape[1] == 2:
        y_prob = y_prob[:, 1]

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_accuracies = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = bin_indices == i
        count = mask.sum()
        if count > 0:
            bin_accuracies[i] = y_true[mask].mean()
            bin_counts[i] = count

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.plot(bin_centers, bin_accuracies, marker="o", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title or "Calibration Curve")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def plot_confusion_matrix_heatmap(
    matrix: List[List[int]],
    class_names: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plot a confusion matrix as a seaborn heatmap."""
    matrix = np.asarray(matrix)
    if class_names is None:
        class_names = [str(i) for i in range(matrix.shape[0])]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar=False,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or "Confusion Matrix")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def plot_roc_curve(
    y_true: List[int],
    y_proba: np.ndarray,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plot ROC curve(s) for binary or multi-class classification."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Random classifier")

    classes = np.unique(y_true)
    if len(classes) == 2 and y_proba.ndim > 1:
        y_proba = y_proba[:, 1]

    if y_proba.ndim == 1:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    else:
        y_true_bin = label_binarize(y_true, classes=classes)
        if y_true_bin.shape[1] == 2:
            y_true_bin = y_true_bin[:, 1]
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
        else:
            for i, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f"Class {cls} (AUC = {roc_auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title or "ROC Curve")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig



def plot_learning_curve(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 5),
    scoring: Optional[str] = None,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plot learning curve showing training and cross-validation scores."""
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes, scoring=scoring, n_jobs=1
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_sizes, train_mean, "o-", color="blue", label="Training score")
    ax.fill_between(
        train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue"
    )
    ax.plot(train_sizes, val_mean, "o-", color="green", label="Cross-validation score")
    ax.fill_between(
        train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="green"
    )

    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score" if scoring is None else f"Score ({scoring})")
    ax.set_title(title or "Learning Curve")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig



def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plot regression residuals: residuals vs fitted and residual distribution."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Residuals vs fitted
    ax = axes[0]
    ax.scatter(y_pred, residuals, alpha=0.6, edgecolors="k")
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")
    ax.grid(True, linestyle="--", alpha=0.6)

    # Residual distribution
    ax = axes[1]
    sns.histplot(residuals, kde=True, ax=ax, color="steelblue")
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Residuals")
    ax.set_title("Residual Distribution")
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle(title or "Residual Analysis")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def plot_feature_importance_comparison(
    model_importances: Dict[str, Dict[str, float]],
    top_n: int = 10,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Grouped horizontal bar plot comparing top feature importances across models."""
    if not model_importances:
        return None

    # Gather all features and average their ranks across models to find overall top features
    feature_scores: Dict[str, float] = {}
    for importances in model_importances.values():
        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        for rank, (feature, _) in enumerate(sorted_features):
            feature_scores[feature] = feature_scores.get(feature, 0) + rank

    # Average rank (lower is better)
    for feature in feature_scores:
        feature_scores[feature] /= len(model_importances)

    top_features = sorted(feature_scores.keys(), key=lambda f: feature_scores[f])[:top_n]
    top_features = list(reversed(top_features))  # reverse for top-down plotting

    models = list(model_importances.keys())
    y = np.arange(len(top_features))
    height = 0.7 / len(models)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(models)))

    fig, ax = plt.subplots(figsize=(max(7, len(models) * 0.5 + 6), max(5, len(top_features) * 0.5)))
    for i, (model, color) in enumerate(zip(models, colors)):
        values = [model_importances[model].get(f, 0.0) for f in top_features]
        offset = height * (i - len(models) / 2 + 0.5)
        ax.barh(y + offset, values, height, label=model, color=color)

    ax.set_yticks(y)
    ax.set_yticklabels(top_features)
    ax.set_xlabel("Importance")
    ax.set_title(title or f"Top {top_n} Feature Importance Comparison")
    ax.legend()
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig
