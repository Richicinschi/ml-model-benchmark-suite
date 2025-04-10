"""Visualization utilities for benchmark results and model comparisons."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


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
