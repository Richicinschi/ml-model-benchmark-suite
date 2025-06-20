"""Overfitting detection based on train/validation metric gaps."""

from typing import Any, Dict, List, Optional


class OverfittingDetector:
    """Detects overfitting by comparing train and validation metrics across CV folds."""

    DEFAULT_THRESHOLDS = {
        "accuracy": 0.05,
        "precision": 0.05,
        "recall": 0.05,
        "f1": 0.05,
        "roc_auc": 0.05,
        "r2": 0.10,
        "mse": 0.15,  # relative gap for regression losses
        "rmse": 0.15,
        "mae": 0.15,
    }

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = thresholds or dict(self.DEFAULT_THRESHOLDS)

    def detect(
        self,
        fold_results: List[Dict[str, Any]],
        primary_metric: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze fold results and return overfitting summary."""
        if not fold_results:
            return {"status": "unknown", "warnings": []}

        # Determine primary metric from available val metrics
        if primary_metric is None:
            sample_val = fold_results[0].get("val_metrics", {})
            for candidate in ("accuracy", "f1", "r2", "roc_auc"):
                if candidate in sample_val:
                    primary_metric = candidate
                    break
            if primary_metric is None:
                primary_metric = list(sample_val.keys())[0]

        train_scores = [f["train_metrics"].get(primary_metric) for f in fold_results]
        val_scores = [f["val_metrics"].get(primary_metric) for f in fold_results]

        if None in train_scores or None in val_scores:
            return {"status": "unknown", "warnings": []}

        avg_train = sum(train_scores) / len(train_scores)
        avg_val = sum(val_scores) / len(val_scores)
        gap = avg_train - avg_val

        # For loss-based metrics, invert interpretation
        is_loss_metric = primary_metric in ("mse", "rmse", "mae")
        if is_loss_metric:
            gap = avg_val - avg_train

        threshold = self.thresholds.get(primary_metric, 0.05)
        relative_gap = gap / (abs(avg_train) + 1e-9)

        warnings: List[str] = []
        if gap > threshold:
            warnings.append(
                f"Train/val gap for {primary_metric} is {gap:.4f} "
                f"(threshold {threshold:.4f}) — possible overfitting."
            )
        if relative_gap > threshold * 2:
            warnings.append(
                f"Relative train/val gap for {primary_metric} is {relative_gap:.2%} — "
                f"model may not generalize well."
            )

        status = "overfitting" if warnings else "ok"
        return {
            "status": status,
            "primary_metric": primary_metric,
            "avg_train": avg_train,
            "avg_val": avg_val,
            "gap": gap,
            "relative_gap": relative_gap,
            "threshold": threshold,
            "warnings": warnings,
        }
