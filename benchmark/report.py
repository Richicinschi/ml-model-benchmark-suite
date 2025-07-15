"""HTML report generation for benchmark experiment runs."""

import base64
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .plots import (
    plot_calibration_curve,
    plot_confusion_matrix_heatmap,
    plot_learning_curve,
    plot_model_comparison,
    plot_multi_metric_comparison,
    plot_precision_recall_curve,
    plot_residuals,
    plot_roc_curve,
    plot_train_val_curves,
)
from .shap_analysis import plot_shap_summary
from .utils import setup_logger


class ReportGenerator:
    """Generates an HTML report from experiment results."""

    def __init__(self, template_dir: Optional[str] = None):
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(["html", "xml"]),
        )
        self.env.filters["tojson"] = lambda x: str(x)
        self.logger = setup_logger("ReportGenerator")

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """Convert a matplotlib figure to a base64-encoded PNG string."""
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{img_base64}"

    def _generate_plots(
        self,
        results: Dict[str, Any],
        X=None,
        y=None,
        models=None,
    ) -> Dict[str, str]:
        """Generate comparison and overfitting plots from experiment results."""
        plots = {}
        model_results = results.get("results", {})
        if not model_results:
            return plots

        # Model comparison on a primary metric
        first_model = list(model_results.values())[0]
        task_type = first_model.get("task_type", "classification")
        metric_key = "accuracy" if task_type == "classification" else "r2"

        comparison_values = {}
        for model_name, model_res in model_results.items():
            agg = model_res.get("aggregated", {})
            val_metrics = agg.get("val", {})
            mean_key = f"{metric_key}_mean"
            if mean_key in val_metrics:
                comparison_values[model_name] = val_metrics[mean_key]

        if len(comparison_values) > 1:
            fig = plot_model_comparison(
                comparison_values,
                metric_name=metric_key,
                title=f"Model Comparison ({metric_key})",
            )
            if fig:
                plots["model_comparison"] = self._fig_to_base64(fig)

        # Multi-metric side-by-side comparison
        multi_metrics = {}
        for model_name, model_res in model_results.items():
            agg = model_res.get("aggregated", {})
            val_metrics = agg.get("val", {})
            model_scores = {}
            for key in val_metrics:
                if key.endswith("_mean"):
                    short_key = key.replace("_mean", "")
                    if short_key not in ("confusion_matrix",):
                        model_scores[short_key] = val_metrics[key]
            if model_scores:
                multi_metrics[model_name] = model_scores

        if len(multi_metrics) > 1:
            fig = plot_multi_metric_comparison(
                multi_metrics,
                title="Multi-Metric Model Comparison",
            )
            if fig:
                plots["multi_metric_comparison"] = self._fig_to_base64(fig)

        # Train vs validation curves for overfitting detection
        for model_name, model_res in model_results.items():
            folds = model_res.get("folds", [])
            if folds:
                fig = plot_train_val_curves(
                    folds,
                    metric_name=metric_key,
                    title=f"{model_name} - Train vs Validation ({metric_key})",
                )
                if fig:
                    plots[f"train_val_{model_name}"] = self._fig_to_base64(fig)

        # Confusion matrix heatmaps and ROC curves for classification models (last fold)
        if task_type == "classification":
            for model_name, model_res in model_results.items():
                folds = model_res.get("folds", [])
                if folds:
                    last_fold = folds[-1]
                    cm = last_fold.get("val_metrics", {}).get("confusion_matrix")
                    if cm is not None:
                        fig = plot_confusion_matrix_heatmap(
                            cm,
                            title=f"{model_name} - Confusion Matrix (last fold)",
                        )
                        if fig:
                            plots[f"confusion_matrix_{model_name}"] = self._fig_to_base64(fig)

                    val_proba = last_fold.get("val_proba")
                    val_true = last_fold.get("val_true")
                    if val_proba is not None and val_true is not None:
                        import numpy as np
                        fig = plot_roc_curve(
                            val_true,
                            np.asarray(val_proba),
                            title=f"{model_name} - ROC Curve (last fold)",
                        )
                        if fig:
                            plots[f"roc_curve_{model_name}"] = self._fig_to_base64(fig)

                        fig = plot_precision_recall_curve(
                            val_true,
                            np.asarray(val_proba),
                            title=f"{model_name} - Precision-Recall Curve (last fold)",
                        )
                        if fig:
                            plots[f"pr_curve_{model_name}"] = self._fig_to_base64(fig)

                        # Calibration curve for binary classification
                        unique_classes = len(np.unique(val_true))
                        if unique_classes == 2:
                            fig = plot_calibration_curve(
                                val_true,
                                np.asarray(val_proba),
                                title=f"{model_name} - Calibration Curve (last fold)",
                            )
                            if fig:
                                plots[f"calibration_{model_name}"] = self._fig_to_base64(fig)

        # Residual plots for regression models (last fold)
        if task_type == "regression":
            for model_name, model_res in model_results.items():
                folds = model_res.get("folds", [])
                if folds:
                    last_fold = folds[-1]
                    val_true = last_fold.get("val_true")
                    val_preds = last_fold.get("val_preds")
                    if val_true is not None and val_preds is not None:
                        import numpy as np
                        fig = plot_residuals(
                            np.asarray(val_true),
                            np.asarray(val_preds),
                            title=f"{model_name} - Residuals (last fold)",
                        )
                        if fig:
                            plots[f"residuals_{model_name}"] = self._fig_to_base64(fig)

        # Learning curves
        if X is not None and y is not None and models:
            for model_name, model_res in model_results.items():
                if model_name not in models:
                    continue
                estimator = models[model_name]
                task_type = model_res.get("task_type", "classification")
                scoring = "accuracy" if task_type == "classification" else "neg_mean_squared_error"
                try:
                    fig = plot_learning_curve(
                        estimator,
                        X,
                        y,
                        cv=3,
                        scoring=scoring,
                        title=f"{model_name} - Learning Curve",
                    )
                    if fig:
                        plots[f"learning_curve_{model_name}"] = self._fig_to_base64(fig)
                except Exception as exc:
                    self.logger.warning(f"Learning curve failed for {model_name}: {exc}")

        # SHAP summary plots
        if X is not None:
            for model_name, model_res in model_results.items():
                shap_data = model_res.get("shap")
                if shap_data and shap_data.get("shap_values") is not None:
                    try:
                        fig = plot_shap_summary(
                            shap_data["shap_values"],
                            X,
                            title=f"{model_name} - SHAP Summary",
                        )
                        if fig:
                            plots[f"shap_summary_{model_name}"] = self._fig_to_base64(fig)
                    except Exception as exc:
                        self.logger.warning(f"SHAP summary plot failed for {model_name}: {exc}")

        return plots

    def generate(
        self,
        results: Dict[str, Any],
        output_path: str,
        X=None,
        y=None,
        models=None,
    ) -> str:
        """Render an HTML report and save it to disk."""
        template = self.env.get_template("report.html")

        model_results = results.get("results", {})
        aggregated_metrics = {
            name: res.get("aggregated", {})
            for name, res in model_results.items()
        }
        feature_importance = {
            name: res.get("feature_importance")
            for name, res in model_results.items()
            if res.get("feature_importance") is not None
        }
        overfitting = {
            name: res.get("overfitting", {})
            for name, res in model_results.items()
            if res.get("overfitting", {}).get("warnings")
        }

        plots = self._generate_plots(results, X=X, y=y, models=models)

        html = template.render(
            experiment_name=results.get("experiment_name", "Benchmark Report"),
            timestamp=datetime.utcnow().isoformat(),
            status=results.get("status", "unknown"),
            dataset=results.get("dataset", {}),
            models=results.get("models", []),
            aggregated_metrics=aggregated_metrics,
            feature_importance=feature_importance,
            overfitting=overfitting,
            plots=plots,
        )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        self.logger.info(f"Report saved to {out}")
        return str(out)
