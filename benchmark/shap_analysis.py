"""SHAP value analysis for model explainability."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    shap = None  # type: ignore

from .utils import setup_logger


def _get_shap_explainer(model: Any, X: Any = None) -> Any:
    """Create an appropriate SHAP explainer for the given model."""
    if not HAS_SHAP:
        raise ImportError("shap is required for SHAP analysis")

    # Tree-based models (sklearn, xgboost, lightgbm, catboost)
    if (
        hasattr(model, "tree_")
        or hasattr(model, "estimators_")
        or hasattr(model, "get_booster")
        or hasattr(model, "_Booster")
        or hasattr(model, "booster")
    ):
        return shap.TreeExplainer(model)
    # Linear models
    elif hasattr(model, "coef_"):
        if X is not None:
            masker = shap.sample(X, 10) if hasattr(X, "shape") else X
            return shap.LinearExplainer(model, masker=masker)
        else:
            return shap.KernelExplainer(model.predict, shap.sample(X, 10) if X is not None else X)
    else:
        background = shap.sample(X, 10) if X is not None else X
        return shap.KernelExplainer(model.predict, background)


def compute_shap_values(
    fitted_model: Any,
    X: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Compute SHAP values for a fitted model and dataset."""
    logger = setup_logger("SHAP")

    if not HAS_SHAP:
        logger.warning("shap package not installed; skipping SHAP analysis")
        return None

    model = fitted_model.model if hasattr(fitted_model, "model") else fitted_model
    cols = feature_names if feature_names is not None else list(X.columns)

    # Sample X for faster SHAP computation during testing
    max_shap_samples = 2000
    if hasattr(X, "shape") and X.shape[0] > max_shap_samples:
        X = X.sample(n=max_shap_samples, random_state=42)

    try:
        explainer = _get_shap_explainer(model, X)
        shap_values = explainer.shap_values(X.values if hasattr(X, "values") else X)

        # For multi-class, use mean absolute SHAP across classes
        if isinstance(shap_values, list):
            mean_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            shap_values = mean_shap
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # New SHAP format: (n_samples, n_features, n_classes)
            mean_shap = np.mean(np.abs(shap_values), axis=(0, 2))
            if shap_values.shape[2] == 2:
                shap_values = shap_values[:, :, 1]
            else:
                shap_values = np.mean(shap_values, axis=2)
        else:
            mean_shap = np.abs(shap_values)

        importance = {
            col: float(np.mean(mean_shap[:, i]))
            for i, col in enumerate(cols)
        }

        logger.info(f"SHAP values computed for {len(cols)} features")
        return {
            "shap_values": shap_values.tolist(),
            "base_value": float(explainer.expected_value) if not isinstance(explainer.expected_value, (list, np.ndarray)) else None,
            "feature_importance": importance,
        }
    except Exception as e:
        logger.error(f"SHAP computation failed: {e}")
        return None


def plot_shap_summary(
    shap_values: Any,
    X: pd.DataFrame,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> Optional[Any]:
    """Generate a SHAP summary plot as a matplotlib figure."""
    if not HAS_SHAP:
        return None
    try:
        import matplotlib.pyplot as plt
        shap_values_arr = np.asarray(shap_values)
        if shap_values_arr.ndim == 3:
            if shap_values_arr.shape[2] == 2:
                shap_values_arr = shap_values_arr[:, :, 1]
            else:
                shap_values_arr = np.mean(shap_values_arr, axis=2)
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values_arr, X, show=False)
        fig = plt.gcf()
        if title:
            fig.suptitle(title, y=1.02)
        plt.tight_layout()
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return None
        return fig
    except Exception:
        return None


def plot_shap_dependence(
    shap_values: Any,
    X: pd.DataFrame,
    feature: str,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> Optional[Any]:
    """Generate a SHAP dependence plot for a specific feature."""
    if not HAS_SHAP:
        return None
    try:
        import matplotlib.pyplot as plt
        shap_values_arr = np.asarray(shap_values)
        if shap_values_arr.ndim == 3:
            if shap_values_arr.shape[2] == 2:
                shap_values_arr = shap_values_arr[:, :, 1]
            else:
                shap_values_arr = np.mean(shap_values_arr, axis=2)
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(feature, shap_values_arr, X, show=False)
        fig = plt.gcf()
        if title:
            fig.suptitle(title, y=1.02)
        plt.tight_layout()
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return None
        return fig
    except Exception:
        return None


def rank_shap_importance(
    importance_dict: Dict[str, float],
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """Rank features by mean absolute SHAP value."""
    df = pd.DataFrame(
        [(name, val) for name, val in importance_dict.items()],
        columns=["feature", "mean_abs_shap"],
    )
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    if top_n is not None:
        df = df.head(top_n)
    return df
