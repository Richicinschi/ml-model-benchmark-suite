"""SHAP value analysis for model explainability."""

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


def _get_shap_explainer(model: Any) -> Any:
    """Create an appropriate SHAP explainer for the given model."""
    if not HAS_SHAP:
        raise ImportError("shap is required for SHAP analysis")

    # Tree-based models
    if hasattr(model, "tree_") or hasattr(model, "estimators_"):
        return shap.TreeExplainer(model)
    # Linear models
    elif hasattr(model, "coef_"):
        return shap.LinearExplainer(model, masker=shap.sample(model, 10))
    else:
        return shap.KernelExplainer(model.predict, shap.sample(model, 10))


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

    try:
        explainer = _get_shap_explainer(model)
        shap_values = explainer.shap_values(X.values if hasattr(X, "values") else X)

        # For multi-class, use mean absolute SHAP across classes
        if isinstance(shap_values, list):
            mean_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            shap_values = mean_shap
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
