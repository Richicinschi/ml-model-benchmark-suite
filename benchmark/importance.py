"""Feature importance extraction and comparison across models."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .registry import REGISTRY


def extract_feature_importance(
    model_name: str,
    fitted_model: Any,
    feature_names: Optional[List[str]] = None,
) -> Optional[Dict[str, float]]:
    """Extract feature importance from a fitted model if available."""
    model = fitted_model.model if hasattr(fitted_model, "model") else fitted_model

    importances = None

    # Tree-based models
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    # Linear models (coefficients as proxy)
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            # For multi-class, average absolute coefficients across classes
            importances = np.mean(np.abs(coef), axis=0)
        else:
            importances = np.abs(coef)

    if importances is None:
        return None

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    return {
        name: float(imp)
        for name, imp in zip(feature_names, importances)
    }


def rank_features(
    importance_dict: Dict[str, float],
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """Rank features by importance and return a sorted DataFrame."""
    df = pd.DataFrame(
        [(name, val) for name, val in importance_dict.items()],
        columns=["feature", "importance"],
    )
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    if top_n is not None:
        df = df.head(top_n)
    return df


def compare_feature_importances(
    model_importances: Dict[str, Dict[str, float]],
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """Compare feature importances across multiple models in a single DataFrame."""
    frames = []
    for model_name, imp_dict in model_importances.items():
        df = rank_features(imp_dict, top_n=top_n)
        df["model"] = model_name
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
