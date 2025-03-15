"""Cross-validation strategies for classification and regression."""

from typing import Any, Dict, Union

import pandas as pd
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, KFold


def build_cv_strategy(
    config: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
) -> Union[StratifiedKFold, TimeSeriesSplit, KFold]:
    """Build a cross-validation splitter based on configuration and target type."""
    strategy = config.get("strategy", "stratified_kfold")
    n_splits = config.get("folds", 5)
    shuffle = config.get("shuffle", True)
    random_state = config.get("random_state", 42)

    if strategy == "stratified_kfold":
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )
    elif strategy == "time_series_split":
        return TimeSeriesSplit(n_splits=n_splits)
    elif strategy == "kfold":
        return KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )
    else:
        raise ValueError(f"Unsupported CV strategy: {strategy}")


def get_cv_splits(
    cv: Union[StratifiedKFold, TimeSeriesSplit, KFold],
    X: pd.DataFrame,
    y: pd.Series,
):
    """Generate train/validation indices from a CV splitter."""
    return list(cv.split(X, y))
