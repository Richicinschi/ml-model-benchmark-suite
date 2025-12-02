"""LightGBM model wrappers registered in the global model registry."""

from typing import Any, Optional

import pandas as pd
from lightgbm import LGBMClassifier as _LGBMClassifier
from lightgbm import LGBMRegressor as _LGBMRegressor

from ..base import ModelWrapper
from ..registry import register_model


@register_model(
    "lightgbm_classifier",
    "classification",
    default_params={"n_estimators": 100, "random_state": 42, "verbosity": -1},
)
class LightGBMClassifier(ModelWrapper):
    """LightGBM classifier wrapper."""

    def __init__(self, **kwargs: Any):
        super().__init__("lightgbm_classifier", _LGBMClassifier(**kwargs), hyperparams=kwargs)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> Any:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Optional[Any]:
        return self.model.predict_proba(X)


@register_model(
    "lightgbm_regressor",
    "regression",
    default_params={"n_estimators": 100, "random_state": 42, "verbosity": -1},
)
class LightGBMRegressor(ModelWrapper):
    """LightGBM regressor wrapper."""

    def __init__(self, **kwargs: Any):
        super().__init__("lightgbm_regressor", _LGBMRegressor(**kwargs), hyperparams=kwargs)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> Any:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Optional[Any]:
        return None
