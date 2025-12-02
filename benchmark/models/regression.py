"""Regression model wrappers registered in the global model registry."""

from typing import Optional, Any

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor as _GradientBoostingRegressor
from sklearn.linear_model import ElasticNet as _ElasticNet
from sklearn.linear_model import Lasso as _Lasso
from sklearn.linear_model import LinearRegression as _LinearRegression
from sklearn.linear_model import Ridge as _Ridge
try:
    from xgboost import XGBRegressor as _XGBRegressor
except ImportError:
    _XGBRegressor = None  # type: ignore

from ..base import ModelWrapper
from ..registry import register_model


@register_model("linear_regression", "regression", default_params={})
class LinearRegression(ModelWrapper):
    """Linear regression wrapper."""

    def __init__(self, **kwargs: Any):
        super().__init__("linear_regression", _LinearRegression(**kwargs), hyperparams=kwargs)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> Any:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Optional[Any]:
        return None


if _XGBRegressor is not None:
    @register_model("xgboost_regressor", "regression", default_params={"n_estimators": 100, "max_depth": 3})
    class XGBoostRegressor(ModelWrapper):
        """XGBoost regressor wrapper."""

        def __init__(self, **kwargs: Any):
            super().__init__("xgboost_regressor", _XGBRegressor(**kwargs), hyperparams=kwargs)

        def train(self, X: pd.DataFrame, y: pd.Series) -> None:
            self.model.fit(X, y)

        def predict(self, X: pd.DataFrame) -> Any:
            return self.model.predict(X)

        def predict_proba(self, X: pd.DataFrame) -> Optional[Any]:
            return None


@register_model("ridge", "regression", default_params={"alpha": 1.0})
class Ridge(ModelWrapper):
    """Ridge regression wrapper."""

    def __init__(self, **kwargs: Any):
        super().__init__("ridge", _Ridge(**kwargs), hyperparams=kwargs)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> Any:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Optional[Any]:
        return None


@register_model("lasso", "regression", default_params={"alpha": 1.0, "max_iter": 1000})
class Lasso(ModelWrapper):
    """Lasso regression wrapper."""

    def __init__(self, **kwargs: Any):
        super().__init__("lasso", _Lasso(**kwargs), hyperparams=kwargs)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> Any:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Optional[Any]:
        return None


@register_model("elasticnet", "regression", default_params={"alpha": 1.0, "l1_ratio": 0.5, "max_iter": 1000})
class ElasticNet(ModelWrapper):
    """ElasticNet regression wrapper."""

    def __init__(self, **kwargs: Any):
        super().__init__("elasticnet", _ElasticNet(**kwargs), hyperparams=kwargs)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> Any:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Optional[Any]:
        return None


@register_model("gradient_boosting", "regression", default_params={"n_estimators": 100})
class GradientBoostingRegressor(ModelWrapper):
    """Gradient boosting regressor wrapper."""

    def __init__(self, **kwargs: Any):
        super().__init__("gradient_boosting", _GradientBoostingRegressor(**kwargs), hyperparams=kwargs)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> Any:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Optional[Any]:
        return None
