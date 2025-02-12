"""Classification model wrappers registered in the global model registry."""

from typing import Optional, Any

import pandas as pd
from sklearn.linear_model import LogisticRegression as _LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as _KNeighborsClassifier

from ..base import ModelWrapper
from ..registry import register_model


@register_model("logistic_regression", "classification", default_params={"max_iter": 1000})
class LogisticRegression(ModelWrapper):
    """Logistic regression classifier wrapper."""

    def __init__(self, **kwargs: Any):
        super().__init__("logistic_regression", _LogisticRegression(**kwargs), hyperparams=kwargs)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> Any:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Optional[Any]:
        return self.model.predict_proba(X)


@register_model("knn", "classification", default_params={"n_neighbors": 5})
class KNeighborsClassifier(ModelWrapper):
    """K-nearest neighbors classifier wrapper."""

    def __init__(self, **kwargs: Any):
        super().__init__("knn", _KNeighborsClassifier(**kwargs), hyperparams=kwargs)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> Any:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Optional[Any]:
        return self.model.predict_proba(X)
