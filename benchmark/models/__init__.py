"""Model implementations for classification and regression."""

from .classification import (
    LogisticRegression,
    KNeighborsClassifier,
    RandomForestClassifier,
    SVM,
)

try:
    from .classification import XGBoostClassifier
except ImportError:
    XGBoostClassifier = None  # type: ignore

from .regression import (
    LinearRegression,
    Ridge,
    Lasso,
)

__all__ = [
    "LogisticRegression",
    "KNeighborsClassifier",
    "RandomForestClassifier",
    "SVM",
    "LinearRegression",
    "Ridge",
    "Lasso",
]

if XGBoostClassifier is not None:
    __all__.append("XGBoostClassifier")
