"""Base abstractions for experiments, models, and datasets."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import pandas as pd


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders.

    Implementations must define the ``load`` method to return a feature
    matrix ``X`` and target vector ``y`` from a given data source.
    """

    @abstractmethod
    def load(self, **kwargs: Any) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and return features (X) and target (y).

        Args:
            **kwargs: Source-specific loading options.

        Returns:
            A tuple of ``(X, y)`` where ``X`` is a :class:`pd.DataFrame`
            of features and ``y`` is a :class:`pd.Series` of target values.
        """
        pass


class ModelWrapper(ABC):
    """Abstract base class for model wrappers.

    Provides a unified interface for training, predicting, and extracting
    probability estimates from any underlying estimator.

    Args:
        name: Registered model identifier.
        model: The underlying estimator instance.
        hyperparams: Optional dictionary of hyperparameters used to build
            the model.
    """

    def __init__(self, name: str, model: Any, hyperparams: Optional[Dict[str, Any]] = None):
        self.name = name
        self.model = model
        self.hyperparams = hyperparams or {}

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model on the provided data.

        Args:
            X: Feature matrix.
            y: Target vector.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Any:
        """Generate predictions for the given features.

        Args:
            X: Feature matrix.

        Returns:
            Predicted labels or values.
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> Optional[Any]:
        """Generate probability predictions if supported by the model.

        Args:
            X: Feature matrix.

        Returns:
            Probability array when supported, otherwise ``None``.
        """
        pass


class Experiment(ABC):
    """Abstract base class for benchmark experiments.

    Args:
        name: Human-readable experiment name.
        config: Experiment configuration dictionary.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Run the experiment and return results.

        Returns:
            A dictionary containing experiment outputs, metrics, and
            metadata.
        """
        pass
