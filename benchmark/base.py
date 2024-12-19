"""Base abstractions for experiments, models, and datasets."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import pandas as pd


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""

    @abstractmethod
    def load(self, **kwargs: Any) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and return features (X) and target (y)."""
        pass


class ModelWrapper(ABC):
    """Abstract base class for model wrappers."""

    def __init__(self, name: str, model: Any, hyperparams: Optional[Dict[str, Any]] = None):
        self.name = name
        self.model = model
        self.hyperparams = hyperparams or {}

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Any:
        """Generate predictions."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> Optional[Any]:
        """Generate probability predictions if supported."""
        pass


class Experiment(ABC):
    """Abstract base class for benchmark experiments."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Run the experiment and return results."""
        pass
