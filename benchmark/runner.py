"""Experiment runner that orchestrates benchmark experiments from config."""

from typing import Any, Dict, List, Tuple

import pandas as pd

from .base import Experiment
from .config import ExperimentConfig
from .data import load_dataset
# Import models to ensure they register themselves in the global registry
from . import models  # noqa: F401
from .preprocessing import PreprocessingPipeline
from .registry import REGISTRY
from .utils import setup_logger


class BenchmarkRunner:
    """High-level runner that executes benchmark experiments from configuration."""

    def __init__(self, config_path: str):
        self.config = ExperimentConfig(config_path)
        self.logger = setup_logger("BenchmarkRunner")
        self._validate_models()
        self.X: pd.DataFrame = pd.DataFrame()
        self.y: pd.Series = pd.Series(dtype="object")
        self.preprocessor: PreprocessingPipeline = PreprocessingPipeline()

    def _validate_models(self) -> None:
        """Ensure all requested models are registered."""
        unknown: List[str] = []
        for name in self.config.models.keys():
            if not REGISTRY.is_registered(name):
                unknown.append(name)
                self.logger.error(f"Unknown model in config: {name}")
        if unknown:
            available = list(REGISTRY.list_models().keys())
            raise ValueError(
                f"Unregistered models: {unknown}. Available: {available}"
            )
        self.logger.info(f"All {len(self.config.models)} models validated")

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load dataset according to configuration."""
        self.X, self.y = load_dataset(self.config.dataset)
        self.logger.info(f"Dataset loaded: X={self.X.shape}, y={self.y.shape}")
        return self.X, self.y

    def preprocess_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Build and apply preprocessing pipeline."""
        self.preprocessor = PreprocessingPipeline(self.config.preprocessing).build(self.X)
        X_transformed = self.preprocessor.fit_transform(self.X)
        y_transformed = self.preprocessor.fit_transform_target(self.y)
        self.logger.info(f"Preprocessing complete: X={X_transformed.shape}")
        return X_transformed, y_transformed

    def resolve_models(self) -> Dict[str, Any]:
        """Build model instances from configuration."""
        instances: Dict[str, Any] = {}
        for name, overrides in self.config.models.items():
            instances[name] = REGISTRY.build(name, overrides or None)
            self.logger.info(f"Instantiated model: {name}")
        return instances

    def run(self) -> Dict[str, Any]:
        """Execute the experiment defined in the configuration."""
        self.logger.info(f"Starting experiment: {self.config.experiment_name}")
        self.logger.info(f"Dataset config: {self.config.dataset}")
        self.logger.info(f"Models: {list(self.config.models.keys())}")

        self.load_data()
        X_processed, y_processed = self.preprocess_data()
        model_instances = self.resolve_models()

        results = {
            "experiment_name": self.config.experiment_name,
            "dataset": self.config.dataset,
            "models": list(self.config.models.keys()),
            "model_types": {
                name: REGISTRY.get(name)["type"]
                for name in self.config.models.keys()
            },
            "preprocessing": self.config.preprocessing,
            "data_shape": {
                "raw": {"X": self.X.shape, "y": self.y.shape},
                "processed": {"X": X_processed.shape, "y": y_processed.shape},
            },
            "status": "completed",
            "results": {},
        }

        self.logger.info("Experiment completed successfully")
        return results


class ExperimentRunner(Experiment):
    """Concrete experiment implementation driven by configuration."""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.logger = setup_logger(f"Experiment:{name}")

    def run(self) -> Dict[str, Any]:
        self.logger.info(f"Running experiment: {self.name}")
        return {
            "experiment_name": self.name,
            "config": self.config,
            "status": "completed",
        }
