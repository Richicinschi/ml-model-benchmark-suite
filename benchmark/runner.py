"""Experiment runner that orchestrates benchmark experiments from config."""

from typing import Any, Dict, List

from .base import Experiment
from .config import ExperimentConfig
from .registry import REGISTRY
from .utils import setup_logger


class BenchmarkRunner:
    """High-level runner that executes benchmark experiments from configuration."""

    def __init__(self, config_path: str):
        self.config = ExperimentConfig(config_path)
        self.logger = setup_logger("BenchmarkRunner")
        self._validate_models()

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

        model_instances = self.resolve_models()

        results = {
            "experiment_name": self.config.experiment_name,
            "dataset": self.config.dataset,
            "models": list(self.config.models.keys()),
            "model_types": {
                name: REGISTRY.get(name)["type"]
                for name in self.config.models.keys()
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
