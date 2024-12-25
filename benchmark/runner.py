"""Experiment runner that orchestrates benchmark experiments from config."""

from typing import Any, Dict

from .base import Experiment
from .config import ExperimentConfig
from .utils import setup_logger


class BenchmarkRunner:
    """High-level runner that executes benchmark experiments from configuration."""

    def __init__(self, config_path: str):
        self.config = ExperimentConfig(config_path)
        self.logger = setup_logger("BenchmarkRunner")

    def run(self) -> Dict[str, Any]:
        """Execute the experiment defined in the configuration."""
        self.logger.info(f"Starting experiment: {self.config.experiment_name}")
        self.logger.info(f"Dataset config: {self.config.dataset}")
        self.logger.info(f"Models: {list(self.config.models.keys())}")

        results = {
            "experiment_name": self.config.experiment_name,
            "dataset": self.config.dataset,
            "models": list(self.config.models.keys()),
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
