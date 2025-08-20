"""Configuration loading and validation for benchmark experiments."""

import json
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class ConfigError(Exception):
    """Raised when a configuration is invalid."""
    pass


class ExperimentConfig:
    """Loads and validates experiment configuration files."""

    REQUIRED_KEYS = {"experiment_name", "dataset", "models"}

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.raw = self._load()
        self._validate()

    def _load(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise ConfigError(f"Config file not found: {self.config_path}")

        suffix = self.config_path.suffix.lower()
        with open(self.config_path, "r", encoding="utf-8") as f:
            if suffix in (".yaml", ".yml"):
                if not HAS_YAML:
                    raise ConfigError("PyYAML is required to load YAML config files")
                return yaml.safe_load(f)
            elif suffix == ".json":
                return json.load(f)
            else:
                raise ConfigError(f"Unsupported config format: {suffix}")

    def _validate(self) -> None:
        missing = self.REQUIRED_KEYS - set(self.raw.keys())
        if missing:
            raise ConfigError(f"Missing required config keys: {missing}")

        if not isinstance(self.raw.get("models"), (list, dict)):
            raise ConfigError("'models' must be a list or dictionary")

    @property
    def experiment_name(self) -> str:
        return self.raw["experiment_name"]

    @property
    def dataset(self) -> Dict[str, Any]:
        return self.raw["dataset"]

    @property
    def models(self) -> Dict[str, Any]:
        models = self.raw["models"]
        if isinstance(models, list):
            return {m: {} for m in models}
        return models

    @property
    def preprocessing(self) -> Dict[str, Any]:
        return self.raw.get("preprocessing", {})

    @property
    def cv(self) -> Dict[str, Any]:
        return self.raw.get("cv", {"strategy": "stratified_kfold", "folds": 5})

    @property
    def metrics(self) -> Dict[str, Any]:
        return self.raw.get("metrics", {})

    @property
    def tuning(self) -> Dict[str, Any]:
        return self.raw.get("tuning", {})

    @property
    def tags(self) -> list[str]:
        tags = self.raw.get("tags", [])
        if isinstance(tags, str):
            return [t.strip() for t in tags.split(",") if t.strip()]
        return list(tags)

    @property
    def notes(self) -> str:
        return self.raw.get("notes", "")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "dataset": self.dataset,
            "models": self.models,
            "preprocessing": self.preprocessing,
            "cv": self.cv,
            "metrics": self.metrics,
            "tuning": self.tuning,
        }
