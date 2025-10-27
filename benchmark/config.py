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
    """Raised when a configuration is invalid or cannot be loaded."""
    pass


class ExperimentConfig:
    """Loads and validates experiment configuration files.

    Supports YAML (``.yaml``, ``.yml``) and JSON (``.json``) formats.
    The configuration must contain at minimum ``experiment_name``,
    ``dataset``, and ``models`` keys.

    Args:
        config_path: Path to the configuration file.

    Raises:
        ConfigError: If the file is missing, the format is unsupported,
            or required keys are absent.
    """

    REQUIRED_KEYS = {"experiment_name", "dataset", "models"}

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.raw = self._load()
        self._validate()

    def _load(self) -> Dict[str, Any]:
        """Load the raw configuration from disk.

        Returns:
            The parsed configuration dictionary.

        Raises:
            ConfigError: If the file does not exist, PyYAML is not
                installed for YAML files, or the file extension is
                unsupported.
        """
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
        """Validate that required keys are present and well-formed.

        Raises:
            ConfigError: If required keys are missing or ``models`` has
                an invalid type.
        """
        missing = self.REQUIRED_KEYS - set(self.raw.keys())
        if missing:
            raise ConfigError(f"Missing required config keys: {missing}")

        if not isinstance(self.raw.get("models"), (list, dict)):
            raise ConfigError("'models' must be a list or dictionary")

    @property
    def experiment_name(self) -> str:
        """Name of the experiment."""
        return self.raw["experiment_name"]

    @property
    def dataset(self) -> Dict[str, Any]:
        """Dataset loading configuration."""
        return self.raw["dataset"]

    @property
    def models(self) -> Dict[str, Any]:
        """Model configuration as a dictionary mapping model names to
        hyperparameter overrides.

        If the raw config specifies a list, it is normalized to a
        dictionary with empty override dictionaries.
        """
        models = self.raw["models"]
        if isinstance(models, list):
            return {m: {} for m in models}
        return models

    @property
    def preprocessing(self) -> Dict[str, Any]:
        """Preprocessing pipeline configuration."""
        return self.raw.get("preprocessing", {})

    @property
    def cv(self) -> Dict[str, Any]:
        """Cross-validation strategy configuration."""
        return self.raw.get("cv", {"strategy": "stratified_kfold", "folds": 5})

    @property
    def metrics(self) -> Dict[str, Any]:
        """Metrics computation configuration."""
        return self.raw.get("metrics", {})

    @property
    def tuning(self) -> Dict[str, Any]:
        """Hyperparameter tuning configuration."""
        return self.raw.get("tuning", {})

    @property
    def tags(self) -> list[str]:
        """Experiment tags as a list of strings.

        Supports both list and comma-separated string inputs.
        """
        tags = self.raw.get("tags", [])
        if isinstance(tags, str):
            return [t.strip() for t in tags.split(",") if t.strip()]
        return list(tags)

    @property
    def notes(self) -> str:
        """Free-form notes for the experiment."""
        return self.raw.get("notes", "")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the resolved configuration to a dictionary.

        Returns:
            Dictionary containing all normalized configuration values.
        """
        return {
            "experiment_name": self.experiment_name,
            "dataset": self.dataset,
            "models": self.models,
            "preprocessing": self.preprocessing,
            "cv": self.cv,
            "metrics": self.metrics,
            "tuning": self.tuning,
        }
