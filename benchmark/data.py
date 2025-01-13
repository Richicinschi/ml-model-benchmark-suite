"""Dataset loaders for CSV, sklearn built-ins, and OpenML."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits, fetch_openml

from .base import DatasetLoader
from .utils import setup_logger


class CSVDatasetLoader(DatasetLoader):
    """Load datasets from local CSV files."""

    def __init__(self, path: str, target_column: str, **read_options: Any):
        self.path = Path(path)
        self.target_column = target_column
        self.read_options = read_options
        self.logger = setup_logger("CSVDatasetLoader")

    def load(self, **kwargs: Any) -> Tuple[pd.DataFrame, pd.Series]:
        if not self.path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.path}")

        options = dict(self.read_options)
        options.update(kwargs)
        df = pd.read_csv(self.path, **options)

        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in CSV")

        y = df[self.target_column]
        X = df.drop(columns=[self.target_column])
        self.logger.info(f"Loaded CSV: {self.path} with shape {X.shape}")
        return X, y


class SklearnDatasetLoader(DatasetLoader):
    """Load built-in datasets from scikit-learn."""

    SUPPORTED = {"iris", "wine", "breast_cancer", "digits"}

    def __init__(self, name: str, as_frame: bool = True):
        self.name = name.lower()
        self.as_frame = as_frame
        self.logger = setup_logger("SklearnDatasetLoader")

    def load(self, **kwargs: Any) -> Tuple[pd.DataFrame, pd.Series]:
        if self.name not in self.SUPPORTED:
            raise ValueError(
                f"Unsupported sklearn dataset: {self.name}. "
                f"Supported: {self.SUPPORTED}"
            )

        loaders = {
            "iris": load_iris,
            "wine": load_wine,
            "breast_cancer": load_breast_cancer,
            "digits": load_digits,
        }

        data = loaders[self.name](as_frame=self.as_frame)
        X = data.data
        y = data.target
        self.logger.info(f"Loaded sklearn dataset: {self.name} with shape {X.shape}")
        return X, y


def load_dataset(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
    """Factory function to load a dataset based on config dictionary."""
    source = config.get("source", "csv")

    if source == "csv":
        loader = CSVDatasetLoader(
            path=config["path"],
            target_column=config["target_column"],
            **config.get("read_options", {}),
        )
    elif source == "sklearn":
        loader = SklearnDatasetLoader(
            name=config["name"],
            as_frame=config.get("as_frame", True),
        )
    else:
        raise ValueError(f"Unsupported dataset source: {source}")

    return loader.load()
