"""Experiment runner that orchestrates benchmark experiments from config."""

from typing import Any, Dict, List, Tuple

import pandas as pd

from .base import Experiment
from .config import ExperimentConfig
from .cv import build_cv_strategy, get_cv_splits
from .data import load_dataset
# Import models to ensure they register themselves in the global registry
from . import models  # noqa: F401
from .importance import extract_feature_importance
from .metrics import compute_metrics
from .shap_analysis import compute_shap_values
from .tracking import ExperimentTracker
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

    def _run_cv(
        self,
        model_name: str,
        model_wrapper: Any,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, Any]:
        """Run cross-validation for a single model and return fold results."""
        cv = build_cv_strategy(self.config.cv, X, y)
        splits = get_cv_splits(cv, X, y)
        task_type = REGISTRY.get(model_name)["type"]

        fold_results = []
        final_model = None
        for fold_idx, (train_idx, val_idx) in enumerate(splits, 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Clone a fresh model instance per fold
            fold_model = REGISTRY.build(model_name)
            fold_model.train(X_train, y_train)
            if fold_idx == len(splits):
                final_model = fold_model

            train_preds = fold_model.predict(X_train)
            val_preds = fold_model.predict(X_val)

            fold_result = {
                "fold": fold_idx,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
            }

            # Metrics
            train_metrics = compute_metrics(
                y_train, train_preds, task_type, y_proba=None
            )
            val_proba = None
            try:
                val_proba = fold_model.predict_proba(X_val)
            except Exception:
                pass
            val_metrics = compute_metrics(y_val, val_preds, task_type, y_proba=val_proba)
            fold_result["train_metrics"] = train_metrics
            fold_result["val_metrics"] = val_metrics
            if val_proba is not None:
                fold_result["val_proba"] = val_proba.tolist() if hasattr(val_proba, "tolist") else val_proba
            fold_result["val_true"] = y_val.tolist() if hasattr(y_val, "tolist") else list(y_val)

            fold_results.append(fold_result)
            self.logger.info(f"{model_name} - fold {fold_idx}/{len(splits)} complete")

        # Aggregate metrics across folds
        aggregated = self._aggregate_fold_metrics(fold_results)

        # Feature importance from final fold model
        feature_importance = None
        if final_model is not None:
            feature_importance = extract_feature_importance(
                model_name, final_model, feature_names=list(X.columns)
            )

        # SHAP analysis from final fold model
        shap_result = None
        if final_model is not None:
            shap_result = compute_shap_values(final_model, X)

        return {
            "model": model_name,
            "task_type": task_type,
            "folds": fold_results,
            "n_folds": len(splits),
            "aggregated": aggregated,
            "feature_importance": feature_importance,
            "shap": shap_result,
        }

    def _aggregate_fold_metrics(
        self, fold_results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """Compute mean and std of scalar metrics across CV folds."""
        if not fold_results:
            return {}

        def _is_scalar(value: Any) -> bool:
            return isinstance(value, (int, float)) and not isinstance(value, bool)

        train_keys = list(fold_results[0]["train_metrics"].keys())
        val_keys = list(fold_results[0]["val_metrics"].keys())

        aggregated: Dict[str, Dict[str, float]] = {"train": {}, "val": {}}

        for key in train_keys:
            values = [f["train_metrics"][key] for f in fold_results if _is_scalar(f["train_metrics"][key])]
            if values:
                aggregated["train"][f"{key}_mean"] = float(pd.Series(values).mean())
                aggregated["train"][f"{key}_std"] = float(pd.Series(values).std())

        for key in val_keys:
            values = [f["val_metrics"][key] for f in fold_results if _is_scalar(f["val_metrics"][key])]
            if values:
                aggregated["val"][f"{key}_mean"] = float(pd.Series(values).mean())
                aggregated["val"][f"{key}_std"] = float(pd.Series(values).std())

        return aggregated

    def run(self) -> Dict[str, Any]:
        """Execute the experiment defined in the configuration."""
        self.logger.info(f"Starting experiment: {self.config.experiment_name}")
        self.logger.info(f"Dataset config: {self.config.dataset}")
        self.logger.info(f"Models: {list(self.config.models.keys())}")

        self.load_data()
        X_processed, y_processed = self.preprocess_data()
        model_instances = self.resolve_models()

        model_results = {}
        for name in self.config.models.keys():
            model_results[name] = self._run_cv(name, model_instances[name], X_processed, y_processed)

        results = {
            "experiment_name": self.config.experiment_name,
            "dataset": self.config.dataset,
            "models": list(self.config.models.keys()),
            "model_types": {
                name: REGISTRY.get(name)["type"]
                for name in self.config.models.keys()
            },
            "preprocessing": self.config.preprocessing,
            "cv": self.config.cv,
            "data_shape": {
                "raw": {"X": self.X.shape, "y": self.y.shape},
                "processed": {"X": X_processed.shape, "y": y_processed.shape},
            },
            "status": "completed",
            "results": model_results,
        }

        tracker = ExperimentTracker()
        run_id = tracker.save_run(results)
        results["run_id"] = run_id

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
