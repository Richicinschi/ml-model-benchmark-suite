"""Hyperparameter tuning with GridSearchCV and RandomizedSearchCV."""

from typing import Any, Dict, Optional

import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from .registry import REGISTRY
from .utils import setup_logger


class TuningConfig:
    """Configuration wrapper for hyperparameter tuning."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.method = self.config.get("method", "grid")
        self.scoring = self.config.get("scoring", "accuracy")
        self.cv_folds = self.config.get("cv_folds", 3)
        self.n_iter = self.config.get("n_iter", 10)
        self.n_jobs = self.config.get("n_jobs", -1)
        self.verbose = self.config.get("verbose", 0)
        self.random_state = self.config.get("random_state", 42)
        self.refit = self.config.get("refit", True)

    def is_enabled(self) -> bool:
        return self.config.get("enabled", False)


def run_grid_search(
    model_name: str,
    param_grid: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str = "accuracy",
    cv: int = 3,
    n_jobs: int = -1,
    verbose: int = 0,
    refit: bool = True,
) -> Dict[str, Any]:
    """Run GridSearchCV for a registered model."""
    logger = setup_logger("Tuning")
    logger.info(f"Starting GridSearchCV for {model_name}")

    base_model = REGISTRY.build(model_name)
    search = GridSearchCV(
        estimator=base_model.model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=refit,
        return_train_score=True,
    )
    search.fit(X, y)

    logger.info(f"GridSearchCV complete. Best score: {search.best_score_:.4f}")
    return {
        "method": "grid",
        "best_params": search.best_params_,
        "best_score": search.best_score_,
        "cv_results": {
            k: v.tolist() if hasattr(v, "tolist") else v
            for k, v in search.cv_results_.items()
        },
    }


def run_randomized_search(
    model_name: str,
    param_distributions: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 10,
    scoring: str = "accuracy",
    cv: int = 3,
    n_jobs: int = -1,
    verbose: int = 0,
    random_state: int = 42,
    refit: bool = True,
) -> Dict[str, Any]:
    """Run RandomizedSearchCV for a registered model."""
    logger = setup_logger("Tuning")
    logger.info(f"Starting RandomizedSearchCV for {model_name}")

    base_model = REGISTRY.build(model_name)
    search = RandomizedSearchCV(
        estimator=base_model.model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=random_state,
        refit=refit,
        return_train_score=True,
    )
    search.fit(X, y)

    logger.info(f"RandomizedSearchCV complete. Best score: {search.best_score_:.4f}")
    return {
        "method": "randomized",
        "best_params": search.best_params_,
        "best_score": search.best_score_,
        "cv_results": {
            k: v.tolist() if hasattr(v, "tolist") else v
            for k, v in search.cv_results_.items()
        },
    }


def run_tuning(
    model_name: str,
    tuning_config: TuningConfig,
    param_space: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
) -> Dict[str, Any]:
    """Dispatch to GridSearchCV or RandomizedSearchCV based on config."""
    if tuning_config.method == "grid":
        return run_grid_search(
            model_name=model_name,
            param_grid=param_space,
            X=X,
            y=y,
            scoring=tuning_config.scoring,
            cv=tuning_config.cv_folds,
            n_jobs=tuning_config.n_jobs,
            verbose=tuning_config.verbose,
            refit=tuning_config.refit,
        )
    elif tuning_config.method == "randomized":
        return run_randomized_search(
            model_name=model_name,
            param_distributions=param_space,
            X=X,
            y=y,
            n_iter=tuning_config.n_iter,
            scoring=tuning_config.scoring,
            cv=tuning_config.cv_folds,
            n_jobs=tuning_config.n_jobs,
            verbose=tuning_config.verbose,
            random_state=tuning_config.random_state,
            refit=tuning_config.refit,
        )
    else:
        raise ValueError(f"Unsupported tuning method: {tuning_config.method}")
