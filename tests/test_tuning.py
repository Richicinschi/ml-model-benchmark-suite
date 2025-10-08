"""Unit tests for hyperparameter tuning module."""

import unittest

import pandas as pd
import numpy as np

# Import to trigger registration
import benchmark.models.classification  # noqa: F401
from benchmark.tuning import TuningConfig, run_grid_search, run_randomized_search, run_tuning


class TestTuningConfig(unittest.TestCase):
    """Tests for TuningConfig."""

    def test_defaults(self):
        config = TuningConfig()
        self.assertEqual(config.method, "grid")
        self.assertEqual(config.scoring, "accuracy")
        self.assertEqual(config.cv_folds, 3)
        self.assertEqual(config.n_iter, 10)
        self.assertEqual(config.n_jobs, -1)
        self.assertEqual(config.verbose, 0)
        self.assertEqual(config.random_state, 42)
        self.assertTrue(config.refit)
        self.assertFalse(config.is_enabled())

    def test_custom_values(self):
        config = TuningConfig({
            "enabled": True,
            "method": "randomized",
            "scoring": "f1",
            "cv_folds": 5,
            "n_iter": 20,
            "n_jobs": 2,
            "verbose": 1,
            "random_state": 123,
            "refit": False,
        })
        self.assertTrue(config.is_enabled())
        self.assertEqual(config.method, "randomized")
        self.assertEqual(config.scoring, "f1")
        self.assertEqual(config.cv_folds, 5)
        self.assertEqual(config.n_iter, 20)
        self.assertEqual(config.n_jobs, 2)
        self.assertEqual(config.verbose, 1)
        self.assertEqual(config.random_state, 123)
        self.assertFalse(config.refit)


class TestRunGridSearch(unittest.TestCase):
    """Tests for run_grid_search."""

    def _make_data(self):
        X = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1, 0]})
        y = pd.Series([0, 0, 0, 1, 1, 1])
        return X, y

    def test_grid_search_returns_expected_keys(self):
        X, y = self._make_data()
        result = run_grid_search(
            model_name="logistic_regression",
            param_grid={"C": [0.1, 1.0]},
            X=X,
            y=y,
            scoring="accuracy",
            cv=2,
            n_jobs=1,
            verbose=0,
            refit=True,
        )
        self.assertEqual(result["method"], "grid")
        self.assertIn("best_params", result)
        self.assertIn("best_score", result)
        self.assertIn("cv_results", result)
        self.assertIn("C", result["best_params"])


class TestRunRandomizedSearch(unittest.TestCase):
    """Tests for run_randomized_search."""

    def _make_data(self):
        X = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1, 0]})
        y = pd.Series([0, 0, 0, 1, 1, 1])
        return X, y

    def test_randomized_search_returns_expected_keys(self):
        X, y = self._make_data()
        result = run_randomized_search(
            model_name="logistic_regression",
            param_distributions={"C": [0.1, 1.0, 10.0]},
            X=X,
            y=y,
            n_iter=2,
            scoring="accuracy",
            cv=2,
            n_jobs=1,
            verbose=0,
            random_state=42,
            refit=True,
        )
        self.assertEqual(result["method"], "randomized")
        self.assertIn("best_params", result)
        self.assertIn("best_score", result)
        self.assertIn("cv_results", result)
        self.assertIn("C", result["best_params"])


class TestRunTuningDispatcher(unittest.TestCase):
    """Tests for run_tuning dispatcher."""

    def _make_data(self):
        X = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1, 0]})
        y = pd.Series([0, 0, 0, 1, 1, 1])
        return X, y

    def test_dispatch_grid(self):
        X, y = self._make_data()
        config = TuningConfig({"method": "grid", "cv_folds": 2, "n_jobs": 1})
        result = run_tuning(
            model_name="logistic_regression",
            tuning_config=config,
            param_space={"C": [0.1, 1.0]},
            X=X,
            y=y,
        )
        self.assertEqual(result["method"], "grid")
        self.assertIn("best_params", result)

    def test_dispatch_randomized(self):
        X, y = self._make_data()
        config = TuningConfig({"method": "randomized", "cv_folds": 2, "n_iter": 2, "n_jobs": 1})
        result = run_tuning(
            model_name="logistic_regression",
            tuning_config=config,
            param_space={"C": [0.1, 1.0, 10.0]},
            X=X,
            y=y,
        )
        self.assertEqual(result["method"], "randomized")
        self.assertIn("best_params", result)

    def test_dispatch_unsupported_method_raises(self):
        X, y = self._make_data()
        config = TuningConfig({"method": "bayesian", "cv_folds": 2})
        with self.assertRaises(ValueError) as ctx:
            run_tuning(
                model_name="logistic_regression",
                tuning_config=config,
                param_space={"C": [0.1, 1.0]},
                X=X,
                y=y,
            )
        self.assertIn("Unsupported tuning method", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
