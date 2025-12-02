"""Unit tests for LightGBM model wrappers."""

import unittest

import pandas as pd
import numpy as np

# Import to trigger registration
from benchmark.models.lightgbm_wrapper import LightGBMClassifier, LightGBMRegressor
from benchmark.registry import REGISTRY


class TestLightGBMModelRegistry(unittest.TestCase):
    """Tests that LightGBM models are properly registered."""

    def test_lightgbm_classifier_registered(self):
        self.assertTrue(REGISTRY.is_registered("lightgbm_classifier"))
        meta = REGISTRY.get("lightgbm_classifier")
        self.assertEqual(meta["type"], "classification")

    def test_lightgbm_regressor_registered(self):
        self.assertTrue(REGISTRY.is_registered("lightgbm_regressor"))
        meta = REGISTRY.get("lightgbm_regressor")
        self.assertEqual(meta["type"], "regression")


class TestLightGBMClassifierWrapper(unittest.TestCase):
    """Tests for LightGBMClassifier wrapper."""

    def _make_data(self):
        X = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1, 0]})
        y = pd.Series([0, 0, 0, 1, 1, 1])
        return X, y

    def test_train_predict(self):
        X, y = self._make_data()
        model = LightGBMClassifier(n_estimators=10, random_state=42, verbosity=-1)
        model.train(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), len(X))
        self.assertEqual(set(preds).issubset({0, 1}), True)

    def test_predict_proba(self):
        X, y = self._make_data()
        model = LightGBMClassifier(n_estimators=10, random_state=42, verbosity=-1)
        model.train(X, y)
        proba = model.predict_proba(X)
        self.assertEqual(proba.shape, (len(X), 2))
        self.assertTrue(((proba >= 0) & (proba <= 1)).all())


class TestLightGBMRegressorWrapper(unittest.TestCase):
    """Tests for LightGBMRegressor wrapper."""

    def _make_data(self):
        X = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1, 0]})
        y = pd.Series([0, 1, 2, 3, 4, 5])
        return X, y

    def test_train_predict(self):
        X, y = self._make_data()
        model = LightGBMRegressor(n_estimators=10, random_state=42, verbosity=-1)
        model.train(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), len(X))
        self.assertTrue(np.issubdtype(type(preds[0]), np.floating) or isinstance(preds[0], (float, int)))

    def test_predict_proba_returns_none(self):
        X, y = self._make_data()
        model = LightGBMRegressor(n_estimators=10, random_state=42, verbosity=-1)
        model.train(X, y)
        self.assertIsNone(model.predict_proba(X))


class TestRegistryBuildLightGBM(unittest.TestCase):
    """Tests building LightGBM models via the registry with hyperparameter overrides."""

    def test_build_lightgbm_classifier(self):
        model = REGISTRY.build("lightgbm_classifier", {"n_estimators": 50})
        self.assertIsInstance(model, LightGBMClassifier)
        self.assertEqual(model.model.n_estimators, 50)

    def test_build_lightgbm_regressor(self):
        model = REGISTRY.build("lightgbm_regressor", {"n_estimators": 50})
        self.assertIsInstance(model, LightGBMRegressor)
        self.assertEqual(model.model.n_estimators, 50)


if __name__ == "__main__":
    unittest.main()
