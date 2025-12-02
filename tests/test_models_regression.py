"""Unit tests for regression model wrappers."""

import unittest

import pandas as pd
import numpy as np

# Import to trigger registration
from benchmark.models.regression import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    GradientBoostingRegressor,
)
from benchmark.registry import REGISTRY

try:
    from benchmark.models.regression import XGBoostRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    XGBoostRegressor = None  # type: ignore


class TestRegressionModelRegistry(unittest.TestCase):
    """Tests that regression models are properly registered."""

    def test_linear_regression_registered(self):
        self.assertTrue(REGISTRY.is_registered("linear_regression"))
        meta = REGISTRY.get("linear_regression")
        self.assertEqual(meta["type"], "regression")

    def test_ridge_registered(self):
        self.assertTrue(REGISTRY.is_registered("ridge"))
        meta = REGISTRY.get("ridge")
        self.assertEqual(meta["type"], "regression")

    def test_lasso_registered(self):
        self.assertTrue(REGISTRY.is_registered("lasso"))
        meta = REGISTRY.get("lasso")
        self.assertEqual(meta["type"], "regression")

    def test_elasticnet_registered(self):
        self.assertTrue(REGISTRY.is_registered("elasticnet"))
        meta = REGISTRY.get("elasticnet")
        self.assertEqual(meta["type"], "regression")

    def test_gradient_boosting_registered(self):
        self.assertTrue(REGISTRY.is_registered("gradient_boosting"))
        meta = REGISTRY.get("gradient_boosting")
        self.assertEqual(meta["type"], "regression")

    @unittest.skipUnless(HAS_XGBOOST, "xgboost not installed")
    def test_xgboost_regressor_registered(self):
        self.assertTrue(REGISTRY.is_registered("xgboost_regressor"))
        meta = REGISTRY.get("xgboost_regressor")
        self.assertEqual(meta["type"], "regression")


class TestLinearRegressionWrapper(unittest.TestCase):
    """Tests for LinearRegression wrapper."""

    def _make_data(self):
        X = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1, 0]})
        y = pd.Series([0, 1, 2, 3, 4, 5])
        return X, y

    def test_train_predict(self):
        X, y = self._make_data()
        model = LinearRegression()
        model.train(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), len(X))
        self.assertTrue(np.issubdtype(type(preds[0]), np.floating) or isinstance(preds[0], (float, int)))

    def test_predict_proba_returns_none(self):
        X, y = self._make_data()
        model = LinearRegression()
        model.train(X, y)
        self.assertIsNone(model.predict_proba(X))


class TestRidgeWrapper(unittest.TestCase):
    """Tests for Ridge wrapper."""

    def _make_data(self):
        X = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1, 0]})
        y = pd.Series([0, 1, 2, 3, 4, 5])
        return X, y

    def test_train_predict(self):
        X, y = self._make_data()
        model = Ridge(alpha=1.0)
        model.train(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), len(X))

    def test_predict_proba_returns_none(self):
        X, y = self._make_data()
        model = Ridge(alpha=1.0)
        model.train(X, y)
        self.assertIsNone(model.predict_proba(X))


class TestLassoWrapper(unittest.TestCase):
    """Tests for Lasso wrapper."""

    def _make_data(self):
        X = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1, 0]})
        y = pd.Series([0, 1, 2, 3, 4, 5])
        return X, y

    def test_train_predict(self):
        X, y = self._make_data()
        model = Lasso(alpha=0.1, max_iter=1000)
        model.train(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), len(X))

    def test_predict_proba_returns_none(self):
        X, y = self._make_data()
        model = Lasso(alpha=0.1, max_iter=1000)
        model.train(X, y)
        self.assertIsNone(model.predict_proba(X))


class TestElasticNetWrapper(unittest.TestCase):
    """Tests for ElasticNet wrapper."""

    def _make_data(self):
        X = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1, 0]})
        y = pd.Series([0, 1, 2, 3, 4, 5])
        return X, y

    def test_train_predict(self):
        X, y = self._make_data()
        model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000)
        model.train(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), len(X))

    def test_predict_proba_returns_none(self):
        X, y = self._make_data()
        model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000)
        model.train(X, y)
        self.assertIsNone(model.predict_proba(X))


class TestGradientBoostingRegressorWrapper(unittest.TestCase):
    """Tests for GradientBoostingRegressor wrapper."""

    def _make_data(self):
        X = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1, 0]})
        y = pd.Series([0, 1, 2, 3, 4, 5])
        return X, y

    def test_train_predict(self):
        X, y = self._make_data()
        model = GradientBoostingRegressor(n_estimators=10, random_state=42)
        model.train(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), len(X))

    def test_predict_proba_returns_none(self):
        X, y = self._make_data()
        model = GradientBoostingRegressor(n_estimators=10, random_state=42)
        model.train(X, y)
        self.assertIsNone(model.predict_proba(X))


class TestRegistryBuildRegression(unittest.TestCase):
    """Tests building regression models via the registry."""

    def test_build_linear_regression(self):
        model = REGISTRY.build("linear_regression", {})
        self.assertIsInstance(model, LinearRegression)

    def test_build_ridge(self):
        model = REGISTRY.build("ridge", {"alpha": 2.0})
        self.assertIsInstance(model, Ridge)
        self.assertEqual(model.model.alpha, 2.0)

    def test_build_lasso(self):
        model = REGISTRY.build("lasso", {"alpha": 0.5})
        self.assertIsInstance(model, Lasso)
        self.assertEqual(model.model.alpha, 0.5)

    def test_build_elasticnet(self):
        model = REGISTRY.build("elasticnet", {"alpha": 0.2, "l1_ratio": 0.3})
        self.assertIsInstance(model, ElasticNet)
        self.assertEqual(model.model.alpha, 0.2)
        self.assertEqual(model.model.l1_ratio, 0.3)

    def test_build_gradient_boosting(self):
        model = REGISTRY.build("gradient_boosting", {"n_estimators": 50})
        self.assertIsInstance(model, GradientBoostingRegressor)
        self.assertEqual(model.model.n_estimators, 50)

    @unittest.skipUnless(HAS_XGBOOST, "xgboost not installed")
    def test_build_xgboost_regressor(self):
        model = REGISTRY.build("xgboost_regressor", {"n_estimators": 50})
        self.assertIsInstance(model, XGBoostRegressor)
        self.assertEqual(model.model.n_estimators, 50)


@unittest.skipUnless(HAS_XGBOOST, "xgboost not installed")
class TestXGBoostRegressorWrapper(unittest.TestCase):
    """Tests for XGBoostRegressor wrapper."""

    def _make_data(self):
        X = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1, 0]})
        y = pd.Series([0, 1, 2, 3, 4, 5])
        return X, y

    def test_train_predict(self):
        X, y = self._make_data()
        model = XGBoostRegressor(n_estimators=10, random_state=42)
        model.train(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), len(X))

    def test_predict_proba_returns_none(self):
        X, y = self._make_data()
        model = XGBoostRegressor(n_estimators=10, random_state=42)
        model.train(X, y)
        self.assertIsNone(model.predict_proba(X))


if __name__ == "__main__":
    unittest.main()
