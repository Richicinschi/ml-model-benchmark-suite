"""Unit tests for classification model wrappers."""

import unittest

import pandas as pd

# Import to trigger registration
from benchmark.models.classification import (
    LogisticRegression,
    KNeighborsClassifier,
    RandomForestClassifier,
    SVM,
)
from benchmark.registry import REGISTRY


class TestClassificationModelRegistry(unittest.TestCase):
    """Tests that classification models are properly registered."""

    def test_logistic_regression_registered(self):
        self.assertTrue(REGISTRY.is_registered("logistic_regression"))
        meta = REGISTRY.get("logistic_regression")
        self.assertEqual(meta["type"], "classification")

    def test_knn_registered(self):
        self.assertTrue(REGISTRY.is_registered("knn"))
        meta = REGISTRY.get("knn")
        self.assertEqual(meta["type"], "classification")

    def test_random_forest_registered(self):
        self.assertTrue(REGISTRY.is_registered("random_forest"))
        meta = REGISTRY.get("random_forest")
        self.assertEqual(meta["type"], "classification")

    def test_svm_registered(self):
        self.assertTrue(REGISTRY.is_registered("svm"))
        meta = REGISTRY.get("svm")
        self.assertEqual(meta["type"], "classification")


class TestLogisticRegressionWrapper(unittest.TestCase):
    """Tests for LogisticRegression wrapper."""

    def _make_data(self):
        X = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1, 0]})
        y = pd.Series([0, 0, 1, 1, 1, 1])
        return X, y

    def test_train_predict(self):
        X, y = self._make_data()
        model = LogisticRegression(max_iter=1000)
        model.train(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), len(X))
        self.assertEqual(set(preds).issubset({0, 1}), True)

    def test_predict_proba(self):
        X, y = self._make_data()
        model = LogisticRegression(max_iter=1000)
        model.train(X, y)
        proba = model.predict_proba(X)
        self.assertEqual(proba.shape, (len(X), 2))
        self.assertTrue(((proba >= 0) & (proba <= 1)).all())


class TestKNeighborsClassifierWrapper(unittest.TestCase):
    """Tests for KNeighborsClassifier wrapper."""

    def _make_data(self):
        X = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1, 0]})
        y = pd.Series([0, 0, 0, 1, 1, 1])
        return X, y

    def test_train_predict(self):
        X, y = self._make_data()
        model = KNeighborsClassifier(n_neighbors=2)
        model.train(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), len(X))

    def test_predict_proba(self):
        X, y = self._make_data()
        model = KNeighborsClassifier(n_neighbors=2)
        model.train(X, y)
        proba = model.predict_proba(X)
        self.assertEqual(proba.shape[0], len(X))
        self.assertTrue(((proba >= 0) & (proba <= 1)).all())


class TestRandomForestClassifierWrapper(unittest.TestCase):
    """Tests for RandomForestClassifier wrapper."""

    def _make_data(self):
        X = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1, 0]})
        y = pd.Series([0, 0, 0, 1, 1, 1])
        return X, y

    def test_train_predict(self):
        X, y = self._make_data()
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.train(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), len(X))

    def test_predict_proba(self):
        X, y = self._make_data()
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.train(X, y)
        proba = model.predict_proba(X)
        self.assertEqual(proba.shape, (len(X), 2))
        self.assertTrue(((proba >= 0) & (proba <= 1)).all())


class TestSVMWrapper(unittest.TestCase):
    """Tests for SVM wrapper."""

    def _make_data(self):
        X = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1, 0]})
        y = pd.Series([0, 0, 0, 1, 1, 1])
        return X, y

    def test_train_predict(self):
        X, y = self._make_data()
        model = SVM(probability=True, random_state=42)
        model.train(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), len(X))

    def test_predict_proba(self):
        X, y = self._make_data()
        model = SVM(probability=True, random_state=42)
        model.train(X, y)
        proba = model.predict_proba(X)
        self.assertEqual(proba.shape, (len(X), 2))
        self.assertTrue(((proba >= 0) & (proba <= 1)).all())


class TestRegistryBuild(unittest.TestCase):
    """Tests building classification models via the registry."""

    def test_build_logistic_regression(self):
        model = REGISTRY.build("logistic_regression", {"C": 0.5})
        self.assertIsInstance(model, LogisticRegression)
        self.assertEqual(model.model.C, 0.5)

    def test_build_random_forest(self):
        model = REGISTRY.build("random_forest", {"n_estimators": 50})
        self.assertIsInstance(model, RandomForestClassifier)
        self.assertEqual(model.model.n_estimators, 50)


if __name__ == "__main__":
    unittest.main()
