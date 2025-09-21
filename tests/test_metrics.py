"""Unit tests for the metrics computation module."""

import unittest

import numpy as np

from benchmark.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_metrics,
    compute_regression_metrics,
)


class TestClassificationMetrics(unittest.TestCase):
    """Tests for compute_classification_metrics."""

    def test_perfect_classification(self):
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 2]
        result = compute_classification_metrics(y_true, y_pred)
        self.assertAlmostEqual(result["accuracy"], 1.0)
        self.assertAlmostEqual(result["precision"], 1.0)
        self.assertAlmostEqual(result["recall"], 1.0)
        self.assertAlmostEqual(result["f1"], 1.0)
        self.assertNotIn("roc_auc", result)

    def test_with_proba_binary(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 1]
        y_proba = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.3, 0.7]])
        result = compute_classification_metrics(y_true, y_pred, y_proba)
        self.assertIn("roc_auc", result)
        self.assertIsInstance(result["roc_auc"], float)
        self.assertGreaterEqual(result["roc_auc"], 0.0)
        self.assertLessEqual(result["roc_auc"], 1.0)

    def test_with_proba_multiclass(self):
        y_true = [0, 1, 2, 0]
        y_pred = [0, 1, 2, 0]
        y_proba = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
        ])
        result = compute_classification_metrics(y_true, y_pred, y_proba)
        self.assertIn("roc_auc", result)
        self.assertIsInstance(result["roc_auc"], float)

    def test_roc_auc_graceful_failure(self):
        y_true = [0, 0, 0, 0]
        y_pred = [0, 0, 0, 0]
        y_proba = np.array([[0.9, 0.1]] * 4)
        result = compute_classification_metrics(y_true, y_pred, y_proba)
        self.assertIn("roc_auc", result)
        self.assertTrue(
            result["roc_auc"] is None or (isinstance(result["roc_auc"], float) and np.isnan(result["roc_auc"]))
        )


class TestRegressionMetrics(unittest.TestCase):
    """Tests for compute_regression_metrics."""

    def test_perfect_regression(self):
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [1.0, 2.0, 3.0, 4.0]
        result = compute_regression_metrics(y_true, y_pred)
        self.assertAlmostEqual(result["mse"], 0.0)
        self.assertAlmostEqual(result["rmse"], 0.0)
        self.assertAlmostEqual(result["mae"], 0.0)
        self.assertAlmostEqual(result["r2"], 1.0)

    def test_imperfect_regression(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.5, 2.5, 2.5]
        result = compute_regression_metrics(y_true, y_pred)
        self.assertGreater(result["mse"], 0.0)
        self.assertGreater(result["rmse"], 0.0)
        self.assertGreater(result["mae"], 0.0)
        self.assertLess(result["r2"], 1.0)

    def test_rmse_is_sqrt_mse(self):
        y_true = [0.0, 0.0]
        y_pred = [3.0, 4.0]
        result = compute_regression_metrics(y_true, y_pred)
        self.assertAlmostEqual(result["rmse"], np.sqrt(result["mse"]), places=5)


class TestConfusionMatrix(unittest.TestCase):
    """Tests for compute_confusion_matrix."""

    def test_simple_confusion_matrix(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 1]
        cm = compute_confusion_matrix(y_true, y_pred)
        self.assertEqual(cm, [[1, 1], [0, 2]])

    def test_perfect_confusion_matrix(self):
        y_true = [0, 1, 2]
        y_pred = [0, 1, 2]
        cm = compute_confusion_matrix(y_true, y_pred)
        self.assertEqual(cm, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])


class TestMetricsDispatcher(unittest.TestCase):
    """Tests for compute_metrics task-type dispatcher."""

    def test_dispatch_classification(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 1]
        result = compute_metrics(y_true, y_pred, "classification")
        self.assertIn("accuracy", result)
        self.assertIn("confusion_matrix", result)

    def test_dispatch_regression(self):
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 2.0, 3.0]
        result = compute_metrics(y_true, y_pred, "regression")
        self.assertIn("mse", result)
        self.assertIn("r2", result)
        self.assertNotIn("confusion_matrix", result)

    def test_unsupported_task_type(self):
        with self.assertRaises(ValueError) as ctx:
            compute_metrics([0, 1], [0, 1], "clustering")
        self.assertIn("clustering", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
