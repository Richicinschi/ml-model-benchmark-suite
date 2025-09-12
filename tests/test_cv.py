"""Unit tests for cross-validation strategy utilities."""

import unittest

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

from benchmark.cv import build_cv_strategy, get_cv_splits


class TestCVStrategies(unittest.TestCase):
    """Tests for build_cv_strategy and get_cv_splits."""

    def _make_data(self, n=20):
        X = pd.DataFrame({"a": range(n), "b": range(n, n * 2)})
        y = pd.Series([0] * (n // 2) + [1] * (n // 2))
        return X, y

    def test_build_stratified_kfold(self):
        X, y = self._make_data()
        cv = build_cv_strategy({"strategy": "stratified_kfold", "folds": 3}, X, y)
        self.assertIsInstance(cv, StratifiedKFold)
        self.assertEqual(cv.n_splits, 3)

    def test_build_kfold(self):
        X, y = self._make_data()
        cv = build_cv_strategy({"strategy": "kfold", "folds": 4}, X, y)
        self.assertIsInstance(cv, KFold)
        self.assertEqual(cv.n_splits, 4)

    def test_build_time_series_split(self):
        X, y = self._make_data()
        cv = build_cv_strategy({"strategy": "time_series_split", "folds": 5}, X, y)
        self.assertIsInstance(cv, TimeSeriesSplit)
        self.assertEqual(cv.n_splits, 5)

    def test_build_unsupported_strategy(self):
        X, y = self._make_data()
        with self.assertRaises(ValueError) as ctx:
            build_cv_strategy({"strategy": "unknown"}, X, y)
        self.assertIn("unknown", str(ctx.exception))

    def test_default_strategy_is_stratified_kfold(self):
        X, y = self._make_data()
        cv = build_cv_strategy({}, X, y)
        self.assertIsInstance(cv, StratifiedKFold)
        self.assertEqual(cv.n_splits, 5)

    def test_shuffle_and_random_state(self):
        X, y = self._make_data()
        cv = build_cv_strategy(
            {"strategy": "kfold", "folds": 3, "shuffle": True, "random_state": 123},
            X,
            y,
        )
        self.assertTrue(cv.shuffle)
        self.assertEqual(cv.random_state, 123)

    def test_get_cv_splits_length(self):
        X, y = self._make_data()
        cv = build_cv_strategy({"strategy": "kfold", "folds": 4}, X, y)
        splits = get_cv_splits(cv, X, y)
        self.assertEqual(len(splits), 4)
        for train_idx, val_idx in splits:
            self.assertEqual(len(train_idx) + len(val_idx), len(X))

    def test_stratified_preserves_class_distribution(self):
        X, y = self._make_data(n=100)
        cv = build_cv_strategy({"strategy": "stratified_kfold", "folds": 5}, X, y)
        splits = get_cv_splits(cv, X, y)
        for train_idx, val_idx in splits:
            train_y = y.iloc[train_idx]
            val_y = y.iloc[val_idx]
            # With 100 samples and 2 balanced classes, each fold should have ~10 of each class in val
            self.assertEqual(val_y.value_counts().get(0, 0), 10)
            self.assertEqual(val_y.value_counts().get(1, 0), 10)


if __name__ == "__main__":
    unittest.main()
