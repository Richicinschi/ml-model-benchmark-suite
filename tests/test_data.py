"""Tests for dataset loaders."""

import os
import tempfile
import unittest

import pandas as pd

from benchmark.data import CSVDatasetLoader, SklearnDatasetLoader, load_dataset


class TestCSVDatasetLoader(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.csv_path = os.path.join(self.tmpdir.name, "test.csv")
        df = pd.DataFrame({
            "feat1": [1.0, 2.0, 3.0],
            "feat2": ["a", "b", "c"],
            "target": [0, 1, 0],
        })
        df.to_csv(self.csv_path, index=False)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_load_basic(self):
        loader = CSVDatasetLoader(self.csv_path, target_column="target")
        X, y = loader.load()
        self.assertEqual(X.shape, (3, 2))
        self.assertEqual(y.shape, (3,))
        self.assertNotIn("target", X.columns)

    def test_missing_file(self):
        loader = CSVDatasetLoader("nonexistent.csv", target_column="target")
        with self.assertRaises(FileNotFoundError):
            loader.load()

    def test_missing_target_column(self):
        loader = CSVDatasetLoader(self.csv_path, target_column="missing")
        with self.assertRaises(ValueError):
            loader.load()


class TestSklearnDatasetLoader(unittest.TestCase):
    def test_load_iris(self):
        loader = SklearnDatasetLoader("iris")
        X, y = loader.load()
        self.assertEqual(X.shape, (150, 4))
        self.assertEqual(y.shape, (150,))

    def test_unsupported_dataset(self):
        loader = SklearnDatasetLoader("unknown_dataset")
        with self.assertRaises(ValueError):
            loader.load()


class TestLoadDatasetFactory(unittest.TestCase):
    def test_factory_sklearn(self):
        X, y = load_dataset({"source": "sklearn", "name": "iris"})
        self.assertEqual(X.shape, (150, 4))

    def test_factory_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "tmp.csv")
            df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "target": [0, 1]})
            df.to_csv(csv_path, index=False)
            X, y = load_dataset({
                "source": "csv",
                "path": csv_path,
                "target_column": "target",
            })
            self.assertEqual(X.shape, (2, 2))

    def test_factory_unsupported_source(self):
        with self.assertRaises(ValueError):
            load_dataset({"source": "unknown"})


if __name__ == "__main__":
    unittest.main()
