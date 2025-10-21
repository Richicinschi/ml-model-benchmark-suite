"""End-to-end integration test for the full benchmark pipeline."""

import tempfile
import unittest
from pathlib import Path

import yaml

from benchmark.runner import BenchmarkRunner
from benchmark.tracking import ExperimentTracker


class TestEndToEndIntegration(unittest.TestCase):
    """Full pipeline integration test using a real config and dataset."""

    def _write_config(self, tmpdir: Path) -> str:
        config = {
            "experiment_name": "integration_test_iris",
            "dataset": {
                "source": "csv",
                "path": "datasets/iris.csv",
                "target_column": "target",
            },
            "models": {
                "logistic_regression": {"max_iter": 1000},
                "knn": {"n_neighbors": 3},
            },
            "preprocessing": {
                "scale": "standard",
                "imputation": "median",
            },
            "cv": {
                "strategy": "stratified_kfold",
                "folds": 2,
            },
            "tags": ["integration", "fast"],
            "notes": "End-to-end integration test run",
        }
        config_path = tmpdir / "integration_config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f)
        return str(config_path)

    def test_full_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_config(Path(tmpdir))
            runner = BenchmarkRunner(config_path)
            results = runner.run()

            # Top-level structure assertions
            self.assertEqual(results["experiment_name"], "integration_test_iris")
            self.assertEqual(results["status"], "completed")
            self.assertIn("run_id", results)
            self.assertIsInstance(results["run_id"], int)
            self.assertEqual(results["tags"], ["integration", "fast"])
            self.assertEqual(results["notes"], "End-to-end integration test run")

            # Dataset assertions
            self.assertIn("data_shape", results)
            self.assertIn("raw", results["data_shape"])
            self.assertIn("processed", results["data_shape"])

            # Models ran
            self.assertEqual(set(results["models"]), {"logistic_regression", "knn"})
            self.assertIn("results", results)
            model_results = results["results"]
            self.assertEqual(set(model_results.keys()), {"logistic_regression", "knn"})

            for name in results["models"]:
                mr = model_results[name]
                self.assertIn("task_type", mr)
                self.assertEqual(mr["task_type"], "classification")
                self.assertIn("folds", mr)
                self.assertEqual(len(mr["folds"]), 2)
                self.assertIn("aggregated", mr)
                self.assertIn("val", mr["aggregated"])
                self.assertIn("overfitting", mr)

                # Each fold has expected keys
                for fold in mr["folds"]:
                    self.assertIn("fold", fold)
                    self.assertIn("train_size", fold)
                    self.assertIn("val_size", fold)
                    self.assertIn("train_metrics", fold)
                    self.assertIn("val_metrics", fold)
                    self.assertIn("val_true", fold)
                    self.assertIn("val_preds", fold)

            # Verify run was persisted in tracker
            tracker = ExperimentTracker()
            persisted = tracker.get_run(results["run_id"])
            self.assertIsNotNone(persisted)
            self.assertEqual(persisted["experiment_name"], "integration_test_iris")
            self.assertIn("results", persisted)

    def test_tuning_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "experiment_name": "integration_test_tuning",
                "dataset": {
                    "source": "csv",
                    "path": "datasets/iris.csv",
                    "target_column": "target",
                },
                "models": {
                    "logistic_regression": {"max_iter": 1000},
                },
                "cv": {
                    "strategy": "stratified_kfold",
                    "folds": 2,
                },
                "tuning": {
                    "enabled": True,
                    "method": "grid",
                    "cv_folds": 2,
                    "n_jobs": 1,
                    "param_spaces": {
                        "logistic_regression": {
                            "C": [0.1, 1.0],
                        }
                    },
                },
            }
            config_path = Path(tmpdir) / "tuning_config.yaml"
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f)

            runner = BenchmarkRunner(str(config_path))
            results = runner.run()

            self.assertEqual(results["status"], "completed")
            self.assertIn("run_id", results)
            lr_result = results["results"]["logistic_regression"]
            self.assertIn("tuning", lr_result)
            self.assertIn("best_params", lr_result["tuning"])
            self.assertIn("C", lr_result["tuning"]["best_params"])

            # Verify tuning result persisted
            tracker = ExperimentTracker()
            tuning_rows = tracker.get_tuning_results(results["run_id"])
            self.assertTrue(len(tuning_rows) > 0)


if __name__ == "__main__":
    unittest.main()
