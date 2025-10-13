"""Unit tests for HTML report generation."""

import base64
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark.report import ReportGenerator


class TestReportGeneratorInit(unittest.TestCase):
    """Tests for ReportGenerator initialization."""

    def test_init_loads_template_dir(self):
        gen = ReportGenerator()
        self.assertIsNotNone(gen.env)


class TestFigToBase64(unittest.TestCase):
    """Tests for _fig_to_base64 conversion."""

    def test_converts_figure_to_base64_png(self):
        gen = ReportGenerator()
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        b64 = gen._fig_to_base64(fig)
        self.assertTrue(b64.startswith("data:image/png;base64,"))
        data = b64.split(",")[1]
        decoded = base64.b64decode(data)
        self.assertTrue(decoded.startswith(b"\x89PNG"))


class TestGenerateReport(unittest.TestCase):
    """Tests for generating HTML reports."""

    def _make_results(self, task_type="classification"):
        return {
            "experiment_name": "test_experiment",
            "status": "completed",
            "dataset": {"source": "csv", "name": "test.csv"},
            "models": ["logistic_regression"],
            "tags": ["unit-test"],
            "notes": "test notes",
            "results": {
                "logistic_regression": {
                    "task_type": task_type,
                    "aggregated": {
                        "val": {
                            "accuracy_mean": 0.95,
                            "accuracy_std": 0.02,
                            "f1_mean": 0.94,
                            "f1_std": 0.03,
                        }
                    },
                    "feature_importance": {"a": 0.6, "b": 0.4},
                    "overfitting": {
                        "status": "overfitting",
                        "primary_metric": "accuracy",
                        "avg_train": 0.99,
                        "avg_val": 0.95,
                        "gap": 0.04,
                        "relative_gap": 0.0404,
                        "threshold": 0.05,
                        "warnings": ["Large train/val gap on accuracy"],
                    },
                    "folds": [
                        {
                            "fold": 1,
                            "train_metrics": {"accuracy": 0.99},
                            "val_metrics": {
                                "accuracy": 0.95,
                                "confusion_matrix": [[5, 1], [0, 4]],
                            },
                            "val_true": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                            "val_proba": [
                                [0.9, 0.1], [0.8, 0.2], [0.85, 0.15],
                                [0.88, 0.12], [0.82, 0.18], [0.2, 0.8],
                                [0.15, 0.85], [0.1, 0.9], [0.12, 0.88], [0.18, 0.82]
                            ],
                        }
                    ],
                }
            },
        }

    def test_generate_creates_html_file(self):
        gen = ReportGenerator()
        results = self._make_results()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            path = gen.generate(results, str(output_path))
            self.assertTrue(Path(path).exists())
            content = Path(path).read_text(encoding="utf-8")
            self.assertIn("test_experiment", content)
            self.assertIn("completed", content)
            self.assertIn("test.csv", content)
            self.assertIn("unit-test", content)
            self.assertIn("test notes", content)

    def test_generate_with_regression(self):
        gen = ReportGenerator()
        results = {
            "experiment_name": "regression_test",
            "status": "completed",
            "dataset": {"source": "csv", "name": "reg.csv"},
            "models": ["linear_regression"],
            "tags": [],
            "notes": "",
            "results": {
                "linear_regression": {
                    "task_type": "regression",
                    "aggregated": {
                        "val": {
                            "r2_mean": 0.85,
                            "r2_std": 0.05,
                            "mse_mean": 0.15,
                        }
                    },
                    "feature_importance": None,
                    "overfitting": {
                        "status": "ok",
                        "primary_metric": "r2",
                        "avg_train": 0.90,
                        "avg_val": 0.85,
                        "gap": 0.05,
                        "relative_gap": 0.0556,
                        "threshold": 0.10,
                        "warnings": [],
                    },
                    "folds": [
                        {
                            "fold": 1,
                            "train_metrics": {"r2": 0.90},
                            "val_metrics": {"r2": 0.85},
                            "val_true": [1.0, 2.0, 3.0],
                            "val_preds": [1.1, 1.9, 3.2],
                        }
                    ],
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            path = gen.generate(results, str(output_path))
            self.assertTrue(Path(path).exists())
            content = Path(path).read_text(encoding="utf-8")
            self.assertIn("regression_test", content)

    def test_generate_plots_with_single_model(self):
        """Single model should still produce report without comparison plots."""
        gen = ReportGenerator()
        results = self._make_results()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            path = gen.generate(results, str(output_path))
            content = Path(path).read_text(encoding="utf-8")
            # Confusion matrix and ROC curve should be present for single classification model
            self.assertIn("data:image/png;base64,", content)


if __name__ == "__main__":
    unittest.main()
