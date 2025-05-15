"""End-to-end example: run a benchmark on the digits dataset and generate a report."""

import json
import sys
from pathlib import Path

# Add project root to path so benchmark package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.report import ReportGenerator
from benchmark.runner import BenchmarkRunner


CONFIG = {
    "experiment_name": "digits_classification_benchmark",
    "dataset": {
        "source": "csv",
        "path": "datasets/digits.csv",
        "target_column": "target",
    },
    "models": {
        "logistic_regression": {"max_iter": 2000},
        "random_forest": {"n_estimators": 100},
        "knn": {"n_neighbors": 5},
        "svm": {"probability": True},
    },
    "preprocessing": {
        "scale": "standard",
        "encode_target": True,
    },
    "cv": {
        "strategy": "stratified_kfold",
        "folds": 5,
    },
}


def main():
    # Write temporary config
    config_path = Path("examples/_tmp_digits_config.json")
    config_path.write_text(json.dumps(CONFIG), encoding="utf-8")

    try:
        # Run benchmark
        runner = BenchmarkRunner(str(config_path))
        results = runner.run()

        print(f"Experiment: {results['experiment_name']}")
        print(f"Run ID: {results.get('run_id')}")
        print(f"Models: {results['models']}")
        print(f"Data shape: {results['data_shape']}")
        print("\nAggregated validation metrics:")
        for model_name, model_res in results["results"].items():
            agg = model_res.get("aggregated", {})
            val_metrics = agg.get("val", {})
            print(f"\n  {model_name}:")
            for key, value in val_metrics.items():
                print(f"    {key}: {value:.4f}")

        # Generate HTML report
        report_path = "outputs/digits_benchmark_report.html"
        gen = ReportGenerator()
        gen.generate(results, report_path)
        print(f"\nReport saved to: {report_path}")
    finally:
        config_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
