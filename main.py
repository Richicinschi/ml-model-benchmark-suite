#!/usr/bin/env python3
"""Entry point for the ML Model Benchmark Suite."""

import argparse
import sys

# Import models to ensure they register themselves
import benchmark.models  # noqa: F401

from benchmark.report import ReportGenerator
from benchmark.runner import BenchmarkRunner
from benchmark.tracking import ExperimentTracker


def main():
    parser = argparse.ArgumentParser(
        description="ML Model Benchmark Suite - train, evaluate, and compare models"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to experiment configuration file (YAML or JSON)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Show experiment history",
    )
    parser.add_argument(
        "--report",
        type=str,
        help="Generate an HTML report at the given path after running the experiment",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        help="Retrieve and generate a report for a historical run by ID",
    )

    args = parser.parse_args()

    if args.list_models:
        from benchmark.registry import REGISTRY
        models = REGISTRY.list_models()
        if not models:
            print("No models registered yet.")
            return 0
        print("Available models:")
        for name, meta in sorted(models.items()):
            print(f"  - {name} ({meta['type']})")
        return 0

    if args.history:
        tracker = ExperimentTracker()
        runs = tracker.list_runs(limit=20)
        if not runs:
            print("No experiment history found.")
            return 0
        print(f"{'ID':<5} {'Experiment':<30} {'Status':<12} {'Timestamp'}")
        print("-" * 70)
        for run in runs:
            print(
                f"{run['id']:<5} {run['experiment_name']:<30} "
                f"{run['status']:<12} {run['timestamp']}"
            )
        return 0

    if args.run_id is not None:
        tracker = ExperimentTracker()
        run = tracker.get_run(args.run_id)
        if run is None:
            print(f"Run {args.run_id} not found.")
            return 1
        report_path = args.report or f"report_run_{args.run_id}.html"
        gen = ReportGenerator()
        gen.generate(run["results"], report_path)
        print(f"Report generated: {report_path}")
        return 0

    if args.config:
        runner = BenchmarkRunner(args.config)
        results = runner.run()
        print(f"Experiment '{results['experiment_name']}' completed.")
        if results.get("run_id") is not None:
            print(f"Run ID: {results['run_id']}")
        if args.report:
            gen = ReportGenerator()
            sklearn_models = {
                name: wrapper.model
                for name, wrapper in runner.model_instances.items()
            }
            gen.generate(
                results,
                args.report,
                X=runner.X_processed,
                y=runner.y_processed,
                models=sklearn_models,
            )
            print(f"Report saved to: {args.report}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
