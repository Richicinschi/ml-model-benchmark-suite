#!/usr/bin/env python3
"""Entry point for the ML Model Benchmark Suite."""

import argparse
import atexit
import gc
import signal
import sys
import threading

# Global flag for graceful shutdown
_shutdown_requested = threading.Event()

def signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    sig_name = 'SIGTERM' if signum == signal.SIGTERM else 'SIGINT'
    print(f"\n[INFO] Received {sig_name} ({signum}). Shutting down gracefully...")
    _shutdown_requested.set()
    sys.exit(0)

def cleanup_resources():
    """Clean up GPU and system resources on exit."""
    try:
        gc.collect()
        
        # Clear CUDA cache if using PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        print("[INFO] Cleanup complete")
    except Exception:
        pass

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

# Register cleanup on exit
atexit.register(cleanup_resources)

# Import models to ensure they register themselves
import benchmark.models  # noqa: F401

from benchmark.batch import BatchRunner
from benchmark.compare import ExperimentComparator
from benchmark.export import ResultsExporter
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
    parser.add_argument(
        "--compare",
        nargs=2,
        type=int,
        metavar=("RUN_A", "RUN_B"),
        help="Compare two experiment runs by ID and generate a comparison report",
    )
    parser.add_argument(
        "--query-model",
        type=str,
        help="Query experiment history by model name",
    )
    parser.add_argument(
        "--query-dataset",
        type=str,
        help="Query experiment history by dataset name or source (e.g., csv, sklearn, iris)",
    )
    parser.add_argument(
        "--tags",
        type=str,
        help="Comma-separated tags for the experiment (overrides config)",
    )
    parser.add_argument(
        "--notes",
        type=str,
        help="Notes for the experiment (overrides config)",
    )
    parser.add_argument(
        "--export-json",
        type=str,
        help="Export a run or all runs to JSON (use with --run-id or --export-all)",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        help="Export a run or all runs to CSV (use with --run-id or --export-all)",
    )
    parser.add_argument(
        "--export-all",
        action="store_true",
        help="Export all runs instead of a single run (use with --export-json or --export-csv)",
    )
    parser.add_argument(
        "--batch",
        nargs="+",
        type=str,
        metavar="CONFIG",
        help="Run multiple experiment configs in batch mode",
    )

    args = parser.parse_args()

    if args.batch:
        batch = BatchRunner(args.batch, report_dir=args.report)
        batch.run_all()
        batch.print_summary()
        return 0

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

    if args.query_model or args.query_dataset:
        tracker = ExperimentTracker()
        model = args.query_model
        dataset = args.query_dataset

        # Heuristic: if dataset looks like a source keyword, use dataset_source
        dataset_source = None
        dataset_name = None
        if dataset:
            if dataset.lower() in {"csv", "sklearn", "openml"}:
                dataset_source = dataset.lower()
            else:
                dataset_name = dataset

        runs = tracker.query_runs(
            model=model,
            dataset_source=dataset_source,
            dataset_name=dataset_name,
            limit=20,
        )
        if not runs:
            print("No matching experiment history found.")
            return 0
        print(f"{'ID':<5} {'Experiment':<30} {'Dataset':<25} {'Models':<30} {'Timestamp'}")
        print("-" * 95)
        for run in runs:
            ds = run.get("dataset", {})
            ds_label = ds.get("name", ds.get("source", "-"))
            models_label = ", ".join(run.get("models", [])[:3])
            print(
                f"{run['id']:<5} {run['experiment_name']:<30} "
                f"{ds_label:<25} {models_label:<30} {run['timestamp']}"
            )
        return 0

    if args.compare:
        run_id_a, run_id_b = args.compare
        comparator = ExperimentComparator()
        comparison = comparator.compare(run_id_a, run_id_b)
        if comparison is None:
            return 1
        report_path = args.report or f"compare_run_{run_id_a}_vs_{run_id_b}.html"
        comparator.generate_report(comparison, report_path)
        print(f"Comparison report saved to: {report_path}")
        return 0

    if args.export_json or args.export_csv:
        exporter = ResultsExporter()
        try:
            if args.export_json:
                if args.export_all:
                    path = exporter.export_all_runs_json(args.export_json)
                else:
                    if args.run_id is None:
                        print("--run-id is required when exporting a single run")
                        return 1
                    path = exporter.export_run_json(args.run_id, args.export_json)
                print(f"Exported to JSON: {path}")
            if args.export_csv:
                if args.export_all:
                    path = exporter.export_all_runs_csv(args.export_csv)
                else:
                    if args.run_id is None:
                        print("--run-id is required when exporting a single run")
                        return 1
                    path = exporter.export_run_csv(args.run_id, args.export_csv)
                print(f"Exported to CSV: {path}")
        except ValueError as exc:
            print(f"Export failed: {exc}")
            return 1
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
        if args.tags is not None:
            runner.config.raw["tags"] = [t.strip() for t in args.tags.split(",") if t.strip()]
        if args.notes is not None:
            runner.config.raw["notes"] = args.notes
        try:
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
        except SystemExit:
            print("[INFO] Benchmark interrupted by user or system.")
            return 0
        except Exception as e:
            print(f"[ERROR] Benchmark failed: {e}")
            return 1
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
