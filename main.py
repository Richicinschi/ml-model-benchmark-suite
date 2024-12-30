#!/usr/bin/env python3
"""Entry point for the ML Model Benchmark Suite."""

import argparse
import sys

from benchmark.runner import BenchmarkRunner


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
        print("Experiment history will be shown here.")
        return 0

    if args.config:
        runner = BenchmarkRunner(args.config)
        results = runner.run()
        print(f"Experiment '{results['experiment_name']}' completed.")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
