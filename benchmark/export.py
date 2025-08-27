"""Export experiment results to JSON and CSV formats."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .tracking import ExperimentTracker
from .utils import setup_logger


class ResultsExporter:
    """Export experiment run results to JSON or CSV."""

    def __init__(self, tracker: Optional[ExperimentTracker] = None):
        self.tracker = tracker or ExperimentTracker()
        self.logger = setup_logger("ResultsExporter")

    def export_run_json(
        self,
        run_id: int,
        output_path: str,
    ) -> str:
        """Export a single run to a JSON file."""
        run = self.tracker.get_run(run_id)
        if run is None:
            raise ValueError(f"Run {run_id} not found")

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(run, f, indent=2, default=str)

        self.logger.info(f"Exported run {run_id} to JSON: {out}")
        return str(out)

    def export_run_csv(
        self,
        run_id: int,
        output_path: str,
    ) -> str:
        """Export a single run's aggregated metrics to a CSV file."""
        run = self.tracker.get_run(run_id)
        if run is None:
            raise ValueError(f"Run {run_id} not found")

        results = run.get("results", {}).get("results", {})
        rows: List[Dict[str, Any]] = []
        for model_name, model_res in results.items():
            agg = model_res.get("aggregated", {})
            val_metrics = agg.get("val", {})
            train_metrics = agg.get("train", {})
            base = {
                "run_id": run["id"],
                "experiment_name": run["experiment_name"],
                "model": model_name,
                "task_type": model_res.get("task_type", ""),
            }
            for key, value in val_metrics.items():
                row = dict(base)
                row[f"val_{key}"] = value
                rows.append(row)
            for key, value in train_metrics.items():
                row = dict(base)
                row[f"train_{key}"] = value
                rows.append(row)

        if not rows:
            raise ValueError(f"Run {run_id} has no metrics to export")

        # Normalize all rows to have the same columns
        all_keys = sorted(set().union(*(r.keys() for r in rows)))

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in all_keys})

        self.logger.info(f"Exported run {run_id} to CSV: {out}")
        return str(out)

    def export_all_runs_json(self, output_path: str, limit: int = 1000) -> str:
        """Export all runs to a single JSON file."""
        runs = self.tracker.list_runs(limit=limit)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(runs, f, indent=2, default=str)

        self.logger.info(f"Exported {len(runs)} runs to JSON: {out}")
        return str(out)

    def export_all_runs_csv(self, output_path: str, limit: int = 1000) -> str:
        """Export all runs' aggregated metrics to a single CSV file."""
        runs = self.tracker.list_runs(limit=limit)
        rows: List[Dict[str, Any]] = []
        for run in runs:
            results = run.get("results", {}).get("results", {})
            for model_name, model_res in results.items():
                agg = model_res.get("aggregated", {})
                val_metrics = agg.get("val", {})
                train_metrics = agg.get("train", {})
                base = {
                    "run_id": run["id"],
                    "experiment_name": run["experiment_name"],
                    "timestamp": run["timestamp"],
                    "model": model_name,
                    "task_type": model_res.get("task_type", ""),
                }
                row = dict(base)
                for key, value in val_metrics.items():
                    row[f"val_{key}"] = value
                for key, value in train_metrics.items():
                    row[f"train_{key}"] = value
                rows.append(row)

        if not rows:
            raise ValueError("No metrics to export")

        all_keys = sorted(set().union(*(r.keys() for r in rows)))
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in all_keys})

        self.logger.info(f"Exported {len(rows)} rows from {len(runs)} runs to CSV: {out}")
        return str(out)
