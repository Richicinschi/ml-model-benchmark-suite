"""Experiment comparison utilities for benchmarking runs."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .tracking import ExperimentTracker
from .utils import setup_logger


class ExperimentComparator:
    """Compare two experiment runs and generate a comparison report."""

    def __init__(self, tracker: Optional[ExperimentTracker] = None):
        self.tracker = tracker or ExperimentTracker()
        self.logger = setup_logger("ExperimentComparator")

    def compare(
        self,
        run_id_a: int,
        run_id_b: int,
    ) -> Optional[Dict[str, Any]]:
        """Load two runs and build a structured comparison."""
        run_a = self.tracker.get_run(run_id_a)
        run_b = self.tracker.get_run(run_id_b)

        if run_a is None:
            print(f"Run {run_id_a} not found.")
            return None
        if run_b is None:
            print(f"Run {run_id_b} not found.")
            return None

        results_a = run_a.get("results", {})
        results_b = run_b.get("results", {})

        comparison = {
            "run_a": {
                "id": run_a["id"],
                "experiment_name": results_a.get("experiment_name", "Unknown"),
                "timestamp": run_a["timestamp"],
                "status": results_a.get("status", "unknown"),
                "dataset": results_a.get("dataset", {}),
                "models": results_a.get("models", []),
                "tags": results_a.get("tags", []),
                "notes": results_a.get("notes", ""),
                "aggregated_metrics": {
                    name: res.get("aggregated", {})
                    for name, res in results_a.get("results", {}).items()
                },
            },
            "run_b": {
                "id": run_b["id"],
                "experiment_name": results_b.get("experiment_name", "Unknown"),
                "timestamp": run_b["timestamp"],
                "status": results_b.get("status", "unknown"),
                "dataset": results_b.get("dataset", {}),
                "models": results_b.get("models", []),
                "tags": results_b.get("tags", []),
                "notes": results_b.get("notes", ""),
                "aggregated_metrics": {
                    name: res.get("aggregated", {})
                    for name, res in results_b.get("results", {}).items()
                },
            },
        }

        # Build side-by-side metric rows
        all_models = sorted(
            set(comparison["run_a"]["aggregated_metrics"].keys())
            | set(comparison["run_b"]["aggregated_metrics"].keys())
        )
        metric_rows: List[Dict[str, Any]] = []
        for model in all_models:
            a_val = comparison["run_a"]["aggregated_metrics"].get(model, {})
            b_val = comparison["run_b"]["aggregated_metrics"].get(model, {})
            a_metrics = a_val.get("val", {})
            b_metrics = b_val.get("val", {})
            all_keys = sorted(
                {k for k in a_metrics.keys() if k.endswith("_mean")}
                | {k for k in b_metrics.keys() if k.endswith("_mean")}
            )
            for key in all_keys:
                metric_rows.append({
                    "model": model,
                    "metric": key,
                    "run_a": a_metrics.get(key),
                    "run_b": b_metrics.get(key),
                    "delta": (
                        (b_metrics.get(key) - a_metrics.get(key))
                        if a_metrics.get(key) is not None and b_metrics.get(key) is not None
                        else None
                    ),
                })

        comparison["metric_rows"] = metric_rows
        return comparison

    def generate_report(
        self,
        comparison: Dict[str, Any],
        output_path: str,
        template_dir: Optional[str] = None,
    ) -> str:
        """Render an HTML comparison report."""
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"
        env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(["html", "xml"]),
        )
        env.filters["tojson"] = lambda x: str(x)
        template = env.get_template("compare.html")

        html = template.render(
            run_a=comparison["run_a"],
            run_b=comparison["run_b"],
            metric_rows=comparison["metric_rows"],
        )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        self.logger.info(f"Comparison report saved to {out}")
        return str(out)
