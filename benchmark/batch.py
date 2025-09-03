"""Batch experiment runner for executing multiple configs."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .report import ReportGenerator
from .runner import BenchmarkRunner
from .utils import setup_logger


class BatchRunner:
    """Run multiple benchmark experiments from a list of config files."""

    def __init__(self, config_paths: List[str], report_dir: Optional[str] = None):
        self.config_paths = [Path(p) for p in config_paths]
        self.report_dir = Path(report_dir) if report_dir else None
        self.logger = setup_logger("BatchRunner")
        self.results: List[Dict[str, Any]] = []

    def run_all(self) -> List[Dict[str, Any]]:
        """Execute all configs sequentially and collect summaries."""
        self.results = []
        for config_path in self.config_paths:
            if not config_path.exists():
                self.logger.error(f"Config not found: {config_path}")
                self.results.append({
                    "config_path": str(config_path),
                    "status": "failed",
                    "error": "Config file not found",
                })
                continue

            self.logger.info(f"Running batch experiment: {config_path}")
            try:
                runner = BenchmarkRunner(str(config_path))
                result = runner.run()
                summary = {
                    "config_path": str(config_path),
                    "run_id": result.get("run_id"),
                    "experiment_name": result.get("experiment_name"),
                    "status": result.get("status", "completed"),
                    "models": result.get("models", []),
                }
                if self.report_dir:
                    self.report_dir.mkdir(parents=True, exist_ok=True)
                    report_path = self.report_dir / f"report_run_{result['run_id']}.html"
                    gen = ReportGenerator()
                    sklearn_models = {
                        name: wrapper.model
                        for name, wrapper in runner.model_instances.items()
                    }
                    gen.generate(
                        result,
                        str(report_path),
                        X=runner.X_processed,
                        y=runner.y_processed,
                        models=sklearn_models,
                    )
                    summary["report_path"] = str(report_path)
                self.results.append(summary)
                self.logger.info(f"Batch experiment complete: {result.get('experiment_name')} (run_id={result.get('run_id')})")
            except Exception as exc:
                self.logger.error(f"Batch experiment failed for {config_path}: {exc}")
                self.results.append({
                    "config_path": str(config_path),
                    "status": "failed",
                    "error": str(exc),
                })

        return self.results

    def print_summary(self) -> None:
        """Print a tabular summary of batch results to stdout."""
        print(f"{'Config':<40} {'Run ID':<8} {'Experiment':<30} {'Status':<12}")
        print("-" * 95)
        for res in self.results:
            config_name = Path(res["config_path"]).name
            run_id = str(res.get("run_id", "-"))
            experiment = res.get("experiment_name", "-")
            status = res.get("status", "unknown")
            print(f"{config_name:<40} {run_id:<8} {experiment:<30} {status:<12}")
