"""Experiment tracking and results storage with SQLite."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .utils import setup_logger


class ExperimentTracker:
    """SQLite-backed tracker for experiment run history."""

    def __init__(self, db_path: str = "experiments.db"):
        self.db_path = Path(db_path)
        self.logger = setup_logger("ExperimentTracker")
        self._init_db()

    def _init_db(self) -> None:
        """Create the results table if it does not exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    dataset TEXT,
                    models TEXT,
                    status TEXT,
                    results_json TEXT
                )
                """
            )
            conn.commit()
        self.logger.info(f"Database ready: {self.db_path}")

    def save_run(self, results: Dict[str, Any]) -> int:
        """Serialize and store an experiment run. Returns the inserted row ID."""
        timestamp = datetime.utcnow().isoformat()
        experiment_name = results.get("experiment_name", "unknown")
        dataset = json.dumps(results.get("dataset", {}))
        models = json.dumps(results.get("models", []))
        status = results.get("status", "unknown")
        results_json = json.dumps(results, default=str)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO experiment_runs (experiment_name, timestamp, dataset, models, status, results_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (experiment_name, timestamp, dataset, models, status, results_json),
            )
            conn.commit()
            row_id = cursor.lastrowid

        self.logger.info(f"Saved experiment run with id={row_id}")
        return row_id

    def get_run(self, run_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a single experiment run by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM experiment_runs WHERE id = ?",
                (run_id,),
            )
            row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_dict(row)

    def list_runs(
        self,
        experiment_name: Optional[str] = None,
        limit: int = 100,
    ) -> list[Dict[str, Any]]:
        """List experiment runs with optional filtering."""
        query = "SELECT * FROM experiment_runs WHERE 1=1"
        params: list[Any] = []

        if experiment_name is not None:
            query += " AND experiment_name = ?"
            params.append(experiment_name)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        return [self._row_to_dict(row) for row in rows]

    def query_runs(
        self,
        model: Optional[str] = None,
        dataset_source: Optional[str] = None,
        dataset_name: Optional[str] = None,
        limit: int = 100,
    ) -> list[Dict[str, Any]]:
        """Query experiment runs by model or dataset criteria."""
        query = "SELECT * FROM experiment_runs WHERE 1=1"
        params: list[Any] = []

        if model is not None:
            query += " AND models LIKE ?"
            params.append(f'%"{model}"%')

        if dataset_source is not None:
            query += " AND dataset LIKE ?"
            params.append(f'%"source": "{dataset_source}"%')

        if dataset_name is not None:
            query += " AND dataset LIKE ?"
            params.append(f'%"name": "{dataset_name}"%')

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

        return [self._row_to_dict(row) for row in rows]

    def _row_to_dict(self, row: Any) -> Dict[str, Any]:
        """Convert a SQLite row tuple to a dictionary."""
        return {
            "id": row[0],
            "experiment_name": row[1],
            "timestamp": row[2],
            "dataset": json.loads(row[3]) if row[3] else {},
            "models": json.loads(row[4]) if row[4] else [],
            "status": row[5],
            "results": json.loads(row[6]) if row[6] else {},
        }
