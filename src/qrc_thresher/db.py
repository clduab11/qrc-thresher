"""SQLite database for experiment results with WAL mode."""

from __future__ import annotations

import csv
import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

import filelock

from qrc_thresher.proof.run_manifest import RunManifest

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    timestamp_utc TEXT NOT NULL,
    git_commit_hash TEXT,
    git_branch TEXT,
    config_path TEXT,
    config_hash TEXT,
    circuit_hash TEXT,
    task_seed INTEGER,
    reservoir_seed INTEGER,
    python_version TEXT,
    backend_device TEXT,
    task_name TEXT NOT NULL,
    primary_metric_name TEXT,
    primary_metric_value REAL,
    success INTEGER NOT NULL,
    failure_reason TEXT,
    gates_passed JSON,
    manifest JSON,
    runtime_seconds REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_task_success ON runs(task_name, success);
CREATE INDEX IF NOT EXISTS idx_timestamp ON runs(timestamp_utc);
"""

_CSV_FIELDNAMES = [
    'run_id',
    'timestamp_utc',
    'git_commit_hash',
    'git_branch',
    'config_path',
    'config_hash',
    'circuit_hash',
    'task_seed',
    'reservoir_seed',
    'python_version',
    'backend_device',
    'runtime_per_stage_seconds',
    'entanglement_metric',
    'success',
    'failure_reason',
    'artifact_paths',
    'package_versions',
    'platform',
    'cli_command',
    'task_name',
    'primary_metric_name',
    'primary_metric_value',
]


class ExperimentDB:
    """SQLite database for experiment results with WAL mode."""

    def __init__(self, db_path: str = "results/experiments.db"):
        """Initialize DB with WAL mode.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_dir_exists()
        self._connect()

    def _ensure_dir_exists(self) -> None:
        """Ensure the database directory exists."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> None:
        """Open connection and set WAL mode."""
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        """Create schema if not exists."""
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def insert(self, manifest: RunManifest) -> None:
        """Insert a run manifest into the DB.

        Also appends to legacy CSV for backward compatibility.

        Args:
            manifest: RunManifest to insert.
        """
        runtime_seconds = None
        if manifest.runtime_per_stage_seconds:
            runtime_seconds = sum(manifest.runtime_per_stage_seconds.values())

        gates_passed = None
        if manifest.artifact_paths:
            for path in manifest.artifact_paths:
                if 'gates_passed' in path.lower():
                    try:
                        with open(path, 'r') as f:
                            gates_passed = json.load(f)
                    except Exception:
                        pass
                    break

        self._conn.execute(
            """
            INSERT INTO runs (
                run_id, timestamp_utc, git_commit_hash, git_branch,
                config_path, config_hash, circuit_hash, task_seed,
                reservoir_seed, python_version, backend_device, task_name,
                primary_metric_name, primary_metric_value, success,
                failure_reason, gates_passed, manifest, runtime_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                manifest.run_id,
                manifest.timestamp_utc,
                manifest.git_commit_hash,
                manifest.git_branch,
                manifest.config_path,
                manifest.config_hash,
                manifest.circuit_hash,
                manifest.task_seed,
                manifest.reservoir_seed,
                manifest.python_version,
                manifest.backend_device,
                manifest.task_name,
                manifest.primary_metric_name,
                manifest.primary_metric_value,
                1 if manifest.success else 0,
                manifest.failure_reason,
                json.dumps(gates_passed) if gates_passed else None,
                self._manifest_to_json(manifest),
                runtime_seconds,
            ),
        )
        self._conn.commit()
        self._append_to_csv(manifest)
        logger.info("Inserted run %s into ExperimentDB", manifest.run_id)

    def _manifest_to_json(self, manifest: RunManifest) -> str:
        """Serialize manifest to JSON string."""
        return json.dumps({
            'run_id': manifest.run_id,
            'timestamp_utc': manifest.timestamp_utc,
            'git_commit_hash': manifest.git_commit_hash,
            'git_branch': manifest.git_branch,
            'config_path': manifest.config_path,
            'config_hash': manifest.config_hash,
            'circuit_hash': manifest.circuit_hash,
            'task_seed': manifest.task_seed,
            'reservoir_seed': manifest.reservoir_seed,
            'python_version': manifest.python_version,
            'package_versions': manifest.package_versions,
            'backend_device': manifest.backend_device,
            'runtime_per_stage_seconds': manifest.runtime_per_stage_seconds,
            'entanglement_metric': manifest.entanglement_metric,
            'success': manifest.success,
            'failure_reason': manifest.failure_reason,
            'artifact_paths': manifest.artifact_paths,
            'platform': manifest.platform,
            'cli_command': manifest.cli_command,
            'task_name': manifest.task_name,
            'primary_metric_name': manifest.primary_metric_name,
            'primary_metric_value': manifest.primary_metric_value,
        })

    def _append_to_csv(self, manifest: RunManifest) -> None:
        """Append manifest to legacy CSV with file locking."""
        csv_path = Path('results') / 'runs.csv'
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = csv_path.with_suffix('.csv.lock')

        try:
            with filelock.FileLock(str(lock_path), timeout=30):
                self._write_csv_row(manifest, csv_path)
        except Exception as exc:
            logger.warning("Failed to write CSV row: %s", exc)

    def _write_csv_row(self, manifest: RunManifest, csv_path: Path) -> None:
        """Write a single row to CSV."""
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0

        row = {
            'run_id': manifest.run_id,
            'timestamp_utc': manifest.timestamp_utc,
            'git_commit_hash': manifest.git_commit_hash,
            'git_branch': manifest.git_branch,
            'config_path': manifest.config_path,
            'config_hash': manifest.config_hash,
            'circuit_hash': manifest.circuit_hash,
            'task_seed': manifest.task_seed,
            'reservoir_seed': manifest.reservoir_seed,
            'python_version': manifest.python_version,
            'backend_device': manifest.backend_device,
            'runtime_per_stage_seconds': json.dumps(manifest.runtime_per_stage_seconds),
            'entanglement_metric': manifest.entanglement_metric,
            'success': manifest.success,
            'failure_reason': manifest.failure_reason,
            'artifact_paths': json.dumps(manifest.artifact_paths),
            'package_versions': json.dumps(manifest.package_versions),
            'platform': manifest.platform,
            'cli_command': manifest.cli_command,
            'task_name': manifest.task_name,
            'primary_metric_name': manifest.primary_metric_name,
            'primary_metric_value': manifest.primary_metric_value,
        }

        with csv_path.open('a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Execute a read query and return results.

        Args:
            sql: SQL query string.
            params: Query parameters.

        Returns:
            List of row dictionaries.
        """
        cursor = self._conn.execute(sql, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_runs_by_task(self, task_name: str) -> list[dict]:
        """Get all runs for a task.

        Args:
            task_name: Task name to filter by.

        Returns:
            List of run records.
        """
        return self.query(
            "SELECT * FROM runs WHERE task_name = ? ORDER BY timestamp_utc DESC",
            (task_name,),
        )

    def get_successful_runs(self, task_name: str) -> list[dict]:
        """Get successful runs for a task.

        Args:
            task_name: Task name to filter by.

        Returns:
            List of successful run records.
        """
        return self.query(
            "SELECT * FROM runs WHERE task_name = ? AND success = 1 ORDER BY timestamp_utc DESC",
            (task_name,),
        )

    def get_gate_results(self, gate_name: str) -> list[dict]:
        """Get gate evaluation results for a specific gate.

        Args:
            gate_name: Name of the gate to look up.

        Returns:
            List of run records where this gate passed.
        """
        return self.query(
            """
            SELECT * FROM runs
            WHERE gates_passed LIKE ?
            ORDER BY timestamp_utc DESC
            """,
            (f'%"{gate_name}"%',),
        )

    def export_csv(self, csv_path: str) -> None:
        """Export all runs to CSV for backward compatibility.

        Args:
            csv_path: Path to output CSV file.
        """
        runs = self.query("SELECT * FROM runs ORDER BY timestamp_utc")
        if not runs:
            return

        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        with csv_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=runs[0].keys())
            writer.writeheader()
            writer.writerows(runs)

        logger.info("Exported %d runs to %s", len(runs), csv_path)

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
