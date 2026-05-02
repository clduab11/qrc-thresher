"""Run manifest writer (schema v1.1).

Every run writes a manifest record. Records are appended to results/runs.csv.
Cumulative compute is tracked in results/cumulative_compute.json.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import platform as _platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

import yaml

logger = logging.getLogger(__name__)

_RUNS_CSV = Path('results') / 'runs.csv'
_COMPUTE_JSON = Path('results') / 'cumulative_compute.json'
_WATTS_PER_RUN = 15.0  # documented constant: estimated CPU power draw per run (watts)

SCHEMA_VERSION = '1.1'

CSV_FIELDNAMES = [
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


@dataclass
class RunManifest:
    """Schema v1.1 run manifest."""

    run_id: str
    timestamp_utc: str
    git_commit_hash: str
    git_branch: str
    config_path: str
    config_hash: str
    circuit_hash: str
    task_seed: int
    reservoir_seed: int
    python_version: str
    package_versions: Dict[str, str]
    backend_device: str
    runtime_per_stage_seconds: Dict[str, float]
    entanglement_metric: Optional[float]
    success: bool
    failure_reason: Optional[str]
    artifact_paths: List[str]
    platform: str = ''
    cli_command: str = ''
    task_name: str = ''
    primary_metric_name: str = ''
    primary_metric_value: Optional[float] = None


def _git_commit_hash() -> str:
    """Get current git commit hash, appending '-dirty' if repo is dirty."""
    try:
        commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f'{commit}-dirty' if status else commit
    except subprocess.CalledProcessError as exc:
        logger.warning('Could not get git commit hash: %s', exc)
        return 'unknown'


def _git_branch() -> str:
    """Get current git branch name."""
    try:
        return subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        logger.warning('Could not get git branch: %s', exc)
        return 'unknown'


def _package_versions() -> Dict[str, str]:
    """Get installed package versions via importlib.metadata."""
    required = [
        'pennylane',
        'qiskit',
        'scikit-learn',
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'pydantic',
        'pyyaml',
        'pytest',
        'ruff',
        'reservoirpy',
        'click',
    ]
    versions: Dict[str, str] = {}
    for pkg in required:
        try:
            versions[pkg] = version(pkg)
        except Exception:
            versions[pkg] = 'unknown'
    return versions


def _config_hash(config_path: Path) -> str:
    """Compute SHA-256 of canonicalized YAML config."""
    if not config_path.exists():
        return 'unknown'
    with config_path.open('r') as f:
        raw = yaml.safe_load(f)
    canonical = json.dumps(raw, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _platform_string() -> str:
    """Get a compact platform descriptor for provenance."""
    try:
        return f'{_platform.system()}-{_platform.release()}-{_platform.machine()}'
    except Exception:
        return 'unknown'


def _cli_command() -> str:
    """Reconstruct the CLI invocation from sys.argv (best-effort)."""
    try:
        return ' '.join(sys.argv)
    except Exception:
        return 'unknown'


def create_manifest(
    config_path: Path,
    circuit_hash: str,
    task_seed: int,
    reservoir_seed: int,
    backend_device: str,
    runtime_per_stage_seconds: Dict[str, float],
    entanglement_metric: Optional[float],
    success: bool,
    failure_reason: Optional[str],
    artifact_paths: List[str],
    task_name: str = '',
    primary_metric_name: str = '',
    primary_metric_value: Optional[float] = None,
) -> RunManifest:
    """Create a new run manifest record.

    Args:
        config_path: Path to YAML config file.
        circuit_hash: SHA-256 of circuit parameters.
        task_seed: Integer task seed.
        reservoir_seed: Integer reservoir seed.
        backend_device: PennyLane device string.
        runtime_per_stage_seconds: Per-stage timing dict.
        entanglement_metric: Partial-transpose log-negativity (or None).
        success: Whether the run succeeded.
        failure_reason: Failure description if not success.
        artifact_paths: Relative paths to result artifacts.
        task_name: Task identifier (e.g. 'stm', 'parity', 'narma',
            'ablation:no_entangle'). Used by gate evaluators.
        primary_metric_name: Name of the run's primary metric (e.g. 'mc',
            'accuracy', 'nrmse'). Empty when not applicable.
        primary_metric_value: Value of the primary metric, if computed.

    Returns:
        RunManifest with all required schema v1.1 fields populated.
    """
    py_info = sys.version_info
    python_version = f'{py_info.major}.{py_info.minor}.{py_info.micro}'

    return RunManifest(
        run_id=str(uuid4()),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        git_commit_hash=_git_commit_hash(),
        git_branch=_git_branch(),
        config_path=str(config_path),
        config_hash=_config_hash(config_path),
        circuit_hash=circuit_hash,
        task_seed=task_seed,
        reservoir_seed=reservoir_seed,
        python_version=python_version,
        package_versions=_package_versions(),
        backend_device=backend_device,
        runtime_per_stage_seconds=runtime_per_stage_seconds,
        entanglement_metric=entanglement_metric,
        success=success,
        failure_reason=failure_reason,
        artifact_paths=artifact_paths,
        platform=_platform_string(),
        cli_command=_cli_command(),
        task_name=task_name,
        primary_metric_name=primary_metric_name,
        primary_metric_value=primary_metric_value,
    )


def append_to_csv(manifest: RunManifest, csv_path: Path = _RUNS_CSV) -> None:
    """Append a manifest record to ``runs.csv``.

    Atomicity note: this opens the target file in append mode and writes a
    single row. CPython buffers and ``write()`` make the row write effectively
    atomic on a single host for typical row sizes, but this function does
    **not** provide cross-process file locking. Concurrent writers MUST
    serialize externally; if interrupted mid-flush the row may be partial.

    Args:
        manifest: RunManifest to append.
        csv_path: Path to runs.csv file.

    Raises:
        OSError: If file cannot be written.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
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
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    logger.info('Manifest appended to %s (run_id=%s)', csv_path, manifest.run_id)


def update_cumulative_compute(
    runtime_seconds: float,
    json_path: Path = _COMPUTE_JSON,
) -> None:
    """Update cumulative compute tracker atomically.

    Args:
        runtime_seconds: Total runtime for this run.
        json_path: Path to cumulative_compute.json.

    Raises:
        OSError: If file cannot be written.
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)

    if json_path.exists():
        with json_path.open('r') as f:
            data = json.load(f)
    else:
        data = {
            'total_runs': 0,
            'total_compute_seconds': 0.0,
            'estimated_kwh': 0.0,
            'last_updated_utc': '',
        }

    data['total_runs'] = int(data.get('total_runs', 0)) + 1
    total_seconds = float(data.get('total_compute_seconds', 0.0)) + runtime_seconds
    data['total_compute_seconds'] = total_seconds
    data['estimated_kwh'] = total_seconds * _WATTS_PER_RUN / 3.6e6
    data['last_updated_utc'] = datetime.now(timezone.utc).isoformat()

    # Atomic write: write to temp, rename
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=json_path.parent, prefix='.tmp_compute_', suffix='.json'
    )
    try:
        with os.fdopen(tmp_fd, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, json_path)
    except Exception:
        os.unlink(tmp_path)
        raise

    logger.debug(
        'Cumulative compute updated: total_runs=%d, total_seconds=%.1f',
        data['total_runs'],
        total_seconds,
    )
