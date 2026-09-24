"""Run manifest writer (schema 1.4).

Every run writes a manifest record. Records are appended to results/runs.csv through one row
builder, ``manifest_row``, which db.py shares (docs/DECISIONS.md D013). Cumulative compute is
tracked in results/cumulative_compute.json.

Schema history: 1.1 the base record; 1.2 adds measurement_model (D004); 1.3 adds the search
budget n_configs and n_validation_evals (D009); 1.4 adds secondary_metrics, device, precision,
design, sweep_id and tuning_record_sha (D011, D013, PI rulings 1 and 3).
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

import yaml

from qrc_thresher.config import MEASUREMENT_LABELS

logger = logging.getLogger(__name__)

_RUNS_CSV = Path('results') / 'runs.csv'
_COMPUTE_JSON = Path('results') / 'cumulative_compute.json'
_WATTS_PER_RUN = 15.0  # documented constant: estimated CPU power draw per run (watts)

SCHEMA_VERSION = '1.4'

# The constants every row carries until D006's GPU extra exists (D013).
DEVICE = 'cpu'
PRECISION = 'float64'
# Design labels (D011, D013, PI ruling 1): the tuned design of the record; the untuned default
# QRC (depth 3, pi, w = 2) or ESN preset; today's w = 1 circuit, reporting-only; an ablation
# inheriting the design it was matched to.
DESIGNS = ('tuned', 'default', 'default_w1', 'inherited')

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
    'measurement_model',
    'n_configs',
    'n_validation_evals',
    'secondary_metrics',
    'device',
    'precision',
    'design',
    'sweep_id',
    'tuning_record_sha',
]


class RunsCsvSchemaError(ValueError):
    """An existing runs.csv has a header that differs from CSV_FIELDNAMES."""


def check_runs_csv_header(csv_path: Path) -> None:
    """Refuse to append to a runs.csv whose header differs from CSV_FIELDNAMES.

    A missing or empty file passes: the writer adds the header. Appending under a
    different header would silently misalign every new row, so this raises instead.

    Args:
        csv_path: Path to the runs.csv file about to be appended to.

    Raises:
        RunsCsvSchemaError: If the file's header differs from CSV_FIELDNAMES.
    """
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return
    with csv_path.open(newline='') as f:
        existing = next(csv.reader(f), [])
    if existing == CSV_FIELDNAMES:
        return
    missing = [c for c in CSV_FIELDNAMES if c not in existing]
    unexpected = [c for c in existing if c not in CSV_FIELDNAMES]
    raise RunsCsvSchemaError(
        f'{csv_path} has a different column schema, so no row was appended.\n'
        f'  existing header ({len(existing)} columns): {", ".join(existing)}\n'
        f'  expected header (manifest schema {SCHEMA_VERSION}, {len(CSV_FIELDNAMES)} columns): '
        f'{", ".join(CSV_FIELDNAMES)}\n'
        f'  missing from the file: {", ".join(missing) or "none"}; '
        f'not in schema {SCHEMA_VERSION}: {", ".join(unexpected) or "none"}\n'
        'Move the file aside (for example to runs.old.csv) or migrate it to the expected '
        'header, then rerun.'
    )


@dataclass
class RunManifest:
    """Schema 1.4 run manifest (one row of results/runs.csv)."""

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
    measurement_model: str = 'exact'
    n_configs: Optional[int] = None
    n_validation_evals: Optional[int] = None
    secondary_metrics: Dict[str, float] = field(default_factory=dict)
    device: str = DEVICE
    precision: str = PRECISION
    design: str = 'default'
    sweep_id: str = ''
    tuning_record_sha: str = ''


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
    measurement_model: str = 'exact',
    n_configs: Optional[int] = None,
    n_validation_evals: Optional[int] = None,
    secondary_metrics: Optional[Dict[str, float]] = None,
    design: str = 'default',
    sweep_id: str = '',
    tuning_record_sha: str = '',
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
        measurement_model: Measurement model the run's features were computed
            under (config field measurement.model). 'exact' is an oracle upper bound.
        n_configs: Model hyperparameter configurations evaluated before this run was
            deployed (1 when nothing was searched).
        n_validation_evals: Configuration x validation-block evaluations used to choose
            among them (0 when nothing was searched). The readout's own RidgeCV is the
            same for every model and is not counted.
        secondary_metrics: Reported, never decided on: STM rows carry mc_total (k = 0..K)
            and mc_k0 beside the primary stm_memory (D013).
        design: One of DESIGNS: 'tuned' (deployed from the tuning record), 'default' (the
            untuned default), 'default_w1' (today's w = 1 circuit, reporting-only) or
            'inherited' (an ablation matched to a design).
        sweep_id: The tuning record's sweep stamp; '' for a config without a tuning block.
        tuning_record_sha: The tuning record's SHA-256; '' likewise.

    Returns:
        RunManifest with all schema 1.4 fields populated.

    Raises:
        ValueError: If measurement_model is not a known measurement model, or design is not
            one of DESIGNS.
    """
    if measurement_model not in MEASUREMENT_LABELS:
        raise ValueError(
            f'Unknown measurement_model {measurement_model!r}; '
            f'expected one of {sorted(MEASUREMENT_LABELS)}'
        )
    if design not in DESIGNS:
        raise ValueError(f'Unknown design {design!r}; expected one of {DESIGNS}')
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
        measurement_model=measurement_model,
        n_configs=n_configs,
        n_validation_evals=n_validation_evals,
        secondary_metrics=dict(secondary_metrics or {}),
        design=design,
        sweep_id=sweep_id,
        tuning_record_sha=tuning_record_sha,
    )


def manifest_row(manifest: RunManifest) -> Dict[str, object]:
    """The runs.csv row of a manifest, in CSV_FIELDNAMES order: the one row builder (D013)."""
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
        'measurement_model': manifest.measurement_model,
        'n_configs': manifest.n_configs,
        'n_validation_evals': manifest.n_validation_evals,
        'secondary_metrics': json.dumps(manifest.secondary_metrics),
        'device': manifest.device,
        'precision': manifest.precision,
        'design': manifest.design,
        'sweep_id': manifest.sweep_id,
        'tuning_record_sha': manifest.tuning_record_sha,
    }
    assert list(row) == CSV_FIELDNAMES
    return row


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
        RunsCsvSchemaError: If the file exists with a header that differs from
            CSV_FIELDNAMES. Nothing is appended in that case.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    check_runs_csv_header(csv_path)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    row = manifest_row(manifest)

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
