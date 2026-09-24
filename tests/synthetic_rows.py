"""Synthetic manifest rows (schema 1.4) for the CP4 statistics and gate tests.

Not a test module. Rows carry the columns runs.csv gains in schema 1.4 (docs/DECISIONS.md D013):
design, sweep_id, secondary_metrics, device and precision, beside the seed keys and budgets. The
hashes are stand-ins with the structure the family relies on: a tuned design's hash per (task,
pair), an ablation's hash as a function of the design hash, and default-design hashes per window.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Iterable, Optional, Sequence, Tuple

import pandas as pd

Pair = Tuple[int, int]

COLUMNS = [
    'run_id', 'timestamp_utc', 'git_commit_hash', 'git_branch', 'config_path', 'config_hash',
    'circuit_hash', 'task_seed', 'reservoir_seed', 'python_version', 'backend_device',
    'runtime_per_stage_seconds', 'entanglement_metric', 'success', 'failure_reason',
    'artifact_paths', 'package_versions', 'platform', 'cli_command', 'task_name',
    'primary_metric_name', 'primary_metric_value', 'measurement_model', 'n_configs',
    'n_validation_evals', 'secondary_metrics', 'device', 'precision', 'design', 'sweep_id',
    'tuning_record_sha',
]
CONFIG_HASH = hashlib.sha256(b'configs/comparative.yaml').hexdigest()
SWEEP_ID = '20260923T120000000000Z'
RECORD_SHA = hashlib.sha256(b'tuning record').hexdigest()
DEFAULT_LABELS = {2: 'default', 1: 'default_w1'}  # PI ruling 1: w = 2 compared, w = 1 reported
PAIRS = [(42 + i, 137 + i) for i in range(12)]
TUNED_BUDGET = (60, 300)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def design_hash(task: str, pair: Pair) -> str:
    """Stand-in for the tuning record's design_<task>(pair) circuit hash."""
    return sha(f'design_{task}:{pair[0]}/{pair[1]}')


def ablation_hash(design: str, name: str, pair: Pair) -> str:
    """Stand-in for D010's suffix rule: the ablation's hash is a function of the design's."""
    return sha(f'{design},{name}')


def default_hash(window: int, pair: Pair) -> str:
    return sha(f'default_w{window}:{pair[0]}/{pair[1]}')


def classical_hash(model: str, pair: Pair, design: str = 'tuned') -> str:
    return sha(f'{model}:{design}:{pair[0]}/{pair[1]}')


def row(
    task_name: str,
    pair: Pair,
    value: float,
    *,
    metric: str,
    circuit_hash: str,
    design: str = 'tuned',
    budget: Tuple[int, int] = TUNED_BUDGET,
    config_hash: str = CONFIG_HASH,
    sweep_id: str = SWEEP_ID,
    tuning_record_sha: str = RECORD_SHA,
    secondary: Optional[Dict[str, float]] = None,
    backend: str = 'default.qubit',
    success: bool = True,
    run_id: Optional[str] = None,
) -> dict:
    task_seed, reservoir_seed = pair
    return {
        'run_id': run_id or sha(f'{task_name}:{design}:{circuit_hash}:{pair}:{value}')[:12],
        'timestamp_utc': '2026-09-23T12:00:00+00:00',
        'git_commit_hash': 'deadbeef',
        'git_branch': 'refactor/2026-09',
        'config_path': 'configs/comparative.yaml',
        'config_hash': config_hash,
        'circuit_hash': circuit_hash,
        'task_seed': task_seed,
        'reservoir_seed': reservoir_seed,
        'python_version': '3.13.0',
        'backend_device': backend,
        'runtime_per_stage_seconds': '{}',
        'entanglement_metric': None,
        'success': success,
        'failure_reason': None,
        'artifact_paths': '[]',
        'package_versions': '{}',
        'platform': 'test',
        'cli_command': 'test',
        'task_name': task_name,
        'primary_metric_name': metric,
        'primary_metric_value': value,
        'measurement_model': 'exact',
        'n_configs': budget[0],
        'n_validation_evals': budget[1],
        'secondary_metrics': json.dumps(secondary or {}),
        'device': 'cpu',
        'precision': 'float64',
        'design': design,
        'sweep_id': sweep_id,
        'tuning_record_sha': tuning_record_sha,
    }


def frame(rows: Iterable[dict]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=COLUMNS)


def arm(
    task_name: str,
    values: Dict[Pair, float],
    *,
    metric: str,
    hashes: Dict[Pair, str],
    design: str = 'tuned',
    budget: Tuple[int, int] = TUNED_BUDGET,
    secondary: Optional[Dict[Pair, Dict[str, float]]] = None,
    **kwargs,
) -> pd.DataFrame:
    """One arm: a row per pair, with per-pair hashes (and optional secondary metrics)."""
    return frame(
        row(
            task_name, pair, values[pair], metric=metric, circuit_hash=hashes[pair],
            design=design, budget=budget, secondary=(secondary or {}).get(pair), **kwargs,
        )
        for pair in values
    )


def qrc_arm(task: str, design_task: str, values: Sequence[float], *, metric: str,
            pairs: Sequence[Pair] = PAIRS, **kwargs) -> pd.DataFrame:
    """Rows of ``task`` run with design_<design_task>(pair) (the tuned QRC arm)."""
    hashes = {p: design_hash(design_task, p) for p in pairs}
    return arm(task, dict(zip(pairs, values)), metric=metric, hashes=hashes, **kwargs)


def inherited_arm(ablation: str, task: str, design_task: str, values: Sequence[float], *,
                  metric: str, pairs: Sequence[Pair] = PAIRS, **kwargs) -> pd.DataFrame:
    """Rows of ``ablation:<ablation>`` on ``task`` inheriting design_<design_task>(pair)."""
    hashes = {p: ablation_hash(design_hash(design_task, p), ablation, p) for p in pairs}
    return arm(f'ablation:{ablation}', dict(zip(pairs, values)), metric=metric, hashes=hashes,
               design='inherited', budget=(1, 0), **kwargs)


def classical_arm(task_name: str, values: Sequence[float], *, metric: str,
                  pairs: Sequence[Pair] = PAIRS, design: str = 'tuned',
                  budget: Tuple[int, int] = TUNED_BUDGET, **kwargs) -> pd.DataFrame:
    """Tuned (or default) ESN / RKS rows: task_name esn, esn_parity, esn_narma, rks_parity ..."""
    model = task_name.split('_')[0]
    hashes = {p: classical_hash(model, p, design) for p in pairs}
    return arm(task_name, dict(zip(pairs, values)), metric=metric, hashes=hashes, design=design,
               budget=budget, backend=f'numpy_{model}', **kwargs)


def default_qrc_arm(task: str, window: int, values: Sequence[float], *, metric: str,
                    pairs: Sequence[Pair] = PAIRS, **kwargs) -> pd.DataFrame:
    """Default QRC rows: design 'default' at w = 2 (compared), 'default_w1' at w = 1 (reported)."""
    hashes = {p: default_hash(window, p) for p in pairs}
    return arm(task, dict(zip(pairs, values)), metric=metric, hashes=hashes,
               design=DEFAULT_LABELS[window], budget=(1, 0), **kwargs)


def default_inherited_arm(ablation: str, task: str, window: int, values: Sequence[float], *,
                          metric: str, pairs: Sequence[Pair] = PAIRS, **kwargs) -> pd.DataFrame:
    hashes = {p: ablation_hash(default_hash(window, p), ablation, p) for p in pairs}
    return arm(f'ablation:{ablation}', dict(zip(pairs, values)), metric=metric, hashes=hashes,
               design='inherited', budget=(1, 0), **kwargs)
