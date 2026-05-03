"""Parallel execution engine for qrc-thresher.

Uses ProcessPoolExecutor to run multiple benchmark seeds in parallel.
Results are collected and written to runs.csv atomically with file locking.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from qrc_thresher.config import AlphaLiteConfig
from qrc_thresher.metrics.runtime import StageTimer
from qrc_thresher.proof.run_manifest import (
    RunManifest,
    append_to_csv,
    create_manifest,
    update_cumulative_compute,
)

logger = logging.getLogger(__name__)

_RUNS_CSV = Path('results') / 'runs.csv'
_RUNS_LOCK = _RUNS_CSV.with_suffix('.csv.lock')


def _run_single_seed(args: Tuple[int, str, Dict[str, Any]]) -> RunManifest:
    """Run a single seed and return manifest.

    Args:
        args: Tuple of (seed_index, task_name, config_dict)

    Returns:
        RunManifest for this seed's run
    """
    seed_index, task_name, config_dict = args

    base_task_seed = config_dict['seeds']['task_seed']
    base_reservoir_seed = config_dict['seeds']['reservoir_seed']
    task_seed = base_task_seed + seed_index
    reservoir_seed = base_reservoir_seed + seed_index

    config_dict = config_dict.copy()
    config_dict['seeds'] = config_dict['seeds'].copy()
    config_dict['seeds']['task_seed'] = task_seed
    config_dict['seeds']['reservoir_seed'] = reservoir_seed
    config_dict['seeds']['n_seeds'] = 1

    config = AlphaLiteConfig.model_validate(config_dict)

    cfg_path_str = config_dict.get('_config_path', 'unknown')
    cfg_path = Path(cfg_path_str)

    rng_task = np.random.default_rng(task_seed)
    rng_reservoir = np.random.default_rng(reservoir_seed)

    timer = StageTimer()
    circuit_hash = 'n/a'
    artifact_paths: List[str] = []
    success = False
    failure_reason: Optional[str] = None
    primary_metric_name = ''
    primary_metric_value: Optional[float] = None
    entanglement_metric: Optional[float] = None

    try:
        if task_name == 'stm':
            from qrc_thresher.metrics.scoring import memory_capacity
            from qrc_thresher.reservoirs.pennylane_qrc import (
                build_reservoir_params,
                compute_circuit_hash,
                extract_features,
                train_readout,
            )
            from qrc_thresher.tasks.stm import generate_stm

            with timer.stage('task_generation'):
                ds = generate_stm(
                    length=config.task.length,
                    delay_max=config.task.delay_max or 20,
                    train_frac=config.task.train_frac,
                    rng=rng_task,
                )

            with timer.stage('reservoir_build'):
                params = build_reservoir_params(
                    n_qubits=config.reservoir.n_qubits,
                    depth=config.reservoir.depth,
                    readout=config.reservoir.readout,
                    backend=config.reservoir.backend,
                    rng=rng_reservoir,
                )
                circuit_hash = compute_circuit_hash(params)

            with timer.stage('feature_extraction'):
                X = extract_features(ds.u, params)

            with timer.stage('readout_training'):
                model = train_readout(
                    X[: ds.train_end],
                    ds.targets[: ds.train_end],
                    config.training.ridge_alphas,
                    config.training.cv_folds,
                )

            with timer.stage('evaluation'):
                y_pred = model.predict(X[ds.train_end :])
                mc = memory_capacity(y_pred, ds.targets[ds.train_end :])
                primary_metric_name = 'mc'
                primary_metric_value = float(mc)

        elif task_name == 'parity':
            from qrc_thresher.metrics.scoring import classification_accuracy
            from qrc_thresher.reservoirs.pennylane_qrc import (
                build_reservoir_params,
                compute_circuit_hash,
                extract_features,
                train_readout,
            )
            from qrc_thresher.tasks.temporal_parity import generate_parity

            window = config.task.parity_window or 3
            with timer.stage('task_generation'):
                ds = generate_parity(
                    length=config.task.length,
                    window=window,
                    train_frac=config.task.train_frac,
                    rng=rng_task,
                )
            with timer.stage('reservoir_build'):
                params = build_reservoir_params(
                    n_qubits=config.reservoir.n_qubits,
                    depth=config.reservoir.depth,
                    readout=config.reservoir.readout,
                    backend=config.reservoir.backend,
                    rng=rng_reservoir,
                )
                circuit_hash = compute_circuit_hash(params)
            with timer.stage('feature_extraction'):
                X = extract_features(ds.u.astype(np.float64), params)
            with timer.stage('readout_training'):
                model = train_readout(
                    X[: ds.train_end],
                    ds.targets[: ds.train_end].astype(np.float64),
                    config.training.ridge_alphas,
                    config.training.cv_folds,
                )
            with timer.stage('evaluation'):
                y_pred = model.predict(X[ds.train_end :])
                acc = classification_accuracy(y_pred, ds.targets[ds.train_end :])
                primary_metric_name = 'accuracy'
                primary_metric_value = float(acc)

        elif task_name == 'narma':
            from qrc_thresher.metrics.scoring import nrmse
            from qrc_thresher.reservoirs.pennylane_qrc import (
                build_reservoir_params,
                compute_circuit_hash,
                extract_features,
                train_readout,
            )
            from qrc_thresher.tasks.narma10 import generate_narma10

            with timer.stage('task_generation'):
                ds = generate_narma10(
                    length=config.task.length,
                    train_frac=config.task.train_frac,
                    rng=rng_task,
                )
            with timer.stage('reservoir_build'):
                params = build_reservoir_params(
                    n_qubits=config.reservoir.n_qubits,
                    depth=config.reservoir.depth,
                    readout=config.reservoir.readout,
                    backend=config.reservoir.backend,
                    rng=rng_reservoir,
                )
                circuit_hash = compute_circuit_hash(params)
            with timer.stage('feature_extraction'):
                X = extract_features(ds.u, params)
            with timer.stage('readout_training'):
                model = train_readout(
                    X[: ds.train_end],
                    ds.targets[: ds.train_end],
                    config.training.ridge_alphas,
                    config.training.cv_folds,
                )
            with timer.stage('evaluation'):
                y_pred = model.predict(X[ds.train_end :])
                err = nrmse(y_pred, ds.targets[ds.train_end :])
                primary_metric_name = 'nrmse'
                primary_metric_value = float(err)

        else:
            raise ValueError(f'Unknown task: {task_name}')

        success = True

    except Exception as exc:
        failure_reason = str(exc)
        logger.error('Seed %d failed: %s', seed_index, exc)
        success = False

    timing = timer.to_dict()
    manifest = create_manifest(
        config_path=cfg_path,
        circuit_hash=circuit_hash,
        task_seed=task_seed,
        reservoir_seed=reservoir_seed,
        backend_device=config.reservoir.backend,
        runtime_per_stage_seconds=timing,
        entanglement_metric=entanglement_metric,
        success=success,
        failure_reason=failure_reason,
        artifact_paths=artifact_paths,
        task_name=task_name,
        primary_metric_name=primary_metric_name,
        primary_metric_value=primary_metric_value,
    )

    return manifest


class ParallelRunner:
    """Parallel benchmark execution engine.

    Uses ProcessPoolExecutor to run multiple seeds concurrently.
    Results are written to runs.csv with file locking for safety.
    """

    def __init__(
        self,
        config: AlphaLiteConfig,
        max_workers: int = 1,
    ) -> None:
        """Initialize parallel runner.

        Args:
            config: Experiment configuration
            max_workers: Number of parallel workers (default 1 for serial)
        """
        self.config = config
        self.max_workers = max_workers

    def run_seeds(
        self,
        task_name: str,
        n_seeds: Optional[int] = None,
        config_path: Optional[Path] = None,
    ) -> List[RunManifest]:
        """Run benchmark for multiple seeds in parallel.

        Args:
            task_name: Task to run ('stm', 'parity', 'narma')
            n_seeds: Number of seeds to run (default from config)
            config_path: Path to config file (for manifest records)

        Returns:
            List of RunManifest objects
        """
        if n_seeds is None:
            n_seeds = self.config.seeds.n_seeds

        config_dict = self.config.model_dump()
        if config_path is None:
            config_dict['_config_path'] = 'unknown'
        else:
            config_dict['_config_path'] = str(config_path)

        work_items = [
            (seed_idx, task_name, config_dict)
            for seed_idx in range(n_seeds)
        ]

        manifests: List[RunManifest] = []

        if self.max_workers == 1:
            for item in work_items:
                manifest = _run_single_seed(item)
                manifests.append(manifest)
                self._write_manifest(manifest)
        else:
            try:
                import filelock
            except ImportError:
                logger.warning(
                    'filelock not installed, using unsafe CSV writes. '
                    'Install with: pip install filelock'
                )
                filelock = None

            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(_run_single_seed, item): item[0]
                    for item in work_items
                }

                try:
                    from tqdm import tqdm
                except ImportError:
                    logger.warning(
                        'tqdm not installed, no progress bar. '
                        'Install with: pip install tqdm'
                    )
                    tqdm = None

                if tqdm is not None:
                    futures_iter = tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc=f'{task_name} seeds',
                        unit='seed',
                    )
                else:
                    futures_iter = as_completed(futures)

                for future in futures_iter:
                    try:
                        manifest = future.result()
                        manifests.append(manifest)
                        self._write_manifest_safe(manifest, filelock)
                    except Exception as exc:
                        logger.error('Worker future failed: %s', exc)

        for m in manifests:
            update_cumulative_compute(
                sum(m.runtime_per_stage_seconds.values()) if m.runtime_per_stage_seconds else 0.0
            )

        return manifests

    def _write_manifest(self, manifest: RunManifest) -> None:
        """Write a single manifest to CSV (serial mode)."""
        append_to_csv(manifest, _RUNS_CSV)

    def _write_manifest_safe(
        self,
        manifest: RunManifest,
        filelock_module: Optional[Any],
    ) -> None:
        """Write manifest with file locking (parallel mode).

        Args:
            manifest: RunManifest to write
            filelock_module: filelock module or None if not available
        """
        if filelock_module is not None:
            lock_path = str(_RUNS_LOCK)
            with filelock_module.FileLock(lock_path, timeout=30):
                append_to_csv(manifest, _RUNS_CSV)
        else:
            self._write_manifest(manifest)
