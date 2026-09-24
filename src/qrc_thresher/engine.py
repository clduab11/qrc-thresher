"""Parallel execution engine for qrc-thresher.

Uses ProcessPoolExecutor to run multiple benchmark seeds in parallel; the serial path shares
``run_pair`` with the run command. Results are written to runs.csv with file locking. Every
deployment follows docs/DECISIONS.md D011 (tuned designs from the tuning record, or the untuned
defaults), D012 (the washout) and D013 (schema 1.4 rows).
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import filelock
except ImportError:
    raise RuntimeError(
        "The 'filelock' package is required for safe CSV writes. "
        "Please install it (e.g., 'pip install filelock')."
    )

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


def _run_single_seed(args: Tuple[int, str, Dict[str, Any]]) -> List[RunManifest]:
    """Run one seed pair and return its manifests (one per deployment; D011).

    Args:
        args: Tuple of (seed_index, task_name, config_dict). The config dict may carry
            ``_config_path``, ``_design`` ('tuned' | 'default') and ``_design_task``.

    Returns:
        One RunManifest per deployment: one for a tuned (or config-fixed) design, two for the
        untuned defaults (design 'default' at w = 2 and 'default_w1' at w = 1).
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
    cfg_path = Path(config_dict.pop('_config_path', 'unknown'))
    design = config_dict.pop('_design', 'tuned')
    design_task = config_dict.pop('_design_task', None)

    config = AlphaLiteConfig.model_validate(config_dict)
    return run_pair(config, task_name, task_seed, reservoir_seed, cfg_path,
                    design=design, design_task=design_task)


def run_pair(
    config: AlphaLiteConfig,
    task_name: str,
    task_seed: int,
    reservoir_seed: int,
    cfg_path: Path,
    *,
    design: str = 'tuned',
    design_task: Optional[str] = None,
    log_artifacts: Optional[Any] = None,
) -> List[RunManifest]:
    """Run every QRC deployment of one seed pair on one task and return the manifests.

    The shared path of the engine and the run command (D012, D013): the task data, the
    deployments of ``deploy.qrc_deployments``, the harness readout on [washout, train_end),
    the task's metric on the test rows. ``log_artifacts(X, model, run_id) -> list[str]`` may
    save per-run artifacts and return their paths.
    """
    from qrc_thresher.deploy import fit_and_score, qrc_deployments
    from qrc_thresher.task_names import qrc_task_name
    from qrc_thresher.tuning import task_data

    timer = StageTimer()
    manifests: List[RunManifest] = []
    try:
        with timer.stage('task_generation'):
            ds = task_data(config, task_name, task_seed)
        with timer.stage('reservoir_build'):
            deployments = qrc_deployments(
                config, task_name, reservoir_seed, task_seed, cfg_path,
                design=design, design_task=design_task,
            )
    except Exception as exc:
        logger.error('Seed pair %d/%d failed: %s', task_seed, reservoir_seed, exc)
        manifests.append(_manifest(
            config, cfg_path, task_name, task_seed, reservoir_seed, timer.to_dict(),
            circuit_hash='n/a', success=False, failure_reason=str(exc), design='default',
        ))
        return manifests

    for dep in deployments:
        dep_timer = StageTimer()
        artifact_paths: List[str] = []
        success, failure_reason = False, None
        metric_name, value, secondary = '', None, {}
        try:
            with dep_timer.stage('feature_extraction'):
                X = dep.features(ds.u)
            with dep_timer.stage('readout_training'):
                metric_name, value, secondary, model, _ = fit_and_score(X, ds, task_name, config)
            if log_artifacts is not None:
                artifact_paths = list(log_artifacts(X, model) or [])
            success = True
        except Exception as exc:
            failure_reason = str(exc)
            logger.error('Seed pair %d/%d (%s) failed: %s', task_seed, reservoir_seed,
                         dep.design, exc)
        timing = {**timer.to_dict(), **dep_timer.to_dict()}
        manifests.append(_manifest(
            config, cfg_path, task_name, task_seed, reservoir_seed, timing,
            circuit_hash=dep.circuit_hash, success=success, failure_reason=failure_reason,
            design=dep.design, artifact_paths=artifact_paths,
            primary_metric_name=metric_name if success else '',
            primary_metric_value=value if success else None,
            secondary_metrics=secondary if success else {},
            n_configs=dep.n_configs, n_validation_evals=dep.n_validation_evals,
            sweep_id=dep.sweep_id, tuning_record_sha=dep.tuning_record_sha,
            task_label=qrc_task_name(task_name),
        ))
    return manifests


def _manifest(
    config: AlphaLiteConfig,
    cfg_path: Path,
    task_name: str,
    task_seed: int,
    reservoir_seed: int,
    timing: Dict[str, float],
    *,
    circuit_hash: str,
    success: bool,
    failure_reason: Optional[str],
    design: str,
    artifact_paths: Optional[List[str]] = None,
    primary_metric_name: str = '',
    primary_metric_value: Optional[float] = None,
    secondary_metrics: Optional[Dict[str, float]] = None,
    n_configs: int = 1,
    n_validation_evals: int = 0,
    sweep_id: str = '',
    tuning_record_sha: str = '',
    task_label: Optional[str] = None,
) -> RunManifest:
    return create_manifest(
        config_path=cfg_path,
        circuit_hash=circuit_hash,
        task_seed=task_seed,
        reservoir_seed=reservoir_seed,
        backend_device=config.reservoir.backend,
        runtime_per_stage_seconds=timing,
        entanglement_metric=None,
        success=success,
        failure_reason=failure_reason,
        artifact_paths=list(artifact_paths or []),
        task_name=task_label or task_name,
        primary_metric_name=primary_metric_name,
        primary_metric_value=primary_metric_value,
        measurement_model=config.measurement.model,
        n_configs=n_configs,
        n_validation_evals=n_validation_evals,
        secondary_metrics=secondary_metrics or {},
        design=design,
        sweep_id=sweep_id,
        tuning_record_sha=tuning_record_sha,
    )


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
        design: str = 'tuned',
        design_task: Optional[str] = None,
    ) -> List[RunManifest]:
        """Run benchmark for multiple seeds in parallel.

        Args:
            task_name: Task to run ('stm', 'parity', 'narma')
            n_seeds: Number of seeds to run (default from config)
            config_path: Path to config file (for manifest records and the tuning record)
            design: 'tuned' (the tuning record's design) or 'default' (D011)
            design_task: Task whose tuned design is deployed (default: task_name)

        Returns:
            List of RunManifest objects (one per deployment and seed pair)
        """
        if n_seeds is None:
            n_seeds = self.config.seeds.n_seeds

        config_dict = self.config.model_dump()
        config_dict['_config_path'] = 'unknown' if config_path is None else str(config_path)
        config_dict['_design'] = design
        config_dict['_design_task'] = design_task

        work_items = [
            (seed_idx, task_name, config_dict)
            for seed_idx in range(n_seeds)
        ]

        manifests: List[RunManifest] = []

        if self.max_workers == 1:
            for item in work_items:
                for manifest in _run_single_seed(item):
                    manifests.append(manifest)
                    self._write_manifest_safe(manifest, filelock)
        else:
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
                        for manifest in future.result():
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
            filelock_module: filelock module (strictly required)
        """
        lock_path = str(_RUNS_LOCK)
        with filelock_module.FileLock(lock_path, timeout=30):
            append_to_csv(manifest, _RUNS_CSV)
