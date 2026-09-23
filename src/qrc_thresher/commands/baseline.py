"""Baseline run path (defect D8): run the enabled classical baselines on every seed pair.

Each run writes a seed-keyed manifest row to results/runs.csv. The rows keep the contract
that G3 and G4 read today: task_name 'esn' for STM (primary metric 'mc', memory capacity
summed over k = 0..K on the test rows) and 'esn_narma' for NARMA-10 (primary metric
'nrmse'). The gates themselves are rewritten in CP4. Every row records the search budget
(n_configs, n_validation_evals), and a JSON record of the search is written under
results/artifacts/esn/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import filelock
import numpy as np

from qrc_thresher.config import AlphaLiteConfig, load_config
from qrc_thresher.proof.run_manifest import (
    RunManifest,
    append_to_csv,
    create_manifest,
    update_cumulative_compute,
)

logger = logging.getLogger(__name__)

TASK_NAMES = {'stm': 'esn', 'narma': 'esn_narma'}
PRIMARY_METRICS = {'stm': 'mc', 'narma': 'nrmse'}
_NOT_IN_RUN_PATH = {
    'random_features': 'RKS joins the run path with its bandwidth fix (defect D13, CP4)',
    'gru': 'the GRU baseline is a stub',
}
_RUNS_CSV = Path('results') / 'runs.csv'
_ARTIFACT_DIR = Path('results') / 'artifacts' / 'esn'


def run_baselines(
    cfg: AlphaLiteConfig,
    task: str,
    config_path: Path,
    csv_path: Optional[Path] = None,
) -> List[RunManifest]:
    """Run every enabled baseline that has a run path, on every seed pair of the config.

    Args:
        cfg: Experiment config.
        task: 'stm' or 'narma'.
        config_path: Path recorded in each manifest.
        csv_path: runs.csv to append to (default results/runs.csv).

    Returns:
        One manifest per (baseline, seed pair), in seed order.
    """
    if task not in TASK_NAMES:
        raise ValueError(f'baseline run path supports {sorted(TASK_NAMES)}; got {task!r}')
    csv_path = csv_path or _RUNS_CSV
    for name in cfg.baseline.enabled:
        if name != 'esn':
            reason = _NOT_IN_RUN_PATH.get(name, 'no run path')
            logger.warning('baseline %r skipped: %s', name, reason)
    manifests: List[RunManifest] = []
    if 'esn' not in cfg.baseline.enabled:
        return manifests
    for i in range(cfg.seeds.n_seeds):
        manifest = _run_esn(
            cfg, task, cfg.seeds.task_seed + i, cfg.seeds.reservoir_seed + i, Path(config_path)
        )
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with filelock.FileLock(str(csv_path.with_suffix('.csv.lock')), timeout=30):
            append_to_csv(manifest, csv_path)
        update_cumulative_compute(sum(manifest.runtime_per_stage_seconds.values()))
        manifests.append(manifest)
    return manifests


def baseline_handler(task: str, config_path: str) -> int:
    """CLI handler: run the enabled baselines. Exit 0 only if every run succeeded."""
    cfg_path = Path(config_path)
    manifests = run_baselines(load_config(cfg_path), task, cfg_path)
    n_ok = sum(1 for m in manifests if m.success)
    print(f'Baseline run ({task}): {n_ok}/{len(manifests)} successful')
    for m in manifests:
        value = m.primary_metric_value
        shown = f'{value:.4f}' if value is not None else m.failure_reason
        print(f'  {m.task_name} seeds {m.task_seed}/{m.reservoir_seed}: '
              f'{m.primary_metric_name or "metric"} = {shown}')
    return 0 if manifests and n_ok == len(manifests) else 1


def _run_esn(
    cfg: AlphaLiteConfig, task: str, task_seed: int, reservoir_seed: int, config_path: Path
) -> RunManifest:
    from qrc_thresher.baselines.esn import _n_features, draw_reservoir, fit_predict_esn, tune_esn
    from qrc_thresher.metrics.runtime import StageTimer
    from qrc_thresher.metrics.scoring import memory_capacity, nrmse

    timer = StageTimer()
    washout = cfg.baseline.esn_washout
    alphas, folds = cfg.training.ridge_alphas, cfg.training.cv_folds
    n_units = _n_features(cfg.reservoir.n_qubits, cfg.reservoir.readout)
    record: dict = {'task': task, 'task_seed': task_seed, 'reservoir_seed': reservoir_seed,
                    'n_units': n_units, 'washout': washout}
    weight_hash, value, failure = 'n/a', None, None
    n_configs = n_evals = None
    try:
        if cfg.baseline.esn_grid is None:
            raise ValueError('baseline.esn_grid is required to run the ESN baseline')
        grid = cfg.baseline.esn_grid.model_dump()
        record['grid'] = grid
        with timer.stage('task_generation'):
            ds = _task_data(cfg, task, task_seed)
        with timer.stage('reservoir_build'):
            draw = draw_reservoir(n_units, reservoir_seed)
        with timer.stage('hyperparameter_search'):
            search = tune_esn(ds.u, ds.targets, ds.train_end, draw, grid, washout, alphas,
                              folds, task)
        n_configs, n_evals = search.n_configs, search.n_validation_evals
        with timer.stage('readout_training'):
            pred, deployed, _ = fit_predict_esn(ds.u, ds.targets, ds.train_end, draw,
                                                search.best, washout, alphas, folds)
        weight_hash = deployed.weight_hash()
        if weight_hash != search.best_hash:
            raise RuntimeError('the deployed ESN is not the validated one (weight hashes differ)')
        with timer.stage('evaluation'):
            y_test = ds.targets[ds.train_end:]
            if task == 'stm':
                value = float(memory_capacity(pred, y_test))
            else:
                value = float(nrmse(np.ravel(pred), np.ravel(y_test)))
        record.update({
            'best': search.best.__dict__, 'validated_hash': search.best_hash,
            'deployed_hash': weight_hash, 'score_name': search.score_name,
            'configs': search.configs, PRIMARY_METRICS[task]: value,
        })
    except Exception as exc:
        failure = str(exc)
        logger.error('ESN baseline failed for seeds %d/%d: %s', task_seed, reservoir_seed, exc)

    manifest = create_manifest(
        config_path=config_path,
        circuit_hash=weight_hash,
        task_seed=task_seed,
        reservoir_seed=reservoir_seed,
        backend_device='numpy_esn',
        runtime_per_stage_seconds=timer.to_dict(),
        entanglement_metric=None,
        success=failure is None,
        failure_reason=failure,
        artifact_paths=[],
        task_name=TASK_NAMES[task],
        primary_metric_name=PRIMARY_METRICS[task] if failure is None else '',
        primary_metric_value=value,
        measurement_model=cfg.measurement.model,
        n_configs=n_configs,
        n_validation_evals=n_evals,
    )
    record.update({'run_id': manifest.run_id, 'n_configs': n_configs,
                   'n_validation_evals': n_evals, 'failure_reason': failure})
    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = _ARTIFACT_DIR / f'{manifest.run_id}.json'
    artifact.write_text(json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')
    manifest.artifact_paths.append(artifact.as_posix())
    return manifest


def _task_data(cfg: AlphaLiteConfig, task: str, task_seed: int):
    rng = np.random.default_rng(task_seed)
    if task == 'stm':
        from qrc_thresher.tasks.stm import generate_stm

        return generate_stm(length=cfg.task.length, delay_max=cfg.task.delay_max or 20,
                            train_frac=cfg.task.train_frac, rng=rng)
    from qrc_thresher.tasks.narma10 import generate_narma10

    return generate_narma10(length=cfg.task.length, train_frac=cfg.task.train_frac, rng=rng)
