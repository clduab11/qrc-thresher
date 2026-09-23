"""Ablation command: matched ablations on every seed pair of the config (D010, defect D12).

phase_random, no_entangle and haar inherit the reservoir's per-pair reservoir_seed, readout,
window and re-upload schedule; only the tested factor changes. random_features (RKS) gets the
reservoir's feature count F and its own per-pair stream. One manifest row is written per seed
pair, under the task name ``ablation:<name>``, with a circuit hash that identifies the
simulated circuit (or the RKS draw). Training and scoring rows are unchanged (D16, CP4).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger('qrc_thresher')

ABLATION_NAMES = ('phase_random', 'no_entangle', 'haar', 'random_features')


def ablation_handler(name: str, config_path: str) -> int:
    """Run ablation ``name`` on every seed pair of the config. Returns exit code."""
    from qrc_thresher.config import load_config

    cfg_path = Path(config_path)
    cfg = load_config(cfg_path)
    if name not in ABLATION_NAMES:
        raise ValueError(f'unknown ablation {name!r}; choose from {list(ABLATION_NAMES)}')
    pairs = [
        (cfg.seeds.task_seed + i, cfg.seeds.reservoir_seed + i) for i in range(cfg.seeds.n_seeds)
    ]
    ok = True
    for task_seed, reservoir_seed in pairs:
        ok &= _run_one(cfg, cfg_path, name, task_seed, reservoir_seed)
    return 0 if ok else 1


def _run_one(cfg, cfg_path: Path, name: str, task_seed: int, reservoir_seed: int) -> bool:
    from qrc_thresher.metrics.runtime import StageTimer
    from qrc_thresher.proof.run_manifest import (
        append_to_csv,
        create_manifest,
        update_cumulative_compute,
    )
    from qrc_thresher.reservoirs.pennylane_qrc import train_readout

    timer = StageTimer()
    success = False
    failure_reason: Optional[str] = None
    circuit_hash = f'ablation:{name}'
    primary_metric_name = ''
    primary_metric_value: Optional[float] = None
    task_name = cfg.task.name

    try:
        with timer.stage('task_generation'):
            ds = _task_data(cfg, task_name, np.random.default_rng(task_seed))
        u = ds.u.astype(np.float64)

        with timer.stage('reservoir_build'):
            if name == 'random_features':
                from qrc_thresher.reservoirs.windowed_qrc import rks_circuit_hash, rks_from_config

                rks = rks_from_config(cfg, reservoir_seed)
                circuit_hash = rks_circuit_hash(rks)
            else:
                from qrc_thresher.reservoirs.windowed_qrc import reservoir_from_config

                reservoir = reservoir_from_config(cfg, reservoir_seed, ablation=name)
                circuit_hash = reservoir.circuit_hash

        with timer.stage('feature_extraction'):
            if name == 'random_features':
                from qrc_thresher.baselines.random_features import extract_rks_features

                X = extract_rks_features(u, rks)
            else:
                X = reservoir.features(u)

        with timer.stage('readout_training'):
            model = train_readout(
                X[: ds.train_end],
                np.asarray(ds.targets[: ds.train_end], dtype=np.float64),
                cfg.training.ridge_alphas,
                cfg.training.cv_folds,
            )
        with timer.stage('evaluation'):
            y_pred = model.predict(X[ds.train_end :])
            primary_metric_name, primary_metric_value = _score(
                task_name, y_pred, ds.targets[ds.train_end :]
            )
            print(
                f'Ablation "{name}" seeds {task_seed}/{reservoir_seed} '
                f'{task_name} {primary_metric_name}: {primary_metric_value:.4f}'
            )
        success = True
    except Exception as exc:
        failure_reason = str(exc)
        logger.error('Ablation %s failed for seeds %d/%d: %s', name, task_seed, reservoir_seed, exc)

    timing = timer.to_dict()
    manifest = create_manifest(
        config_path=cfg_path,
        circuit_hash=circuit_hash,
        task_seed=task_seed,
        reservoir_seed=reservoir_seed,
        backend_device=cfg.reservoir.backend if name != 'random_features' else 'numpy_rks',
        runtime_per_stage_seconds=timing,
        entanglement_metric=None,
        success=success,
        failure_reason=failure_reason,
        artifact_paths=[],
        task_name=f'ablation:{name}',
        primary_metric_name=primary_metric_name,
        primary_metric_value=primary_metric_value,
        measurement_model=cfg.measurement.model,
        n_configs=1,
        n_validation_evals=0,
    )
    append_to_csv(manifest)
    update_cumulative_compute(sum(timing.values()))
    return success


def _task_data(cfg, task_name: str, rng: np.random.Generator):
    if task_name == 'parity':
        from qrc_thresher.tasks.temporal_parity import generate_parity

        return generate_parity(
            length=cfg.task.length,
            window=cfg.task.parity_window or 3,
            train_frac=cfg.task.train_frac,
            rng=rng,
        )
    if task_name == 'narma':
        from qrc_thresher.tasks.narma10 import generate_narma10

        return generate_narma10(length=cfg.task.length, train_frac=cfg.task.train_frac, rng=rng)
    from qrc_thresher.tasks.stm import generate_stm

    return generate_stm(
        length=cfg.task.length,
        delay_max=cfg.task.delay_max or 20,
        train_frac=cfg.task.train_frac,
        rng=rng,
    )


def _score(task_name: str, y_pred: np.ndarray, y_true: np.ndarray):
    if task_name == 'parity':
        from qrc_thresher.metrics.scoring import classification_accuracy

        return 'accuracy', float(classification_accuracy(y_pred, y_true))
    if task_name == 'narma':
        from qrc_thresher.metrics.scoring import nrmse

        return 'nrmse', float(nrmse(y_pred, y_true))
    from qrc_thresher.metrics.scoring import memory_capacity

    return 'mc', float(memory_capacity(y_pred, y_true))


__all__ = ['ABLATION_NAMES', 'ablation_handler']
