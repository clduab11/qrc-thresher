"""Ablation command implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def ablation_handler(name: str, config_path: str, seed: Optional[int]) -> int:
    """Handle ablation command. Returns exit code."""
    import logging

    from qrc_thresher.config import load_config
    from qrc_thresher.metrics.runtime import StageTimer
    from qrc_thresher.proof.run_manifest import (
        append_to_csv,
        create_manifest,
        update_cumulative_compute,
    )

    logger = logging.getLogger('qrc_thresher')

    cfg_path = Path(config_path)
    cfg = load_config(cfg_path)
    task_seed = seed if seed is not None else cfg.seeds.task_seed
    reservoir_seed = cfg.seeds.reservoir_seed

    timer = StageTimer()
    success = False
    failure_reason = None
    circuit_hash = f'ablation:{name}'
    primary_metric_name = ''
    primary_metric_value: Optional[float] = None

    try:
        rng_task = np.random.default_rng(task_seed)
        rng_reservoir = np.random.default_rng(reservoir_seed)

        from qrc_thresher.reservoirs.pennylane_qrc import build_reservoir_params, train_readout

        task_name = cfg.task.name
        with timer.stage('task_generation'):
            if task_name == 'parity':
                from qrc_thresher.tasks.temporal_parity import generate_parity

                window = cfg.task.parity_window or 3
                ds = generate_parity(
                    length=cfg.task.length,
                    window=window,
                    train_frac=cfg.task.train_frac,
                    rng=rng_task,
                )
            elif task_name == 'narma':
                from qrc_thresher.tasks.narma10 import generate_narma10

                ds = generate_narma10(
                    length=cfg.task.length,
                    train_frac=cfg.task.train_frac,
                    rng=rng_task,
                )
            else:
                from qrc_thresher.tasks.stm import generate_stm

                ds = generate_stm(
                    length=cfg.task.length,
                    delay_max=cfg.task.delay_max or 20,
                    train_frac=cfg.task.train_frac,
                    rng=rng_task,
                )
        with timer.stage('reservoir_build'):
            params = build_reservoir_params(
                n_qubits=cfg.reservoir.n_qubits,
                depth=cfg.reservoir.depth,
                readout=cfg.reservoir.readout,
                backend=cfg.reservoir.backend,
                rng=rng_reservoir,
            )

        with timer.stage('feature_extraction'):
            if name == 'phase_random':
                from qrc_thresher.reservoirs.ablations import extract_features_phase_random

                X = extract_features_phase_random(
                    ds.u, params, rng=np.random.default_rng(reservoir_seed + 1)
                )
            elif name == 'no_entangle':
                from qrc_thresher.reservoirs.ablations import extract_features_no_entangle

                X = extract_features_no_entangle(ds.u, params)
            elif name == 'haar':
                from qrc_thresher.reservoirs.ablations import extract_features_haar

                X = extract_features_haar(
                    ds.u,
                    cfg.reservoir.n_qubits,
                    cfg.reservoir.backend,
                    rng=np.random.default_rng(reservoir_seed + 2),
                )
            else:  # random_features
                from qrc_thresher.baselines.random_features import (
                    build_rks_params,
                    extract_rks_features,
                )

                rks_params = build_rks_params(
                    n_features=cfg.reservoir.n_qubits, sigma=1.0, rng=rng_reservoir
                )
                X = extract_rks_features(ds.u, rks_params)

        with timer.stage('readout_training'):
            model = train_readout(
                X[: ds.train_end],
                ds.targets[: ds.train_end],
                cfg.training.ridge_alphas,
                cfg.training.cv_folds,
            )
        with timer.stage('evaluation'):
            y_pred = model.predict(X[ds.train_end :])
            if task_name == 'parity':
                from qrc_thresher.metrics.scoring import classification_accuracy

                acc = classification_accuracy(y_pred, ds.targets[ds.train_end :])
                primary_metric_name = 'accuracy'
                primary_metric_value = float(acc)
                print(f'Ablation "{name}" parity accuracy: {acc:.4f}')
            elif task_name == 'narma':
                from qrc_thresher.metrics.scoring import nrmse

                err = nrmse(y_pred, ds.targets[ds.train_end :])
                primary_metric_name = 'nrmse'
                primary_metric_value = float(err)
                print(f'Ablation "{name}" NARMA NRMSE: {err:.4f}')
            else:
                from qrc_thresher.metrics.scoring import memory_capacity

                mc = memory_capacity(y_pred, ds.targets[ds.train_end :])
                primary_metric_name = 'mc'
                primary_metric_value = float(mc)
                print(f'Ablation "{name}" STM MC: {mc:.4f}')

        success = True

    except Exception as exc:
        failure_reason = str(exc)
        logger.error('Ablation failed: %s', exc)

    timing = timer.to_dict()
    manifest = create_manifest(
        config_path=cfg_path,
        circuit_hash=circuit_hash,
        task_seed=task_seed,
        reservoir_seed=reservoir_seed,
        backend_device=cfg.reservoir.backend,
        runtime_per_stage_seconds=timing,
        entanglement_metric=None,
        success=success,
        failure_reason=failure_reason,
        artifact_paths=[],
        task_name=f'ablation:{name}',
        primary_metric_name=primary_metric_name,
        primary_metric_value=primary_metric_value,
    )
    append_to_csv(manifest)
    update_cumulative_compute(sum(timing.values()))

    return 1 if not success else 0
