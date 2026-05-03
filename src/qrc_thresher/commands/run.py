"""Run command implementation."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional
from uuid import uuid4

import numpy as np
from sklearn.linear_model import RidgeCV

logger = logging.getLogger('qrc_thresher')


def run_handler(task: str, config_path: str, seed: Optional[int]) -> int:
    """Handle run command. Returns exit code."""
    from qrc_thresher.config import load_config
    from qrc_thresher.db import ExperimentDB
    from qrc_thresher.metrics.runtime import StageTimer
    from qrc_thresher.proof.run_manifest import (
        create_manifest,
        update_cumulative_compute,
    )

    cfg_path = Path(config_path)
    cfg = load_config(cfg_path)

    task_seed = seed if seed is not None else cfg.seeds.task_seed
    reservoir_seed = cfg.seeds.reservoir_seed

    timer = StageTimer()
    circuit_hash = 'n/a'
    artifact_paths: list[str] = []
    success = False
    failure_reason = None
    primary_metric_name = ''
    primary_metric_value: Optional[float] = None

    run_id = str(uuid4())

    try:
        rng_task = np.random.default_rng(task_seed)
        rng_reservoir = np.random.default_rng(reservoir_seed)

        if task == 'stm':
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
                circuit_hash = compute_circuit_hash(params)

            with timer.stage('feature_extraction'):
                X = extract_features(ds.u, params)

            if cfg.proof.log_artifacts:
                features_path = _save_features(X, run_id)
                if features_path:
                    artifact_paths.append(features_path)

            with timer.stage('readout_training'):
                model = train_readout(
                    X[: ds.train_end],
                    ds.targets[: ds.train_end],
                    cfg.training.ridge_alphas,
                    cfg.training.cv_folds,
                )

            if cfg.proof.log_artifacts:
                model_path = _save_model(model, run_id)
                if model_path:
                    artifact_paths.append(model_path)

            with timer.stage('evaluation'):
                y_pred = model.predict(X[ds.train_end :])
                mc = memory_capacity(y_pred, ds.targets[ds.train_end :])
                primary_metric_name = 'mc'
                primary_metric_value = float(mc)
                print(f'STM Memory Capacity: {mc:.4f}')

        elif task == 'parity':
            from qrc_thresher.metrics.scoring import classification_accuracy
            from qrc_thresher.reservoirs.pennylane_qrc import (
                build_reservoir_params,
                compute_circuit_hash,
                extract_features,
                train_readout,
            )
            from qrc_thresher.tasks.temporal_parity import generate_parity

            window = cfg.task.parity_window or 3
            with timer.stage('task_generation'):
                ds = generate_parity(
                    length=cfg.task.length,
                    window=window,
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
                circuit_hash = compute_circuit_hash(params)
            with timer.stage('feature_extraction'):
                X = extract_features(ds.u.astype(np.float64), params)

            if cfg.proof.log_artifacts:
                features_path = _save_features(X, run_id)
                if features_path:
                    artifact_paths.append(features_path)

            with timer.stage('readout_training'):
                model = train_readout(
                    X[: ds.train_end],
                    ds.targets[: ds.train_end].astype(np.float64),
                    cfg.training.ridge_alphas,
                    cfg.training.cv_folds,
                )

            if cfg.proof.log_artifacts:
                model_path = _save_model(model, run_id)
                if model_path:
                    artifact_paths.append(model_path)

            with timer.stage('evaluation'):
                y_pred = model.predict(X[ds.train_end :])
                acc = classification_accuracy(y_pred, ds.targets[ds.train_end :])
                primary_metric_name = 'accuracy'
                primary_metric_value = float(acc)
                print(f'Parity accuracy (window={window}): {acc:.4f}')

        else:  # narma
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
                    length=cfg.task.length,
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
                circuit_hash = compute_circuit_hash(params)
            with timer.stage('feature_extraction'):
                X = extract_features(ds.u, params)

            if cfg.proof.log_artifacts:
                features_path = _save_features(X, run_id)
                if features_path:
                    artifact_paths.append(features_path)

            with timer.stage('readout_training'):
                model = train_readout(
                    X[: ds.train_end],
                    ds.targets[: ds.train_end],
                    cfg.training.ridge_alphas,
                    cfg.training.cv_folds,
                )

            if cfg.proof.log_artifacts:
                model_path = _save_model(model, run_id)
                if model_path:
                    artifact_paths.append(model_path)

            with timer.stage('evaluation'):
                y_pred = model.predict(X[ds.train_end :])
                err = nrmse(y_pred, ds.targets[ds.train_end :])
                primary_metric_name = 'nrmse'
                primary_metric_value = float(err)
                print(f'NARMA-10 NRMSE: {err:.4f}')

        success = True

    except Exception as exc:
        failure_reason = str(exc)
        logger.error('Run failed: %s', exc)
        success = False

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
        artifact_paths=artifact_paths,
        task_name=task,
        primary_metric_name=primary_metric_name,
        primary_metric_value=primary_metric_value,
    )
    manifest.run_id = run_id

    try:
        db = ExperimentDB()
        db.insert(manifest)
        db.close()
    except Exception as exc:
        logger.warning('Failed to insert into ExperimentDB: %s', exc)
    total_seconds = sum(timing.values())
    update_cumulative_compute(total_seconds)

    return 1 if not success else 0


def _save_features(X: np.ndarray, run_id: str) -> Optional[str]:
    """Save feature matrix to npz file.

    Args:
        X: Feature matrix of shape (T, F).
        run_id: Unique run identifier.

    Returns:
        Relative path to saved file, or None on failure.
    """
    try:
        features_dir = Path('results') / 'artifacts' / 'features'
        features_dir.mkdir(parents=True, exist_ok=True)
        path = features_dir / f'{run_id}.npz'
        np.savez(path, X=X)
        logger.info('Feature matrix saved to %s', path)
        return str(path)
    except Exception as exc:
        logger.warning('Failed to save feature matrix: %s', exc)
        return None


def _save_model(model: RidgeCV, run_id: str) -> Optional[str]:
    """Save trained readout model to pkl file.

    Args:
        model: Fitted RidgeCV model.
        run_id: Unique run identifier.

    Returns:
        Relative path to saved file, or None on failure.
    """
    try:
        models_dir = Path('results') / 'artifacts' / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        path = models_dir / f'{run_id}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        logger.info('Readout model saved to %s', path)
        return str(path)
    except Exception as exc:
        logger.warning('Failed to save readout model: %s', exc)
        return None


def run_parallel_handler(
    task: str,
    config_path: str,
    seed: Optional[int],
    workers: int,
) -> int:
    """Handle parallel run command. Returns exit code."""
    from qrc_thresher.config import load_config
    from qrc_thresher.engine import ParallelRunner

    cfg_path = Path(config_path)
    cfg = load_config(cfg_path)

    if seed is not None:
        cfg.seeds.task_seed = seed

    if cfg.task.name != task:
        cfg.task.name = task

    runner = ParallelRunner(config=cfg, max_workers=workers)
    manifests = runner.run_seeds(task_name=task, n_seeds=cfg.seeds.n_seeds, config_path=cfg_path)

    n_total = len(manifests)
    n_success = sum(1 for m in manifests if m.success)
    print(f'Parallel run complete: {n_success}/{n_total} successful (workers={workers})')

    return 0 if n_success == n_total else 1
