"""Run command implementation (docs/DECISIONS.md D011, D012, D013).

`qrc-thresher run TASK --config FILE` runs every seed pair of the config, serial or parallel,
through the engine's shared ``run_pair``; there is no --seed. ``--design tuned`` (default)
deploys the tuning record's design_TASK(pair) (or the config's own reservoir when the config has
no tuning block); ``--design default`` deploys the untuned defaults; ``--design-task`` deploys
another task's tuned design (G1(b) runs design_STM on parity, D014).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.linear_model import RidgeCV

logger = logging.getLogger('qrc_thresher')


def run_handler(
    task: str,
    config_path: str,
    design: str = 'tuned',
    design_task: Optional[str] = None,
) -> int:
    """Handle the serial run command: every seed pair of the config. Returns exit code."""
    from qrc_thresher.config import load_config
    from qrc_thresher.db import ExperimentDB
    from qrc_thresher.engine import run_pair
    from qrc_thresher.proof.run_manifest import RunsCsvSchemaError, update_cumulative_compute
    from qrc_thresher.tuning import seed_pairs

    cfg_path = Path(config_path)
    cfg = load_config(cfg_path)

    def log_artifacts(X: np.ndarray, model) -> List[str]:
        if not cfg.proof.log_artifacts:
            return []
        from uuid import uuid4

        run_id = str(uuid4())
        paths = [p for p in (_save_features(X, run_id), _save_model(model, run_id)) if p]
        return paths

    all_ok = True
    for task_seed, reservoir_seed in seed_pairs(cfg):
        manifests = run_pair(
            cfg, task, task_seed, reservoir_seed, cfg_path, design=design,
            design_task=design_task, log_artifacts=log_artifacts,
        )
        for manifest in manifests:
            if manifest.success:
                print(
                    f'{task} seeds {task_seed}/{reservoir_seed} [{manifest.design}] '
                    f'{manifest.primary_metric_name}: {manifest.primary_metric_value:.4f}'
                )
            else:
                all_ok = False
                print(f'{task} seeds {task_seed}/{reservoir_seed} FAILED: '
                      f'{manifest.failure_reason}')
            try:
                db = ExperimentDB()
                db.insert(manifest)
                db.close()
            except RunsCsvSchemaError:
                raise
            except Exception as exc:
                logger.warning('Failed to insert into ExperimentDB: %s', exc)
            update_cumulative_compute(sum(manifest.runtime_per_stage_seconds.values()))
    return 0 if all_ok else 1


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
    workers: int,
    design: str = 'tuned',
    design_task: Optional[str] = None,
) -> int:
    """Handle parallel run command. Returns exit code."""
    from qrc_thresher.config import load_config
    from qrc_thresher.engine import ParallelRunner

    cfg_path = Path(config_path)
    cfg = load_config(cfg_path)

    runner = ParallelRunner(config=cfg, max_workers=workers)
    manifests = runner.run_seeds(
        task_name=task, n_seeds=cfg.seeds.n_seeds, config_path=cfg_path, design=design,
        design_task=design_task,
    )

    n_total = len(manifests)
    n_success = sum(1 for m in manifests if m.success)
    print(f'Parallel run complete: {n_success}/{n_total} successful (workers={workers})')

    return 0 if n_success == n_total else 1
