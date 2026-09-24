"""Shared deployment and scoring for every writer of results/runs.csv (D011, D012, D013).

``qrc_deployments`` resolves what a QRC (or matched-ablation) run deploys for one seed pair:
the tuned design of the tuning record (design=tuned; the ablation of it, design=inherited),
or the untuned defaults (design=default at w = 2 and design=default_w1 at w = 1; PI ruling 1),
or, for a config without a tuning block, the config's own reservoir (design=default). Every
deployment carries the record's sweep_id and tuning_record_sha and its budget.

``fit_and_score`` is the one training-and-scoring path: the harness readout fitted on rows
[washout, train_end) and scored on [train_end, T) with the task's metric (STM: stm_memory with
mc_total and mc_k0 as secondary metrics; PI ruling 10).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from qrc_thresher.config import AlphaLiteConfig

DESIGN_CHOICES = ('tuned', 'default')
DEFAULT_QRC = {'depth': 3, 'encoding_scale': float(np.pi)}
DEFAULT_WINDOWS = ((2, 'default'), (1, 'default_w1'))  # PI ruling 1


@dataclass
class Deployment:
    """One thing to run for one seed pair."""

    design: str
    circuit_hash: str
    n_configs: int
    n_validation_evals: int
    sweep_id: str = ''
    tuning_record_sha: str = ''
    reservoir: object = None  # WindowedReservoir for QRC / ablation deployments
    details: Dict[str, object] = field(default_factory=dict)

    def features(self, u: np.ndarray) -> np.ndarray:
        return self.reservoir.features(np.asarray(u, dtype=np.float64))


def _record(cfg: AlphaLiteConfig, task: str, config_path: Path) -> Optional[dict]:
    """The tuning record when the config has a tuning block (raises if it is missing)."""
    if cfg.tuning is None:
        return None
    from qrc_thresher.tuning import load_record

    return load_record(Path(config_path), task)


def qrc_deployments(
    cfg: AlphaLiteConfig,
    task: str,
    reservoir_seed: int,
    task_seed: int,
    config_path: Path,
    *,
    design: str = 'tuned',
    design_task: Optional[str] = None,
    ablation: Optional[str] = None,
) -> List[Deployment]:
    """The QRC deployments of one seed pair (one for tuned; two for the defaults).

    Args:
        design: 'tuned' (design_<design_task or task>(pair) from the record) or 'default'.
        design_task: The task whose tuned design is deployed (default: ``task``); G1(b) runs
            design_STM on parity (D014).
        ablation: A matched ablation name; the deployment then inherits the design
            (design=inherited for a tuned design; the default labels otherwise).

    Raises:
        TuningRecordMissing: If the config has a tuning block and no record exists.
        ValueError: For an unknown design choice.
    """
    from qrc_thresher.reservoirs.windowed_qrc import reservoir_from_config
    from qrc_thresher.tuning import design_for_pair, tuned_reservoir

    if design not in DESIGN_CHOICES:
        raise ValueError(f'design must be one of {DESIGN_CHOICES}; got {design!r}')
    record = _record(cfg, design_task or task, Path(config_path))
    provenance = {}
    if record is not None:
        provenance = {'sweep_id': record['sweep_id'], 'tuning_record_sha': record['record_sha256']}

    if record is None:
        # No tuning block: the config's reservoir is the design, and it is the untuned default.
        reservoir = reservoir_from_config(cfg, reservoir_seed, ablation=ablation)
        label = 'inherited' if ablation else 'default'
        return [Deployment(design=label, circuit_hash=reservoir.circuit_hash, n_configs=1,
                           n_validation_evals=0, reservoir=reservoir,
                           details={'depth': cfg.reservoir.depth, 'window': cfg.reservoir.window,
                                    'encoding_scale': cfg.reservoir.encoding_scale})]

    if design == 'tuned':
        entry = design_for_pair(record, 'qrc', task_seed, reservoir_seed)
        reservoir = tuned_reservoir(cfg, entry, reservoir_seed, ablation=ablation)
        if ablation:
            label, budget = 'inherited', (1, 0)
        else:
            label, budget = 'tuned', (entry['n_configs'], entry['n_validation_evals'])
        return [Deployment(design=label, circuit_hash=reservoir.circuit_hash, n_configs=budget[0],
                           n_validation_evals=budget[1], reservoir=reservoir,
                           details={k: entry[k] for k in ('depth', 'window', 'encoding_scale')},
                           **provenance)]

    deployments = []
    for window, label in DEFAULT_WINDOWS:
        entry = {**DEFAULT_QRC, 'window': window}
        reservoir = tuned_reservoir(cfg, entry, reservoir_seed, ablation=ablation)
        deployments.append(Deployment(
            design='inherited' if ablation else label, circuit_hash=reservoir.circuit_hash,
            n_configs=1, n_validation_evals=0, reservoir=reservoir,
            details={**entry, 'default_label': label}, **provenance,
        ))
    return deployments


def score_task(task: str, y_pred: np.ndarray, y_true: np.ndarray):
    """(primary_metric_name, value, secondary_metrics) for a task's held-out predictions."""
    from qrc_thresher.metrics.scoring import (
        classification_accuracy,
        memory_capacity,
        nrmse,
        stm_memory,
    )

    if task == 'stm':
        y_pred = np.asarray(y_pred, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.float64)
        memory = float(stm_memory(y_pred, y_true))
        total = float(memory_capacity(y_pred, y_true))
        return 'stm_memory', memory, {'mc_total': total, 'mc_k0': total - memory}
    if task == 'parity':
        return 'accuracy', float(classification_accuracy(np.ravel(y_pred), np.ravel(y_true))), {}
    if task == 'narma':
        return 'nrmse', float(nrmse(np.ravel(y_pred), np.ravel(y_true))), {}
    raise ValueError(f'unknown task {task!r}')


def fit_and_score(X: np.ndarray, ds, task: str, cfg: AlphaLiteConfig):
    """Fit the harness readout on [washout, train_end) and score [train_end, T) (D012).

    Returns:
        (primary_metric_name, value, secondary_metrics, fitted model, predictions).
    """
    from qrc_thresher.readout import fit_ridge_cv

    washout = cfg.training.washout
    targets = np.asarray(ds.targets, dtype=np.float64)
    model = fit_ridge_cv(
        X[washout:ds.train_end], targets[washout:ds.train_end],
        cfg.training.ridge_alphas, cfg.training.cv_folds,
    )
    y_pred = model.predict(X[ds.train_end:])
    name, value, secondary = score_task(task, y_pred, targets[ds.train_end:])
    return name, value, secondary, model, y_pred


__all__ = [
    'DEFAULT_QRC',
    'DEFAULT_WINDOWS',
    'DESIGN_CHOICES',
    'Deployment',
    'fit_and_score',
    'qrc_deployments',
    'score_task',
]
