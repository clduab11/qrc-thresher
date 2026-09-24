"""Matched tuning under one budget (docs/DECISIONS.md D011; defects D5, D13 and the CP2/CP3
carry-overs).

One tuner for every tuned model: ``select_configuration`` scores every candidate feature matrix
on ``cv_folds`` contiguous validation blocks of the training rows after the washout with the
harness readout (``readout.fit_ridge_cv_batched``), one feature matrix per configuration; the
best mean wins, ties go to grid order; degenerate or non-finite configurations are flagged and
never selected; test rows are never touched. The selection metric per task is ``stm_memory``
(k >= 1), parity accuracy, or NARMA-10 NRMSE (lower is better).

``tune_config`` runs the tuner for every seed pair of a config with a ``tuning`` block and for
every tuned model (QRC over depth x window x encoding_scale, ESN over its D009 grid with the
D011 bias, RKS over sigma x window), and writes the tuning record
results/tuning/<config_hash>/<task>.json: one ``sweep_id``, ``selection_scope: train_cv``, the
seeds used, every grid point with its score, the chosen point per pair and the record's own
SHA-256 (PI ruling 3). ``run``, the engine, ``ablation`` and ``baseline`` deploy from it and
every row inherits ``sweep_id`` and ``tuning_record_sha``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from qrc_thresher.config import AlphaLiteConfig

logger = logging.getLogger(__name__)

TUNING_DIR = Path('results') / 'tuning'
RECORD_SCHEMA = 'tuning-record-1'
SELECTION_SCOPE = 'train_cv'
MODELS = ('qrc', 'esn', 'rks')
QRC_KEYS = ('depth', 'window', 'encoding_scale')
ESN_KEYS = ('spectral_radius', 'input_scaling', 'leak_rate')
RKS_KEYS = ('sigma', 'window')
HYPERPARAMETERS = {'qrc': QRC_KEYS, 'esn': ESN_KEYS, 'rks': RKS_KEYS}
Pair = Tuple[int, int]


class TuningRecordMissing(FileNotFoundError):
    """A config with a tuning block has no tuning record yet: run `qrc-thresher tune TASK`."""


def selection_metric(task: str):
    """(name, score(pred, y), higher_is_better) for a task (D011; one scoring function each)."""
    from qrc_thresher.metrics.scoring import classification_accuracy, nrmse, stm_memory

    if task == 'stm':
        return 'stm_memory', (lambda pred, y: stm_memory(pred, y)), True
    if task == 'parity':
        return 'accuracy', (
            lambda pred, y: classification_accuracy(np.ravel(pred), np.ravel(y))
        ), True
    if task == 'narma':
        return 'nrmse', (lambda pred, y: nrmse(np.ravel(pred), np.ravel(y))), False
    raise ValueError(f'unknown task for tuning: {task!r}')


@dataclass
class Candidate:
    """One grid point: its hyperparameters, its identifying hash and a feature builder."""

    hyperparameters: dict
    circuit_hash: str
    features: Callable[[], np.ndarray]


@dataclass
class Selection:
    """Outcome of ``select_configuration``."""

    records: List[dict]
    best_index: int
    n_configs: int
    n_validation_evals: int
    metric: str
    higher_is_better: bool
    scope: str = SELECTION_SCOPE
    cv_folds: int = 0
    rows: List[int] = field(default_factory=list)


def select_configuration(
    candidates: Sequence[Candidate],
    targets: np.ndarray,
    washout: int,
    ridge_alphas: Sequence[float],
    cv_folds: int,
    task: str,
    *,
    degenerate_errors: Tuple[type, ...] = (),
) -> Selection:
    """Score every candidate on the validation blocks and pick the best (D011).

    The routine only ever sees training rows: ``targets`` is ``targets[:train_end]`` and every
    candidate's feature matrix has the same number of rows. Test rows cannot reach it, which is
    what ``selection_scope: train_cv`` in the record states (PI ruling 3; CP4b.1 item A2).

    Args:
        candidates: The grid points; each builds its feature matrix once, shape (n_train, F).
        targets: Training targets, shape (n_train,) or (n_train, K).
        washout: Rows [0, washout) are dropped from selection.
        ridge_alphas: Harness readout penalties.
        cv_folds: Number of contiguous validation blocks (and the readout's inner folds).
        task: 'stm', 'parity' or 'narma' (the selection metric).
        degenerate_errors: Extra exception types that flag a configuration as degenerate.

    Raises:
        ValueError: If there are no candidates, too few training rows, or a candidate's
            feature matrix has a different number of rows than ``targets``.
        RuntimeError: If every configuration is degenerate.
    """
    from sklearn.model_selection import KFold

    from qrc_thresher.metrics.scoring import DegeneratePredictionError
    from qrc_thresher.readout import fit_ridge_cv_batched

    if not candidates:
        raise ValueError('select_configuration needs at least one candidate')
    metric, score, higher = selection_metric(task)
    targets = np.asarray(targets, dtype=np.float64)
    n_train = int(targets.shape[0])
    rows = np.arange(int(washout), n_train)
    if len(rows) < 2 * cv_folds:
        raise ValueError(f'too few training rows after the washout: {len(rows)}')
    blocks = list(KFold(n_splits=cv_folds).split(rows))
    flagged = (DegeneratePredictionError, *degenerate_errors)

    records: List[dict] = []
    for cand in candidates:
        record = {'score': None, 'degenerate': False, 'reason': None}
        try:
            X = np.asarray(cand.features(), dtype=np.float64)
            if X.ndim != 2 or X.shape[0] != n_train:
                raise ValueError(
                    f'features must have shape (n_train, F) with n_train = {n_train}; '
                    f'got {X.shape}'
                )
            if not np.all(np.isfinite(X)):
                raise DegeneratePredictionError('features contain non-finite values')
            block_scores = []
            for fit_idx, val_idx in blocks:
                fit_rows, val_rows = rows[fit_idx], rows[val_idx]
                y_fit = targets[fit_rows].reshape(len(fit_rows), -1, 1)
                fit = fit_ridge_cv_batched(X[fit_rows], y_fit, ridge_alphas, cv_folds)
                pred = fit.predict(X[val_rows])[:, :, 0].reshape(targets[val_rows].shape)
                block_scores.append(float(score(pred, targets[val_rows])))
            value = float(np.mean(block_scores))
            if not np.isfinite(value):
                raise DegeneratePredictionError('non-finite validation score')
            record['score'] = value
        except flagged as exc:
            record.update({'degenerate': True, 'reason': str(exc)})
        records.append(record)

    scored = [i for i, r in enumerate(records) if not r['degenerate']]
    if not scored:
        raise RuntimeError('every configuration was degenerate; nothing to select')
    sign = 1.0 if higher else -1.0
    best = max(scored, key=lambda i: (sign * records[i]['score'], -i))
    return Selection(
        records=records,
        best_index=best,
        n_configs=len(candidates),
        n_validation_evals=len(candidates) * len(blocks),
        metric=metric,
        higher_is_better=higher,
        cv_folds=int(cv_folds),
        rows=[int(rows[0]), int(rows[-1]) + 1],
    )


# --- the record -------------------------------------------------------------------------------

def record_path(config_hash: str, task: str) -> Path:
    """results/tuning/<config_hash>/<task>.json, relative to the working directory."""
    return TUNING_DIR / str(config_hash) / f'{task}.json'


def record_sha256(record: dict) -> str:
    """SHA-256 of the canonical JSON of the record without its own hash field."""
    without = {k: v for k, v in record.items() if k != 'record_sha256'}
    canonical = json.dumps(without, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def pair_key(task_seed: int, reservoir_seed: int) -> str:
    return f'{int(task_seed)}/{int(reservoir_seed)}'


def seed_pairs(cfg: AlphaLiteConfig) -> List[Pair]:
    return [(cfg.seeds.task_seed + i, cfg.seeds.reservoir_seed + i)
            for i in range(cfg.seeds.n_seeds)]


def sweep_id_now() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def task_data(cfg: AlphaLiteConfig, task: str, task_seed: int):
    """The task dataset for a seed, as every writer generates it."""
    rng = np.random.default_rng(int(task_seed))
    if task == 'stm':
        from qrc_thresher.tasks.stm import generate_stm

        if cfg.task.delay_max is None:  # no silent K = 20 (CP4b.1 item C8)
            raise ValueError('task.delay_max must be set to run STM (the washout is checked '
                             'against it; D012)')
        return generate_stm(length=cfg.task.length, delay_max=cfg.task.delay_max,
                            train_frac=cfg.task.train_frac, rng=rng)
    if task == 'parity':
        from qrc_thresher.tasks.temporal_parity import generate_parity

        if cfg.task.parity_window is None:
            raise ValueError('task.parity_window must be set to run parity (the washout is '
                             'checked against it; D012)')
        return generate_parity(length=cfg.task.length, window=cfg.task.parity_window,
                               train_frac=cfg.task.train_frac, rng=rng)
    if task == 'narma':
        from qrc_thresher.tasks.narma10 import generate_narma10

        return generate_narma10(length=cfg.task.length, train_frac=cfg.task.train_frac, rng=rng)
    raise ValueError(f'unknown task {task!r}')


def tuned_config(cfg: AlphaLiteConfig, design: dict) -> AlphaLiteConfig:
    """A deep copy of ``cfg`` whose reservoir carries the design's depth, window and scale."""
    raw = copy.deepcopy(cfg.model_dump())
    raw['reservoir'].update(
        {'depth': int(design['depth']), 'window': int(design['window']),
         'encoding_scale': float(design['encoding_scale'])}
    )
    return AlphaLiteConfig.model_validate(raw)


def tuned_reservoir(
    cfg: AlphaLiteConfig, design: dict, reservoir_seed: int, ablation: Optional[str] = None
):
    """The tuned design (or its matched ablation) built through reservoir_from_config (D011)."""
    from qrc_thresher.reservoirs.windowed_qrc import reservoir_from_config

    return reservoir_from_config(tuned_config(cfg, design), int(reservoir_seed), ablation=ablation)


def _qrc_features(reservoir, u: np.ndarray) -> np.ndarray:
    """The reservoir's features; non-finite values flag the configuration, other errors raise."""
    from qrc_thresher.metrics.scoring import DegeneratePredictionError

    try:
        return reservoir.features(u)
    except ValueError as exc:
        if 'non-finite' in str(exc):
            raise DegeneratePredictionError(str(exc)) from exc
        raise


def _qrc_candidates(cfg: AlphaLiteConfig, reservoir_seed: int, u: np.ndarray) -> List[Candidate]:
    grid = cfg.tuning.qrc
    candidates = []
    for depth, window, scale in product(grid.depth, grid.window, grid.encoding_scale):
        design = {'depth': int(depth), 'window': int(window), 'encoding_scale': float(scale)}
        reservoir = tuned_reservoir(cfg, design, reservoir_seed)
        candidates.append(Candidate(
            hyperparameters=design,
            circuit_hash=reservoir.circuit_hash,
            features=(lambda r=reservoir: _qrc_features(r, u)),
        ))
    return candidates


def _esn_candidates(cfg: AlphaLiteConfig, reservoir_seed: int, u: np.ndarray) -> List[Candidate]:
    from qrc_thresher.baselines import esn as esn_module
    from qrc_thresher.reservoirs.windowed_qrc import n_features_from_config

    draw = esn_module.draw_reservoir(n_features_from_config(cfg), reservoir_seed)
    grid = cfg.tuning.esn
    candidates = []
    for rho, s, a in product(grid.spectral_radius, grid.input_scaling, grid.leak_rate):
        params = esn_module.ESNParams(
            spectral_radius=float(rho), input_scaling=float(s), leak_rate=float(a)
        )
        model = esn_module.build_esn(draw, params)
        candidates.append(Candidate(
            hyperparameters=params.__dict__.copy(),
            circuit_hash=model.weight_hash(),
            features=(lambda m=model: m.states(u)),
        ))
    return candidates


def _rks_candidates(cfg: AlphaLiteConfig, reservoir_seed: int, u: np.ndarray) -> List[Candidate]:
    from qrc_thresher.baselines.random_features import extract_rks_features
    from qrc_thresher.reservoirs.windowed_qrc import rks_circuit_hash, rks_from_config

    grid = cfg.tuning.rks
    candidates = []
    for sigma, window in product(grid.sigma, grid.window):
        params = rks_from_config(cfg, reservoir_seed, sigma=float(sigma), window=int(window))
        candidates.append(Candidate(
            hyperparameters={'sigma': float(sigma), 'window': int(window)},
            circuit_hash=rks_circuit_hash(params),
            features=(lambda p=params: extract_rks_features(u, p)),
        ))
    return candidates


_CANDIDATES = {'qrc': _qrc_candidates, 'esn': _esn_candidates, 'rks': _rks_candidates}


def _degenerate_errors(model: str) -> Tuple[type, ...]:
    if model == 'esn':
        from qrc_thresher.baselines.esn import NonFiniteStatesError

        return (NonFiniteStatesError,)
    return ()


def _entry(
    task_seed: int, reservoir_seed: int, candidates: Sequence[Candidate], sel: Selection
) -> dict:
    winner = candidates[sel.best_index]
    return {
        'task_seed': int(task_seed),
        'reservoir_seed': int(reservoir_seed),
        **winner.hyperparameters,
        'circuit_hash': winner.circuit_hash,
        'n_configs': sel.n_configs,
        'n_validation_evals': sel.n_validation_evals,
        'winner_index': sel.best_index,
        'configs': [
            {**c.hyperparameters, 'circuit_hash': c.circuit_hash, **r}
            for c, r in zip(candidates, sel.records)
        ],
    }


def tune_config(
    cfg: AlphaLiteConfig, task: str, config_path: Path, sweep_id: Optional[str] = None
) -> dict:
    """Run the D011 tuner for every seed pair and every model, and write the record.

    Only the training rows reach the tuner: the candidates are built on ``u[:train_end]`` and
    scored against ``targets[:train_end]`` (every reservoir here is causal, so these are the
    first train_end rows of the full feature matrix); the record's ``selection_scope`` states it.
    ``sweep_id`` stamps the record (default: a fresh stamp); ``tune_all`` shares one stamp
    across the three tasks (CP4b.1 item B6).

    Raises:
        ValueError: If the config has no tuning block or the task is unknown.
    """
    from qrc_thresher.proof.run_manifest import _config_hash

    if cfg.tuning is None:
        raise ValueError('tuning needs a `tuning` block in the config (docs/DECISIONS.md D011)')
    metric, _, higher = selection_metric(task)
    config_path = Path(config_path)
    pairs = seed_pairs(cfg)
    record: dict = {
        'schema': RECORD_SCHEMA,
        'config_hash': _config_hash(config_path),
        'config_path': config_path.as_posix(),
        'task': task,
        'sweep_id': sweep_id or sweep_id_now(),
        'washout': int(cfg.training.washout),
        'cv_folds': int(cfg.training.cv_folds),
        'selection_scope': SELECTION_SCOPE,
        'selection_rows': None,
        'seeds': [list(p) for p in pairs],
        'reservoir': {
            'n_qubits': cfg.reservoir.n_qubits,
            'readout': cfg.reservoir.readout,
            'backend': cfg.reservoir.backend,
        },
        'selection_metric': {'name': metric, 'higher_is_better': higher},
        'grids': cfg.tuning.model_dump(),
        **{model: {} for model in MODELS},
    }
    for task_seed, reservoir_seed in pairs:
        ds = task_data(cfg, task, task_seed)
        train_end = int(ds.train_end)
        u_train = np.asarray(ds.u, dtype=np.float64)[:train_end]
        targets_train = np.asarray(ds.targets, dtype=np.float64)[:train_end]
        for model in MODELS:
            candidates = _CANDIDATES[model](cfg, reservoir_seed, u_train)
            sel = select_configuration(
                candidates, targets_train, cfg.training.washout,
                cfg.training.ridge_alphas, cfg.training.cv_folds, task,
                degenerate_errors=_degenerate_errors(model),
            )
            record['selection_scope'] = sel.scope
            record['selection_rows'] = sel.rows  # [washout, train_end): the only rows seen
            record[model][pair_key(task_seed, reservoir_seed)] = _entry(
                task_seed, reservoir_seed, candidates, sel
            )
            logger.info('tune %s %s pair %d/%d: %s', task, model, task_seed, reservoir_seed,
                        record[model][pair_key(task_seed, reservoir_seed)]['circuit_hash'][:12])
    record['record_sha256'] = record_sha256(record)
    path = record_path(record['config_hash'], task)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')
    logger.info('tuning record written to %s', path)
    return record


def tune_all(cfg: AlphaLiteConfig, config_path: Path) -> Dict[str, dict]:
    """Tune stm, parity and narma under one ``sweep_id`` (CP4b.1 item B6; supersedes the
    per-record stamps of D015 item 1 for the registered run). Each record keeps its own
    ``record_sha256``."""
    from qrc_thresher.task_names import TASKS

    stamp = sweep_id_now()
    return {task: tune_config(cfg, task, config_path, sweep_id=stamp) for task in TASKS}


def load_record(config_path: Path, task: str) -> dict:
    """The tuning record of a config file and task, from the working directory.

    Raises:
        TuningRecordMissing: If the record does not exist.
        ValueError: If the record's own hash does not match its content.
    """
    from qrc_thresher.proof.run_manifest import _config_hash

    path = record_path(_config_hash(Path(config_path)), task)
    if not path.exists():
        raise TuningRecordMissing(
            f'no tuning record at {path.as_posix()} for {Path(config_path).as_posix()} / {task}; '
            f'run `qrc-thresher tune {task} --config {Path(config_path).as_posix()}` first (D011)'
        )
    record = json.loads(path.read_text(encoding='utf-8'))
    if record.get('record_sha256') != record_sha256(record):
        raise ValueError(f'tuning record {path.as_posix()} does not match its own record_sha256')
    return record


def design_for_pair(record: dict, model: str, task_seed: int, reservoir_seed: int) -> dict:
    """The record's chosen configuration for a model and seed pair.

    Raises:
        KeyError: Naming the seeds, if the pair is not in the record.
    """
    key = pair_key(task_seed, reservoir_seed)
    try:
        return record[model][key]
    except KeyError:
        raise KeyError(
            f'tuning record ({record.get("task")}, sweep {record.get("sweep_id")}) has no {model} '
            f'design for seed pair {key} (task_seed {task_seed}, reservoir_seed {reservoir_seed})'
        ) from None


def tune_handler(task: Optional[str], config_path: str) -> int:
    """CLI handler for `qrc-thresher tune [TASK] --config FILE`. Returns exit code.

    Without TASK, the three tasks are tuned in one invocation under one sweep_id (item B6);
    with TASK, that record alone is (re)written under its own stamp.
    """
    from qrc_thresher.config import load_config

    cfg_path = Path(config_path)
    cfg = load_config(cfg_path)
    records = tune_all(cfg, cfg_path) if task is None else {task: tune_config(cfg, task, cfg_path)}
    for name, record in records.items():
        path = record_path(record['config_hash'], name)
        print(f'tuning record: {path.as_posix()} (sweep {record["sweep_id"]})')
        for model in MODELS:
            for key, entry in record[model].items():
                chosen = {k: entry[k] for k in HYPERPARAMETERS[model]}
                print(f'  {model} {key}: {chosen} ({entry["n_configs"]} configurations, '
                      f'{entry["n_validation_evals"]} validation fits)')
    return 0


__all__ = [
    'Candidate',
    'HYPERPARAMETERS',
    'MODELS',
    'RECORD_SCHEMA',
    'SELECTION_SCOPE',
    'Selection',
    'TUNING_DIR',
    'TuningRecordMissing',
    'design_for_pair',
    'load_record',
    'pair_key',
    'record_path',
    'record_sha256',
    'seed_pairs',
    'select_configuration',
    'selection_metric',
    'sweep_id_now',
    'task_data',
    'tune_all',
    'tune_config',
    'tune_handler',
    'tuned_config',
    'tuned_reservoir',
]
