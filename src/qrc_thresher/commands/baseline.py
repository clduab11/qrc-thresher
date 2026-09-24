"""Baseline run path (defect D8; docs/DECISIONS.md D009, D011, D013).

`qrc-thresher baseline TASK --config FILE` runs every enabled classical baseline on every seed
pair of the config and writes one seed-keyed manifest row each, named by model and task
(``esn``, ``esn_parity``, ``esn_narma``, ``rks``, ``rks_parity``, ``rks_narma``; task_names.py).

With a tuning block the ESN and RKS configurations come from the tuning record (design=tuned,
the record's sweep_id and hash, its budget); the deployed model's hash must equal the validated
one. Without a tuning block the ESN is tuned in-line over ``baseline.esn_grid`` (D009;
design=tuned, empty sweep) and RKS runs at D010's default sigma = 1, d = 1 (design=default).
``--design default`` runs the ``esn_nonlinear`` preset with the bias (design=default). Every
model is fitted on rows [washout, train_end) and scored on the test rows with the task's metric
(STM: stm_memory, with mc_total and mc_k0 as secondary metrics). A JSON record of each run is
written under results/artifacts/baselines/.
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
from qrc_thresher.task_names import TASKS, baseline_task_name

logger = logging.getLogger(__name__)

_NOT_IN_RUN_PATH = {'gru': 'the GRU baseline is a stub (defect D20)'}
_RUNS_CSV = Path('results') / 'runs.csv'
_ARTIFACT_DIR = Path('results') / 'artifacts' / 'baselines'
DEFAULT_ESN_PRESET = 'esn_nonlinear'  # the untuned ESN default (D011)


def run_baselines(
    cfg: AlphaLiteConfig,
    task: str,
    config_path: Path,
    csv_path: Optional[Path] = None,
    design: str = 'tuned',
) -> List[RunManifest]:
    """Run every enabled baseline that has a run path, on every seed pair of the config.

    Args:
        cfg: Experiment config.
        task: 'stm', 'parity' or 'narma'.
        config_path: Path recorded in each manifest (and the tuning record's key).
        csv_path: runs.csv to append to (default results/runs.csv).
        design: 'tuned' or 'default' (the esn_nonlinear preset; RKS has no default row).

    Returns:
        One manifest per (baseline, seed pair), in seed order.
    """
    if task not in TASKS:
        raise ValueError(f'baseline run path supports {list(TASKS)}; got {task!r}')
    if design not in ('tuned', 'default'):
        raise ValueError(f"design must be 'tuned' or 'default'; got {design!r}")
    csv_path = csv_path or _RUNS_CSV
    record = None
    if cfg.tuning is not None:
        from qrc_thresher.tuning import load_record

        record = load_record(Path(config_path), task)
    manifests: List[RunManifest] = []
    for name in cfg.baseline.enabled:
        if name in _NOT_IN_RUN_PATH:
            logger.warning('baseline %r skipped: %s', name, _NOT_IN_RUN_PATH[name])
            continue
        if name == 'random_features' and design == 'default':
            logger.info('baseline random_features has no default row (D011); skipped')
            continue
        for i in range(cfg.seeds.n_seeds):
            pair = (cfg.seeds.task_seed + i, cfg.seeds.reservoir_seed + i)
            if name == 'esn':
                manifest = _run_esn(cfg, task, *pair, Path(config_path), record, design)
            else:
                manifest = _run_rks(cfg, task, *pair, Path(config_path), record)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with filelock.FileLock(str(csv_path.with_suffix('.csv.lock')), timeout=30):
                append_to_csv(manifest, csv_path)
            update_cumulative_compute(sum(manifest.runtime_per_stage_seconds.values()))
            manifests.append(manifest)
    return manifests


def baseline_handler(task: str, config_path: str, design: str = 'tuned') -> int:
    """CLI handler: run the enabled baselines. Exit 0 only if every run succeeded."""
    from qrc_thresher.proof.run_manifest import RunsCsvWriteError

    cfg_path = Path(config_path)
    try:
        manifests = run_baselines(load_config(cfg_path), task, cfg_path, design=design)
    except RunsCsvWriteError as exc:  # CP4b.1 item A3: a lost row fails the run, loudly
        print(f'baseline {task}: aborted; {exc}')
        return 1
    n_ok = sum(1 for m in manifests if m.success)
    print(f'Baseline run ({task}): {n_ok}/{len(manifests)} successful')
    for m in manifests:
        value = m.primary_metric_value
        shown = f'{value:.4f}' if value is not None else m.failure_reason
        print(f'  {m.task_name} seeds {m.task_seed}/{m.reservoir_seed} [{m.design}]: '
              f'{m.primary_metric_name or "metric"} = {shown}')
    return 0 if manifests and n_ok == len(manifests) else 1


def _finish(cfg, task, task_seed, reservoir_seed, config_path, *, task_name, backend, timer,
            circuit_hash, failure, metric_name, value, secondary, n_configs, n_evals, design,
            provenance, record_extra) -> RunManifest:
    manifest = create_manifest(
        config_path=config_path,
        circuit_hash=circuit_hash,
        task_seed=task_seed,
        reservoir_seed=reservoir_seed,
        backend_device=backend,
        runtime_per_stage_seconds=timer.to_dict(),
        entanglement_metric=None,
        success=failure is None,
        failure_reason=failure,
        artifact_paths=[],
        task_name=task_name,
        primary_metric_name=metric_name if failure is None else '',
        primary_metric_value=value,
        measurement_model=cfg.measurement.model,
        n_configs=n_configs,
        n_validation_evals=n_evals,
        secondary_metrics=secondary if failure is None else {},
        design=design,
        **provenance,
    )
    record = {
        'task': task, 'task_name': task_name, 'task_seed': task_seed,
        'reservoir_seed': reservoir_seed, 'design': design, 'run_id': manifest.run_id,
        'n_configs': n_configs, 'n_validation_evals': n_evals, 'failure_reason': failure,
        'washout': cfg.training.washout, **provenance, **record_extra,
    }
    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = _ARTIFACT_DIR / f'{manifest.run_id}.json'
    artifact.write_text(json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')
    manifest.artifact_paths.append(artifact.as_posix())
    return manifest


def _provenance(record) -> dict:
    if record is None:
        return {'sweep_id': '', 'tuning_record_sha': ''}
    return {'sweep_id': record['sweep_id'], 'tuning_record_sha': record['record_sha256']}


def _run_esn(cfg, task, task_seed, reservoir_seed, config_path, record, design) -> RunManifest:
    from qrc_thresher.baselines.esn import (
        ESN_PRESETS,
        ESNParams,
        build_esn,
        draw_reservoir,
        tune_esn,
    )
    from qrc_thresher.deploy import fit_and_score
    from qrc_thresher.metrics.runtime import StageTimer
    from qrc_thresher.reservoirs.windowed_qrc import n_features_from_config
    from qrc_thresher.tuning import design_for_pair, task_data

    timer = StageTimer()
    n_units = n_features_from_config(cfg)
    weight_hash, value, failure, metric_name, secondary = 'n/a', None, None, '', {}
    n_configs, n_evals = 1, 0
    extra: dict = {'n_units': n_units}
    label = design
    try:
        with timer.stage('task_generation'):
            ds = task_data(cfg, task, task_seed)
        with timer.stage('reservoir_build'):
            draw = draw_reservoir(n_units, reservoir_seed)
        if design == 'default':
            params = ESN_PRESETS[DEFAULT_ESN_PRESET]
            validated_hash = None
            extra['preset'] = DEFAULT_ESN_PRESET
        elif record is not None:
            entry = design_for_pair(record, 'esn', task_seed, reservoir_seed)
            params = ESNParams(**{k: entry[k] for k in ('spectral_radius', 'input_scaling',
                                                        'leak_rate')})
            validated_hash = entry['circuit_hash']
            n_configs, n_evals = entry['n_configs'], entry['n_validation_evals']
            label = 'tuned'
        else:
            if cfg.baseline.esn_grid is None:
                raise ValueError('baseline.esn_grid is required to tune the ESN in-line (D009)')
            grid = cfg.baseline.esn_grid.model_dump()
            with timer.stage('hyperparameter_search'):
                search = tune_esn(ds.u, ds.targets, ds.train_end, draw, grid,
                                  cfg.training.washout, cfg.training.ridge_alphas,
                                  cfg.training.cv_folds, task)
            params, validated_hash = search.best, search.best_hash
            n_configs, n_evals = search.n_configs, search.n_validation_evals
            extra.update({'grid': grid, 'configs': search.configs,
                          'score_name': search.score_name})
            label = 'tuned'
        extra['params'] = params.__dict__
        esn = build_esn(draw, params)
        weight_hash = esn.weight_hash()
        if validated_hash is not None and weight_hash != validated_hash:
            raise RuntimeError('the deployed ESN is not the validated one (weight hashes differ)')
        with timer.stage('feature_extraction'):
            X = esn.states(np.asarray(ds.u, dtype=np.float64))
        with timer.stage('readout_training'):
            metric_name, value, secondary, _, _ = fit_and_score(X, ds, task, cfg)
        extra.update({'validated_hash': validated_hash, 'deployed_hash': weight_hash,
                      metric_name: value})
    except Exception as exc:
        failure = str(exc)
        logger.error('ESN baseline failed for seeds %d/%d: %s', task_seed, reservoir_seed, exc)
    return _finish(cfg, task, task_seed, reservoir_seed, config_path,
                   task_name=baseline_task_name('esn', task), backend='numpy_esn', timer=timer,
                   circuit_hash=weight_hash, failure=failure, metric_name=metric_name,
                   value=value, secondary=secondary, n_configs=n_configs, n_evals=n_evals,
                   design=label, provenance=_provenance(record), record_extra=extra)


def _run_rks(cfg, task, task_seed, reservoir_seed, config_path, record) -> RunManifest:
    from qrc_thresher.baselines.random_features import extract_rks_features
    from qrc_thresher.deploy import fit_and_score
    from qrc_thresher.metrics.runtime import StageTimer
    from qrc_thresher.reservoirs.windowed_qrc import rks_circuit_hash, rks_from_config
    from qrc_thresher.tuning import design_for_pair, task_data

    timer = StageTimer()
    rks_hash, value, failure, metric_name, secondary = 'n/a', None, None, '', {}
    n_configs, n_evals = 1, 0
    label = 'default'
    extra: dict = {}
    try:
        with timer.stage('task_generation'):
            ds = task_data(cfg, task, task_seed)
        with timer.stage('reservoir_build'):
            if record is not None:
                entry = design_for_pair(record, 'rks', task_seed, reservoir_seed)
                params = rks_from_config(cfg, reservoir_seed, sigma=entry['sigma'],
                                         window=entry['window'])
                n_configs, n_evals = entry['n_configs'], entry['n_validation_evals']
                label = 'tuned'
                rks_hash = rks_circuit_hash(params)
                if rks_hash != entry['circuit_hash']:
                    raise RuntimeError('the deployed RKS is not the validated one (hashes differ)')
            else:
                params = rks_from_config(cfg, reservoir_seed)
                rks_hash = rks_circuit_hash(params)
        extra.update({'sigma': params.sigma, 'window': params.window,
                      'n_features': params.n_features})
        with timer.stage('feature_extraction'):
            X = extract_rks_features(np.asarray(ds.u, dtype=np.float64), params)
        with timer.stage('readout_training'):
            metric_name, value, secondary, _, _ = fit_and_score(X, ds, task, cfg)
        extra[metric_name] = value
    except Exception as exc:
        failure = str(exc)
        logger.error('RKS baseline failed for seeds %d/%d: %s', task_seed, reservoir_seed, exc)
    return _finish(cfg, task, task_seed, reservoir_seed, config_path,
                   task_name=baseline_task_name('random_features', task), backend='numpy_rks',
                   timer=timer, circuit_hash=rks_hash, failure=failure, metric_name=metric_name,
                   value=value, secondary=secondary, n_configs=n_configs, n_evals=n_evals,
                   design=label, provenance=_provenance(record), record_extra=extra)
