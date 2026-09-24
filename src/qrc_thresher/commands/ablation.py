"""Ablation command: matched ablations on every seed pair of the config (D010, D011, D013).

`qrc-thresher ablation NAME TASK --config FILE`: phase_random, no_entangle and haar inherit the
reservoir's per-pair reservoir_seed, readout, window, encoding scale and re-upload schedule; only
the tested factor changes. Under a config with a tuning block the ablation inherits
design_TASK(pair) (``--design-task`` names another task's design; D014's G1(b)) and writes
design=inherited; ``--design default`` ablates the untuned defaults. One row per seed pair and
deployment is written under the task name ``ablation:<name>``, trained on rows
[washout, train_end) and scored on the test rows (D012). RKS is a baseline, not an ablation
(D011): see `qrc-thresher baseline`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from qrc_thresher.task_names import ABLATIONS as ABLATION_NAMES

logger = logging.getLogger('qrc_thresher')


def ablation_handler(
    name: str,
    task: str,
    config_path: str,
    design: str = 'tuned',
    design_task: Optional[str] = None,
) -> int:
    """Run ablation ``name`` of ``task`` on every seed pair of the config. Returns exit code.

    Every pair's design is resolved (record, pair, hash; CP4b.1 items A1, A4) before the first
    row; a failure there exits 1 with the message and writes nothing. A runs.csv that cannot be
    written exits 1 naming the path (item A3).
    """
    from qrc_thresher.config import load_config
    from qrc_thresher.deploy import resolve_deployments
    from qrc_thresher.proof.run_manifest import RunsCsvWriteError

    cfg_path = Path(config_path)
    cfg = load_config(cfg_path)
    if name not in ABLATION_NAMES:
        raise ValueError(f'unknown ablation {name!r}; choose from {list(ABLATION_NAMES)}')
    try:
        deployments = resolve_deployments(cfg, task, cfg_path, design=design,
                                          design_task=design_task, ablation=name)
    except Exception as exc:
        text = str(exc)
        if isinstance(exc, KeyError) and text[:1] in ('"', "'"):
            text = text[1:-1]
        print(f'ablation {name} {task}: nothing run; {text}')
        return 1
    ok = True
    for (task_seed, reservoir_seed), deps in deployments.items():
        try:
            manifests = _run_one(cfg, cfg_path, name, task, task_seed, reservoir_seed, deps)
        except RunsCsvWriteError as exc:
            print(f'ablation {name} {task} seeds {task_seed}/{reservoir_seed}: aborted; {exc}')
            return 1
        for manifest in manifests:
            ok &= bool(manifest.success)
    return 0 if ok else 1


def _run_one(cfg, cfg_path: Path, name: str, task: str, task_seed: int, reservoir_seed: int,
             deployments: List) -> List:
    from qrc_thresher.deploy import fit_and_score
    from qrc_thresher.metrics.runtime import StageTimer
    from qrc_thresher.proof.run_manifest import (
        append_to_csv,
        create_manifest,
        update_cumulative_compute,
    )
    from qrc_thresher.task_names import ablation_task_name
    from qrc_thresher.tuning import task_data

    timer = StageTimer()
    manifests = []
    with timer.stage('task_generation'):
        ds = task_data(cfg, task, task_seed)

    for dep in deployments:
        dep_timer = StageTimer()
        success, failure_reason = False, None
        metric_name, value, secondary = '', None, {}
        try:
            with dep_timer.stage('feature_extraction'):
                X = dep.features(ds.u)
            with dep_timer.stage('readout_training'):
                metric_name, value, secondary, _, _ = fit_and_score(X, ds, task, cfg)
            print(
                f'Ablation "{name}" seeds {task_seed}/{reservoir_seed} {task} '
                f'[{dep.details.get("default_label", dep.design)}] {metric_name}: {value:.4f}'
            )
            success = True
        except Exception as exc:
            failure_reason = str(exc)
            logger.error('Ablation %s failed for seeds %d/%d: %s', name, task_seed,
                         reservoir_seed, exc)
        timing = {**timer.to_dict(), **dep_timer.to_dict()}
        manifest = create_manifest(
            config_path=cfg_path,
            circuit_hash=dep.circuit_hash,
            task_seed=task_seed,
            reservoir_seed=reservoir_seed,
            backend_device=cfg.reservoir.backend,
            runtime_per_stage_seconds=timing,
            entanglement_metric=None,
            success=success,
            failure_reason=failure_reason,
            artifact_paths=[],
            task_name=ablation_task_name(name),
            primary_metric_name=metric_name if success else '',
            primary_metric_value=value if success else None,
            measurement_model=cfg.measurement.model,
            n_configs=1,
            n_validation_evals=0,
            secondary_metrics=secondary if success else {},
            design=dep.design,
            sweep_id=dep.sweep_id,
            tuning_record_sha=dep.tuning_record_sha,
        )
        append_to_csv(manifest)
        update_cumulative_compute(sum(timing.values()))
        manifests.append(manifest)
    return manifests


__all__ = ['ABLATION_NAMES', 'ablation_handler']
