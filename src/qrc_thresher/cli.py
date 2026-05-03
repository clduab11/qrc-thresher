"""CLI surface for qrc_thresher.

Commands:
  health   - Run all health checks.
  run      - Run a task benchmark.
  ablation - Run an ablation study.
  gate     - Evaluate a decision gate.
  plot     - Generate figures for a run.
  summary  - Aggregate run history into a markdown report.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click
import numpy as np

logger = logging.getLogger('qrc_thresher')


def _setup_logging(verbose: bool = False) -> None:
    """Configure root logger."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )


@click.group()
@click.option('--verbose', is_flag=True, default=False, help='Enable DEBUG logging.')
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """qrc_thresher: falsification-first quantum reservoir benchmark."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose


@cli.command('health')
@click.option(
    '--out-dir',
    default='results/health',
    show_default=True,
    help='Output directory for health report JSON.',
)
def health_cmd(out_dir: str) -> None:
    """Run all health checks and write a report JSON.

    Exit code 0 on full pass, 1 on any failure.
    """
    from qrc_thresher.proof.benchmark_health import run_benchmark_health
    from qrc_thresher.proof.source_health import run_source_health

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    source = run_source_health()
    bench = run_benchmark_health()

    overall = source['overall'] == 'PASS' and bench['overall'] == 'PASS'
    report = {
        'version': '1.0',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'checks': {
            'env': source['env']['status'],
            'imports': source['imports'],
            'git': source['git']['status'],
            'config': bench['checks'].get('config', 'FAIL'),
            'tasks': bench['checks'].get('tasks', 'FAIL'),
            'qrc_smoke': bench['checks'].get('qrc_smoke', 'FAIL'),
            'esn_smoke': bench['checks'].get('esn_smoke', 'FAIL'),
            'metrics': bench['checks'].get('metrics', 'FAIL'),
            'manifest': bench['checks'].get('manifest', 'FAIL'),
            'reproducibility': bench['checks'].get('reproducibility', 'FAIL'),
        },
        'details': {
            'source': source,
            'benchmark': bench['details'],
        },
        'overall': 'PASS' if overall else 'FAIL',
    }

    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    report_path = out_path / f'{ts}.json'
    with report_path.open('w') as f:
        json.dump(report, f, indent=2)

    click.echo(json.dumps(report, indent=2))
    sys.exit(0 if overall else 1)


@cli.command('run')
@click.argument('task', type=click.Choice(['stm', 'parity', 'narma']))
@click.option(
    '--config',
    'config_path',
    default='configs/alpha_lite.yaml',
    show_default=True,
    help='Path to YAML config file.',
)
@click.option('--seed', default=None, type=int, help='Override task seed from config.')
def run_cmd(task: str, config_path: str, seed: Optional[int]) -> None:
    """Run a task benchmark and write a run manifest."""
    from qrc_thresher.config import load_config
    from qrc_thresher.metrics.runtime import StageTimer
    from qrc_thresher.proof.run_manifest import (
        append_to_csv,
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

    try:
        rng_task = np.random.default_rng(task_seed)
        rng_reservoir = np.random.default_rng(reservoir_seed)

        if task == 'narma' and cfg.task.name != 'narma':
            raise NotImplementedError(
                'NARMA task is gated behind G3; set task.name=narma in config'
            )

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

            with timer.stage('readout_training'):
                model = train_readout(
                    X[: ds.train_end],
                    ds.targets[: ds.train_end],
                    cfg.training.ridge_alphas,
                    cfg.training.cv_folds,
                )

            with timer.stage('evaluation'):
                y_pred = model.predict(X[ds.train_end :])
                mc = memory_capacity(y_pred, ds.targets[ds.train_end :])
                primary_metric_name = 'mc'
                primary_metric_value = float(mc)
                click.echo(f'STM Memory Capacity: {mc:.4f}')

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
            with timer.stage('readout_training'):
                model = train_readout(
                    X[: ds.train_end],
                    ds.targets[: ds.train_end].astype(np.float64),
                    cfg.training.ridge_alphas,
                    cfg.training.cv_folds,
                )
            with timer.stage('evaluation'):
                y_pred = model.predict(X[ds.train_end :])
                acc = classification_accuracy(y_pred, ds.targets[ds.train_end :])
                primary_metric_name = 'accuracy'
                primary_metric_value = float(acc)
                click.echo(f'Parity accuracy (window={window}): {acc:.4f}')

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
            with timer.stage('readout_training'):
                model = train_readout(
                    X[: ds.train_end],
                    ds.targets[: ds.train_end],
                    cfg.training.ridge_alphas,
                    cfg.training.cv_folds,
                )
            with timer.stage('evaluation'):
                y_pred = model.predict(X[ds.train_end :])
                err = nrmse(y_pred, ds.targets[ds.train_end :])
                primary_metric_name = 'nrmse'
                primary_metric_value = float(err)
                click.echo(f'NARMA-10 NRMSE: {err:.4f}')

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
    append_to_csv(manifest)
    total_seconds = sum(timing.values())
    update_cumulative_compute(total_seconds)

    if not success:
        sys.exit(1)


@cli.command('ablation')
@click.argument(
    'name',
    type=click.Choice(['phase_random', 'no_entangle', 'random_features', 'haar']),
)
@click.option('--config', 'config_path', default='configs/alpha_lite.yaml')
@click.option('--seed', default=None, type=int)
def ablation_cmd(name: str, config_path: str, seed: Optional[int]) -> None:
    """Run an ablation study."""
    from qrc_thresher.config import load_config
    from qrc_thresher.metrics.runtime import StageTimer
    from qrc_thresher.proof.run_manifest import (
        append_to_csv,
        create_manifest,
        update_cumulative_compute,
    )

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

        from qrc_thresher.metrics.scoring import memory_capacity
        from qrc_thresher.reservoirs.pennylane_qrc import build_reservoir_params, train_readout
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
            mc = memory_capacity(y_pred, ds.targets[ds.train_end :])
            primary_metric_name = 'mc'
            primary_metric_value = float(mc)
            click.echo(f'Ablation "{name}" STM MC: {mc:.4f}')

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

    if not success:
        sys.exit(1)


@cli.command('gate')
@click.argument(
    'name',
    type=click.Choice(['G0', 'G0.5', 'G1', 'G2', 'G2.5', 'G3', 'G4', 'G5']),
)
def gate_cmd(name: str) -> None:
    """Evaluate a decision gate from results/runs.csv.

    Gates are machine-checkable kill-gates. Exit code:
        0 = PASS
        1 = FAIL
        2 = INSUFFICIENT_EVIDENCE (not enough data to decide)

    Each gate writes results/gates/<name>.json with the verdict, the
    contributing run_ids, and the numerical evidence so reviewers can audit
    the decision against the manifest.
    """
    import pandas as pd

    gates_dir = Path('results') / 'gates'
    gates_dir.mkdir(parents=True, exist_ok=True)

    # G0 is special: it reads the latest health report rather than runs.csv.
    if name == 'G0':
        result, evidence, run_ids = _evaluate_gate_g0()
        _write_gate_result(gates_dir, name, result, evidence, run_ids)
        click.echo(f'Gate {name}: {result}')
        sys.exit({'PASS': 0, 'FAIL': 1, 'INSUFFICIENT_EVIDENCE': 2}.get(result, 2))

    runs_csv = Path('results') / 'runs.csv'
    if not runs_csv.exists():
        click.echo(f'INSUFFICIENT_EVIDENCE: no runs.csv found at {runs_csv}')
        _write_gate_result(
            gates_dir,
            name,
            'INSUFFICIENT_EVIDENCE',
            {'message': f'{runs_csv} not found'},
            [],
        )
        sys.exit(2)

    df = pd.read_csv(runs_csv)
    successful = df[df['success'].astype(str).str.lower() == 'true'].copy()

    if name == 'G1':
        result, evidence, run_ids = _evaluate_gate_g1(successful)
    elif name == 'G2':
        result, evidence, run_ids = _evaluate_gate_g2(successful)
    elif name == 'G2.5':
        result, evidence, run_ids = _evaluate_gate_g25(successful)
    elif name in ('G0.5', 'G3', 'G4', 'G5'):
        # These gates require artifacts (cross-check residuals, paired ESN
        # comparison data, multi-cell sweeps) that the current CLI does not
        # yet collect into runs.csv. They emit INSUFFICIENT_EVIDENCE with a
        # descriptive message and a non-zero exit code so CI fails loudly
        # rather than silently passing.
        result = 'INSUFFICIENT_EVIDENCE'
        evidence = {
            'message': (
                f'{name} requires supplementary artifacts (paired baseline '
                f'runs, cross-check residuals) that are not yet logged in '
                f'runs.csv. Implement the corresponding sweep before '
                f'evaluating this gate.'
            ),
            'n_successful_runs': int(len(successful)),
        }
        run_ids = successful['run_id'].astype(str).tolist()
    else:
        result = 'INSUFFICIENT_EVIDENCE'
        evidence = {'message': f'Unknown gate: {name}'}
        run_ids = []

    _write_gate_result(gates_dir, name, result, evidence, run_ids)
    click.echo(f'Gate {name}: {result}')

    exit_codes = {'PASS': 0, 'FAIL': 1, 'INSUFFICIENT_EVIDENCE': 2}
    sys.exit(exit_codes.get(result, 2))


def _latest_health_report() -> Optional[dict]:
    """Return the JSON content of the most recent health report, or None."""
    health_dir = Path('results') / 'health'
    if not health_dir.exists():
        return None
    candidates = sorted(health_dir.glob('*.json'))
    if not candidates:
        return None
    try:
        with candidates[-1].open('r') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _evaluate_gate_g0() -> tuple[str, dict, list]:
    """G0 PASS iff the latest health report's overall is PASS."""
    report = _latest_health_report()
    if report is None:
        return (
            'INSUFFICIENT_EVIDENCE',
            {'message': 'No health report found. Run `qrc-thresher health` first.'},
            [],
        )
    overall = report.get('overall', 'FAIL')
    return (
        'PASS' if overall == 'PASS' else 'FAIL',
        {'overall': overall, 'timestamp_utc': report.get('timestamp_utc', '')},
        [],
    )


def _filter_task(df, task_name: str):  # type: ignore[no-untyped-def]
    """Return successful rows whose task_name matches (handles legacy rows)."""
    if 'task_name' in df.columns:
        return df[df['task_name'].astype(str) == task_name]
    return df.iloc[0:0]


def _metric_values(df, metric_name: str) -> list:  # type: ignore[no-untyped-def]
    """Pull primary_metric_value from rows whose primary_metric_name matches."""
    if 'primary_metric_name' not in df.columns or 'primary_metric_value' not in df.columns:
        return []
    sub = df[df['primary_metric_name'].astype(str) == metric_name]
    if sub.empty:
        return []
    import pandas as pd

    vals = pd.to_numeric(sub['primary_metric_value'], errors='coerce').dropna()
    return [float(v) for v in vals.tolist()]


def _evaluate_gate_g1(successful) -> tuple[str, dict, list]:  # type: ignore[no-untyped-def]
    """G1: STM MC > 1.0 AND >=20% above no_entangle ablation MC.

    Pre-registered minimum n_seeds is 5 (per BUILD_SPEC §15.3).
    """
    qrc = _filter_task(successful, 'stm')
    qrc_mcs = _metric_values(qrc, 'mc')
    abl = _filter_task(successful, 'ablation:no_entangle')
    abl_mcs = _metric_values(abl, 'mc')

    n_seeds = len(qrc_mcs)
    evidence = {
        'qrc_n_runs': n_seeds,
        'no_entangle_n_runs': len(abl_mcs),
        'qrc_mc_mean': float(sum(qrc_mcs) / n_seeds) if qrc_mcs else None,
        'no_entangle_mc_mean': (
            float(sum(abl_mcs) / len(abl_mcs)) if abl_mcs else None
        ),
    }
    run_ids = qrc['run_id'].astype(str).tolist() + abl['run_id'].astype(str).tolist()

    if n_seeds < 5 or not abl_mcs:
        evidence['message'] = (
            'Need >=5 STM runs and >=1 no_entangle ablation run with '
            'primary_metric_value=mc.'
        )
        return 'INSUFFICIENT_EVIDENCE', evidence, run_ids

    qrc_mean = evidence['qrc_mc_mean']
    abl_mean = evidence['no_entangle_mc_mean']
    margin_pct = (qrc_mean - abl_mean) / abl_mean if abl_mean and abl_mean > 0 else 0.0
    evidence['margin_pct'] = margin_pct
    if qrc_mean > 1.0 and margin_pct >= 0.20:
        return 'PASS', evidence, run_ids
    return 'FAIL', evidence, run_ids


def _evaluate_gate_g2(successful) -> tuple[str, dict, list]:  # type: ignore[no-untyped-def]
    """G2: parity accuracy > 70% AND random_features ablation accuracy < 60%."""
    qrc = _filter_task(successful, 'parity')
    qrc_accs = _metric_values(qrc, 'accuracy')
    abl = _filter_task(successful, 'ablation:random_features')
    # The random_features ablation runs against STM by default; for a parity
    # comparison we look up its accuracy if logged. Either accuracy (parity
    # ablation) or fall back to the captured value.
    abl_accs = _metric_values(abl, 'accuracy')

    n_seeds = len(qrc_accs)
    evidence = {
        'qrc_n_runs': n_seeds,
        'random_features_n_runs': len(abl_accs),
        'qrc_accuracy_mean': (
            float(sum(qrc_accs) / n_seeds) if qrc_accs else None
        ),
        'random_features_accuracy_mean': (
            float(sum(abl_accs) / len(abl_accs)) if abl_accs else None
        ),
    }
    run_ids = qrc['run_id'].astype(str).tolist() + abl['run_id'].astype(str).tolist()

    if n_seeds < 5:
        evidence['message'] = 'Need >=5 parity runs with primary_metric_value=accuracy.'
        return 'INSUFFICIENT_EVIDENCE', evidence, run_ids

    qrc_mean = evidence['qrc_accuracy_mean']
    abl_mean = evidence['random_features_accuracy_mean']
    if abl_mean is None:
        evidence['message'] = (
            'Need at least one parity-task random_features ablation run '
            'with accuracy logged.'
        )
        return 'INSUFFICIENT_EVIDENCE', evidence, run_ids

    if qrc_mean > 0.70 and abl_mean < 0.60:
        return 'PASS', evidence, run_ids
    return 'FAIL', evidence, run_ids


def _evaluate_gate_g25(successful) -> tuple[str, dict, list]:  # type: ignore[no-untyped-def]
    """G2.5: full QRC outperforms Haar-random ablation by >=1 SE on STM-MC or parity-acc."""
    import math

    qrc_stm = _metric_values(_filter_task(successful, 'stm'), 'mc')
    haar = _metric_values(_filter_task(successful, 'ablation:haar'), 'mc')

    evidence: dict = {
        'qrc_stm_n_runs': len(qrc_stm),
        'haar_n_runs': len(haar),
    }
    run_ids = (
        _filter_task(successful, 'stm')['run_id'].astype(str).tolist()
        + _filter_task(successful, 'ablation:haar')['run_id'].astype(str).tolist()
    )

    if len(qrc_stm) < 3 or len(haar) < 3:
        evidence['message'] = 'Need >=3 STM runs and >=3 Haar ablation runs.'
        return 'INSUFFICIENT_EVIDENCE', evidence, run_ids

    qrc_mean = sum(qrc_stm) / len(qrc_stm)
    haar_mean = sum(haar) / len(haar)
    qrc_var = sum((v - qrc_mean) ** 2 for v in qrc_stm) / max(len(qrc_stm) - 1, 1)
    haar_var = sum((v - haar_mean) ** 2 for v in haar) / max(len(haar) - 1, 1)
    pooled_se = math.sqrt(qrc_var / len(qrc_stm) + haar_var / len(haar))
    evidence['qrc_stm_mean'] = qrc_mean
    evidence['haar_stm_mean'] = haar_mean
    evidence['pooled_se'] = pooled_se
    evidence['delta'] = qrc_mean - haar_mean

    if pooled_se == 0:
        evidence['message'] = 'Zero pooled standard error; check inputs.'
        return 'INSUFFICIENT_EVIDENCE', evidence, run_ids

    if (qrc_mean - haar_mean) >= pooled_se:
        return 'PASS', evidence, run_ids
    return 'FAIL', evidence, run_ids


def _write_gate_result(
    gates_dir: Path,
    name: str,
    result: str,
    evidence: dict,
    run_ids: list,
) -> None:
    """Write gate result JSON to results/gates/<name>.json."""
    gate_file = gates_dir / f'{name}.json'
    data = {
        'gate': name,
        'result': result,
        'evidence': evidence,
        'run_ids': run_ids,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    }
    with gate_file.open('w') as f:
        json.dump(data, f, indent=2)


@cli.command('plot')
@click.argument('run_id')
@click.option('--out', 'out_path', default=None, help='Output directory.')
def plot_cmd(run_id: str, out_path: Optional[str]) -> None:
    """Generate figures for a run."""
    out_dir = Path(out_path) if out_path else Path('results') / 'figures' / run_id
    click.echo(f'Figures would be written to {out_dir}')
    click.echo('Use plot functions from qrc_thresher.viz.plots directly for now.')


@cli.command('summary')
@click.option('--phase', default='phase1', show_default=True)
def summary_cmd(phase: str) -> None:
    """Aggregate run history into a markdown report."""
    import pandas as pd

    summaries_dir = Path('results') / 'summaries'
    summaries_dir.mkdir(parents=True, exist_ok=True)
    runs_csv = Path('results') / 'runs.csv'

    if not runs_csv.exists():
        click.echo('No runs.csv found. Run some experiments first.')
        return

    df = pd.read_csv(runs_csv)
    n_total = len(df)
    n_success = int(df['success'].astype(str).str.lower().eq('true').sum())

    out_file = summaries_dir / f'{phase}_summary.md'
    with out_file.open('w') as f:
        f.write(f'# {phase} Summary\n\n')
        f.write(f'- Total runs: {n_total}\n')
        f.write(f'- Successful runs: {n_success}\n')
        f.write(f'- Failed runs: {n_total - n_success}\n\n')
        f.write('## Runs\n\n')
        f.write(df.to_markdown(index=False))
        f.write('\n')

    click.echo(f'Summary written to {out_file}')


def main() -> None:
    """Entry point for qrc_thresher CLI."""
    cli()


if __name__ == '__main__':
    main()
