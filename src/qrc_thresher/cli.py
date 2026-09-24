"""CLI surface for qrc_thresher - thin dispatcher.

Commands:
  health   - Run all health checks.
  tune     - Tune every model under the matched budget and write the tuning record (D011).
  run      - Run a task benchmark on every seed pair.
  ablation - Run a matched ablation of a task on every seed pair.
  baseline - Run the enabled classical baselines on every seed pair.
  gate     - Evaluate a decision gate (or the comparative family).
  plugins  - List registered plugins.
  perf     - Run lightweight performance benchmarks.
  noise-sweep - Run noise model scaffold sweep.
  plot     - Generate figures for a run.
  summary  - Aggregate run history into a markdown report.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

import click

from qrc_thresher.commands import (
    ablation_handler,
    baseline_handler,
    gate_handler,
    health_handler,
    noise_sweep_handler,
    perf_handler,
    plot_handler,
    plugins_handler,
    run_handler,
    summary_handler,
)

logger = logging.getLogger('qrc_thresher')


def _setup_logging(verbose: bool = False) -> None:
    """Configure root logger."""
    from qrc_thresher.observability import configure_logging

    configure_logging(verbose=verbose, json_logs=False)


@click.group()
@click.option('--verbose', is_flag=True, default=False, help='Enable DEBUG logging.')
@click.option('--json-logs', is_flag=True, default=False, help='Emit machine-parseable JSON logs.')
@click.option('--trace', is_flag=True, default=False, help='Enable OpenTelemetry console tracing.')
@click.pass_context
def cli(ctx: click.Context, verbose: bool, json_logs: bool, trace: bool) -> None:
    """qrc_thresher: falsification-first quantum reservoir benchmark."""
    from qrc_thresher.observability import configure_logging, configure_tracing

    configure_logging(verbose=verbose, json_logs=json_logs)
    if trace:
        configure_tracing(service_name='qrc_thresher')

    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['json_logs'] = json_logs
    ctx.obj['trace'] = trace


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
    sys.exit(health_handler(out_dir))


_TASKS = click.Choice(['stm', 'parity', 'narma'])
_DESIGN = click.option(
    '--design',
    default='tuned',
    show_default=True,
    type=click.Choice(['tuned', 'default']),
    help='tuned: the tuning record\'s design; default: the untuned defaults (D011).',
)
_DESIGN_TASK = click.option(
    '--design-task',
    'design_task',
    default=None,
    type=_TASKS,
    help='Deploy another task\'s tuned design (G1(b) runs design_STM on parity; D014).',
)


@cli.command('tune')
@click.argument('task', type=_TASKS)
@click.option(
    '--config',
    'config_path',
    default='configs/comparative.yaml',
    show_default=True,
    help='Config with a tuning block.',
)
def tune_cmd(task: str, config_path: str) -> None:
    """Tune the QRC, the ESN and RKS under one budget and write the tuning record (D011).

    Writes results/tuning/<config_hash>/<task>.json; run, ablation and baseline deploy from it.
    """
    from qrc_thresher.tuning import tune_handler

    sys.exit(tune_handler(task, config_path))


@cli.command('run')
@click.argument('task', type=_TASKS)
@click.option(
    '--config',
    'config_path',
    default='configs/alpha_lite.yaml',
    show_default=True,
    help='Path to YAML config file.',
)
@click.option('--workers', default=1, type=int, help='Number of parallel workers.')
@_DESIGN
@_DESIGN_TASK
def run_cmd(task: str, config_path: str, workers: int, design: str,
            design_task: Optional[str]) -> None:
    """Run a task benchmark on every seed pair of the config and write run manifests."""
    if workers > 1:
        from qrc_thresher.commands.run import run_parallel_handler

        sys.exit(run_parallel_handler(task, config_path, workers, design, design_task))
    else:
        sys.exit(run_handler(task, config_path, design, design_task))


@cli.command('ablation')
@click.argument('name', type=click.Choice(['phase_random', 'no_entangle', 'haar']))
@click.argument('task', type=_TASKS)
@click.option(
    '--config',
    'config_path',
    default='configs/alpha_lite.yaml',
    show_default=True,
    help='Path to YAML config file.',
)
@_DESIGN
@_DESIGN_TASK
def ablation_cmd(name: str, task: str, config_path: str, design: str,
                 design_task: Optional[str]) -> None:
    """Run a matched ablation of TASK on every seed pair of the config (D010, D011).

    The ablation inherits the design's seed, readout, window, encoding scale and re-upload
    schedule; only the tested factor changes. One manifest row is written per seed pair and
    deployment. RKS is a baseline: see `baseline`.
    """
    sys.exit(ablation_handler(name, task, config_path, design, design_task))


@cli.command('baseline')
@click.argument('task', type=_TASKS)
@click.option(
    '--config',
    'config_path',
    default='configs/alpha_lite.yaml',
    show_default=True,
    help='Path to YAML config file.',
)
@_DESIGN
def baseline_cmd(task: str, config_path: str, design: str) -> None:
    """Run the enabled classical baselines on every seed pair and write manifest rows.

    Rows are named by model and task (esn, esn_parity, esn_narma, rks, rks_parity, rks_narma)
    and record the search budget. Exit 0 only if every run succeeded.
    """
    sys.exit(baseline_handler(task, config_path, design))


@cli.command('gate')
@click.argument(
    'name',
    type=click.Choice(
        ['G0', 'G0.5', 'G0.7', 'family', 'G1', 'G2', 'G2.5', 'G3', 'G4', 'G5', 'G6', 'G7']
    ),
)
@click.option(
    '--config',
    'config_path',
    default='configs/alpha_lite.yaml',
    show_default=True,
    help='Experiment config: seed pairs, reservoir (window included) and readout.',
)
@click.option(
    '--model',
    default='pennylane_qrc',
    show_default=True,
    type=click.Choice(['pennylane_qrc', 'no_entangle', 'tuned_qrc', 'esn_linear', 'esn_nonlinear']),
    help='Model evaluated by G0.7 (other gates ignore it); no_entangle is the matched ablation, '
         'tuned_qrc the tuning record\'s design_STM (D011).',
)
@click.option(
    '--tuning-config',
    'tuning_config',
    default='configs/comparative.yaml',
    show_default=True,
    help='Config whose tuning record supplies design_STM for --model tuned_qrc.',
)
def gate_cmd(name: str, config_path: str, model: str, tuning_config: str) -> None:
    """Evaluate a decision gate.

    Gates are machine-checkable kill-gates. Exit code:
        0 = PASS
        1 = FAIL
        2 = INSUFFICIENT_EVIDENCE (not enough data to decide)

    G0, G0.5, G5, G6 and G7 write results/gates/<name>.json. G0.7 (memory sanity,
    pre-registered in configs/gates/G0.7.v1.yaml) writes a new timestamped JSON and
    forgetting-curve figure on every evaluation. `family` (and any member G1, G2, G2.5, G3, G4)
    evaluates the comparative family of configs/gates/COMPARATIVE.v1.yaml as a unit on the
    rows of --config (D013, D014), writes COMPARATIVE.v1.<stamp>.json plus one view per member,
    and exits with the named member's code.
    """
    sys.exit(gate_handler(name, model=model, config_path=config_path,
                          tuning_config=tuning_config))


@cli.command('plot')
@click.argument('run_id')
@click.option('--out', 'out_path', default=None, help='Output directory.')
def plot_cmd(run_id: str, out_path: Optional[str]) -> None:
    """Generate figures for a run."""
    sys.exit(plot_handler(run_id, out_path))


@cli.command('summary')
@click.option('--phase', default='phase1', show_default=True)
def summary_cmd(phase: str) -> None:
    """Aggregate run history into a markdown report."""
    sys.exit(summary_handler(phase))


@cli.command('plugins')
def plugins_cmd() -> None:
    """List discovered plugins by extension group."""
    sys.exit(plugins_handler())


@cli.command('perf')
@click.option('--iterations', default=5, show_default=True, type=int)
def perf_cmd(iterations: int) -> None:
    """Run lightweight performance profile benchmarks."""
    sys.exit(perf_handler(iterations=iterations))


@cli.command('noise-sweep')
def noise_sweep_cmd() -> None:
    """Run a simple Aer noise-model sweep scaffold."""
    sys.exit(noise_sweep_handler())


def main() -> None:
    """Entry point for qrc_thresher CLI."""
    cli()


if __name__ == '__main__':
    main()
