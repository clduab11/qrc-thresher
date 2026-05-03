"""CLI surface for qrc_thresher - thin dispatcher.

Commands:
  health   - Run all health checks.
  run      - Run a task benchmark.
  ablation - Run an ablation study.
  gate     - Evaluate a decision gate.
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
@click.option('--workers', default=1, type=int, help='Number of parallel workers.')
def run_cmd(task: str, config_path: str, seed: Optional[int], workers: int) -> None:
    """Run a task benchmark and write a run manifest."""
    if workers > 1:
        from qrc_thresher.commands.run import run_parallel_handler

        sys.exit(run_parallel_handler(task, config_path, seed, workers))
    else:
        sys.exit(run_handler(task, config_path, seed))


@cli.command('ablation')
@click.argument(
    'name',
    type=click.Choice(['phase_random', 'no_entangle', 'random_features', 'haar']),
)
@click.option('--config', 'config_path', default='configs/alpha_lite.yaml')
@click.option('--seed', default=None, type=int)
def ablation_cmd(name: str, config_path: str, seed: Optional[int]) -> None:
    """Run an ablation study."""
    sys.exit(ablation_handler(name, config_path, seed))


@cli.command('gate')
@click.argument(
    'name',
    type=click.Choice(['G0', 'G0.5', 'G1', 'G2', 'G2.5', 'G3', 'G4', 'G5', 'G6', 'G7']),
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
    sys.exit(gate_handler(name))


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
