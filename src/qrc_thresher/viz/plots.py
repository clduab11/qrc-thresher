"""Plotting utilities for qrc_thresher.

Matplotlib only. No seaborn, plotly, or bokeh.
Default figure size: (8, 5) for single panels, (12, 5) for side-by-side.
DPI: 150 for screen, 300 for export.
Color palette: colorblind-safe (tab10).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')

logger = logging.getLogger(__name__)

_FIGURE_SIZE_SINGLE = (8, 5)
_FIGURE_SIZE_DUAL = (12, 5)
_DPI_SCREEN = 150
_DPI_EXPORT = 300
_COLORS = plt.cm.tab10.colors  # type: ignore[attr-defined]


def plot_stm_mc(
    mc_per_delay: np.ndarray,
    run_id: str,
    out_dir: Path,
    gate_threshold: Optional[float] = None,
    title: str = 'Short-Term Memory Capacity',
) -> List[Path]:
    """Plot per-delay memory capacity.

    Args:
        mc_per_delay: MC contribution per delay, shape (K+1,).
        run_id: Run identifier for filename.
        out_dir: Output directory.
        gate_threshold: Optional horizontal line at gate threshold.
        title: Plot title.

    Returns:
        List of paths to saved figure files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=_FIGURE_SIZE_SINGLE)
    delays = np.arange(len(mc_per_delay))
    ax.bar(delays, mc_per_delay, color=_COLORS[0], alpha=0.8)
    if gate_threshold is not None:
        ax.axhline(
            y=gate_threshold,
            color='red',
            linestyle='--',
            label=f'G1 threshold={gate_threshold}',
        )
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel('Delay k')
    ax.set_ylabel('Correlation^2')
    ax.set_ylim(0, 1.05)
    fig.tight_layout()

    paths = []
    for ext in ('png', 'pdf'):
        p = out_dir / f'{run_id}_stm_mc.{ext}'
        dpi = _DPI_EXPORT if ext == 'pdf' else _DPI_SCREEN
        fig.savefig(p, dpi=dpi)
        paths.append(p)
        logger.debug('Saved plot: %s', p)
    plt.close(fig)
    return paths


def plot_comparison(
    scores_dict: Dict[str, float],
    metric_name: str,
    run_id: str,
    out_dir: Path,
    title: str = 'Method Comparison',
    higher_is_better: bool = True,
) -> List[Path]:
    """Plot bar chart comparing methods.

    Args:
        scores_dict: Dict mapping method name to scalar score.
        metric_name: Name of the metric for y-axis label.
        run_id: Run identifier for filename.
        out_dir: Output directory.
        title: Plot title.
        higher_is_better: Whether higher score is better (affects bar color).

    Returns:
        List of paths to saved figure files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = list(scores_dict.keys())
    scores = [scores_dict[m] for m in methods]

    fig, ax = plt.subplots(figsize=_FIGURE_SIZE_SINGLE)
    colors = [_COLORS[i % len(_COLORS)] for i in range(len(methods))]
    bars = ax.bar(methods, scores, color=colors, alpha=0.8)
    ax.bar_label(bars, fmt='%.3f', padding=2)
    ax.set_title(title)
    ax.set_xlabel('Method')
    ax.set_ylabel(metric_name)
    fig.tight_layout()

    paths = []
    for ext in ('png', 'pdf'):
        p = out_dir / f'{run_id}_comparison.{ext}'
        dpi = _DPI_EXPORT if ext == 'pdf' else _DPI_SCREEN
        fig.savefig(p, dpi=dpi)
        paths.append(p)
    plt.close(fig)
    return paths


def plot_stm_delay_heatmap(
    mc_by_delay_and_seed: np.ndarray,
    run_id: str,
    out_dir: Path,
    title: str = 'STM Delay Heatmap',
) -> List[Path]:
    """Plot heatmap of MC contribution by seed (rows) and delay (cols)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=_FIGURE_SIZE_SINGLE)
    im = ax.imshow(mc_by_delay_and_seed, aspect='auto', cmap='viridis')
    fig.colorbar(im, ax=ax, label='corr^2')
    ax.set_title(title)
    ax.set_xlabel('Delay')
    ax.set_ylabel('Seed index')
    fig.tight_layout()

    paths = []
    for ext in ('png', 'pdf'):
        p = out_dir / f'{run_id}_stm_delay_heatmap.{ext}'
        dpi = _DPI_EXPORT if ext == 'pdf' else _DPI_SCREEN
        fig.savefig(p, dpi=dpi)
        paths.append(p)
    plt.close(fig)
    return paths


def plot_runtime_breakdown(
    stage_times: Dict[str, float],
    run_id: str,
    out_dir: Path,
    title: str = 'Runtime Breakdown',
) -> List[Path]:
    """Plot stage-wise runtime bar chart."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = list(stage_times.keys())
    values = [stage_times[s] for s in stages]

    fig, ax = plt.subplots(figsize=_FIGURE_SIZE_SINGLE)
    bars = ax.bar(stages, values, color=_COLORS[1], alpha=0.85)
    ax.bar_label(bars, fmt='%.2fs', padding=2)
    ax.set_title(title)
    ax.set_ylabel('Seconds')
    ax.set_xlabel('Stage')
    ax.tick_params(axis='x', rotation=25)
    fig.tight_layout()

    paths = []
    for ext in ('png', 'pdf'):
        p = out_dir / f'{run_id}_runtime_breakdown.{ext}'
        dpi = _DPI_EXPORT if ext == 'pdf' else _DPI_SCREEN
        fig.savefig(p, dpi=dpi)
        paths.append(p)
    plt.close(fig)
    return paths


def plot_gate_decision_tree(
    gate_results: Dict[str, str],
    run_id: str,
    out_dir: Path,
    title: str = 'Gate Decision Tree',
) -> List[Path]:
    """Render a simple gate-decision flow summary chart."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gates = list(gate_results.keys())
    verdicts = [gate_results[g] for g in gates]
    score = [
        1 if v == 'PASS' else (0 if v == 'INSUFFICIENT_EVIDENCE' else -1)
        for v in verdicts
    ]
    colors = ['#2e7d32' if s > 0 else ('#f9a825' if s == 0 else '#c62828') for s in score]

    fig, ax = plt.subplots(figsize=_FIGURE_SIZE_DUAL)
    ax.bar(gates, score, color=colors, alpha=0.9)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(['FAIL', 'INSUFFICIENT', 'PASS'])
    ax.set_title(title)
    ax.set_xlabel('Gate')
    ax.set_ylabel('Verdict')
    fig.tight_layout()

    paths = []
    for ext in ('png', 'pdf'):
        p = out_dir / f'{run_id}_gate_tree.{ext}'
        dpi = _DPI_EXPORT if ext == 'pdf' else _DPI_SCREEN
        fig.savefig(p, dpi=dpi)
        paths.append(p)
    plt.close(fig)
    return paths


def plot_metric_correlation_matrix(
    metrics: Dict[str, np.ndarray],
    run_id: str,
    out_dir: Path,
    title: str = 'Metric Correlation Matrix',
) -> List[Path]:
    """Plot correlation matrix for named metric series."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(metrics.keys())
    if len(names) < 2:
        raise ValueError('Need at least 2 metrics for correlation matrix')

    mat = np.vstack([metrics[n] for n in names])
    corr = np.corrcoef(mat)

    fig, ax = plt.subplots(figsize=_FIGURE_SIZE_SINGLE)
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1.0, vmax=1.0)
    fig.colorbar(im, ax=ax, label='Pearson r')
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.set_title(title)
    fig.tight_layout()

    paths = []
    for ext in ('png', 'pdf'):
        p = out_dir / f'{run_id}_metric_corr.{ext}'
        dpi = _DPI_EXPORT if ext == 'pdf' else _DPI_SCREEN
        fig.savefig(p, dpi=dpi)
        paths.append(p)
    plt.close(fig)
    return paths


# Okabe-Ito colours: distinguishable under common colour-vision deficiencies.
_SEED_COLORS = ('#0072B2', '#E69F00', '#CC79A7', '#009E73', '#56B4E9', '#D55E00', '#F0E442')
_NULL_COLOR = '0.62'
_FONT_BASE, _FONT_NOTE, _FONT_TICK = 10, 9, 8


def plot_forgetting_curve(result: dict):
    """Plot a G0.7 result: the forgetting curve and both clauses against their nulls.

    Left: held-out r2_k against delay k for each seed, with that seed's permutation-null
    95th percentile shaded in the same colour. k = 0 is shown but is not memory.
    Middle: each seed's memory score S = sum over k >= 1 of r2_k, against its null.
    Right: each seed's window-2 parity accuracy, against its shuffled-label null.
    The footer carries the measurement-model label and the sample sizes.

    Args:
        result: Output of qrc_thresher.gates.g07.evaluate.

    Returns:
        The matplotlib Figure. The caller saves and closes it.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    stm = result['clauses']['stm']
    parity = result['clauses']['parity']
    seeds = stm['seeds']
    n_perm = result.get('n_permutations', 0)
    colors = [_SEED_COLORS[i % len(_SEED_COLORS)] for i in range(len(seeds))]
    tags = [f"{s['task_seed']}/{s['reservoir_seed']}" for s in seeds]

    fig = plt.figure(figsize=(13.0, 5.8))
    grid = fig.add_gridspec(
        1, 3, width_ratios=[2.5, 1.0, 1.0], wspace=0.36,
        left=0.055, right=0.985, top=0.76, bottom=0.21,
    )
    ax = fig.add_subplot(grid[0])
    ax_s = fig.add_subplot(grid[1])
    ax_p = fig.add_subplot(grid[2])

    # Left: forgetting curves with their per-seed null bands.
    scored = [s for s in seeds if not s['degenerate']]
    n_delays = len(scored[0]['r2']) if scored else 21
    delays = np.arange(n_delays)
    ax.axvspan(-0.5, 0.5, color='0.92', lw=0, zorder=0)
    handles = []
    for s, color, tag in zip(seeds, colors, tags):
        if s['degenerate']:
            continue
        ax.fill_between(delays, 0.0, s['null_r2_q95'], color=color, alpha=0.22, lw=0, zorder=1)
        line, = ax.plot(
            delays, s['r2'], color=color, marker='o', ms=3.5, lw=1.6, zorder=3,
            label=f"seed {tag}: S = {s['S']:.2f}, p = {s['p']:.3f}",
        )
        handles.append(line)
    if scored:
        handles += [
            Patch(facecolor='0.45', alpha=0.3, lw=0,
                  label=f'shaded: null 95th percentile ({n_perm} permutations)'),
            Patch(facecolor='0.92', lw=0, label='grey column: k = 0, reported but not memory'),
        ]
        ax.legend(handles=handles, loc='upper right', frameon=False, fontsize=_FONT_NOTE)
    else:
        ax.text(0.5, 0.5, _degenerate_note(seeds, tags), transform=ax.transAxes,
                ha='center', va='center', fontsize=_FONT_NOTE, color='0.25')
    ax.set_xlim(-0.5, n_delays - 0.5)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(range(0, n_delays, 2))
    ax.set_xlabel('delay k (time steps into the past)', fontsize=_FONT_BASE)
    ax.set_ylabel(r'held-out $r^2_k$: squared correlation with $u_{t-k}$', fontsize=_FONT_BASE)
    ax.set_title('How well past inputs can be read back, per seed', loc='left',
                 fontsize=_FONT_BASE)

    # Middle and right: each clause's statistic against its null, per seed.
    _observed_vs_null(ax_s, seeds, colors, tags, 'S', 'null_S', 'null_S_q95')
    ax_s.set_ylabel(r'memory score $S = \sum_{k \geq 1} r^2_k$', fontsize=_FONT_BASE)
    ax_s.set_title(_clause_title('STM clause', stm, result), loc='left', fontsize=_FONT_BASE)
    _observed_vs_null(
        ax_p, parity['seeds'], colors, tags, 'accuracy', 'null_accuracy', 'null_accuracy_q95'
    )
    ax_p.axhline(0.5, color='0.35', lw=0.9, ls='--', zorder=0)
    ax_p.set_ylabel('held-out accuracy, window-2 parity', fontsize=_FONT_BASE)
    ax_p.set_title(_clause_title('Parity clause', parity, result), loc='left',
                   fontsize=_FONT_BASE)
    for axis in (ax_s, ax_p):
        axis.set_xlabel('seed pair (task/reservoir)', fontsize=_FONT_BASE)
    for axis in (ax, ax_s, ax_p):
        axis.tick_params(labelsize=_FONT_TICK)
        axis.spines[['top', 'right']].set_visible(False)

    key = [
        Line2D([], [], marker='D', ls='', color='0.3', ms=6, label='observed (seed colour)'),
        Line2D([], [], marker='o', ls='', color=_NULL_COLOR, ms=3.5, label='null draws'),
        Line2D([], [], color='0.25', lw=1.2, label='null 95th percentile'),
        Line2D([], [], color='0.35', lw=0.9, ls='--', label='chance (parity)'),
    ]
    fig.legend(handles=key, loc='upper right', bbox_to_anchor=(0.985, 0.915), ncol=2,
               frameon=False, fontsize=_FONT_TICK, handlelength=1.6, columnspacing=1.2)
    model = result.get('model') or 'model'
    fig.text(0.055, 0.965, f"G0.7 memory sanity gate: {result['result']}  ({model})",
             fontsize=_FONT_BASE + 1, weight='bold', ha='left', va='top')
    fig.text(0.055, 0.915, result.get('message', ''), fontsize=_FONT_NOTE, ha='left', va='top')
    level = stm['significance_level']
    fig.text(
        0.055, 0.03,
        f"Measurement: {result.get('measurement_label', '')}. Every value here assumes exact "
        f"expectation values, an upper bound on what a device could measure.\n"
        f"n = {result.get('n_seeds', len(seeds))} seed pairs. A clause passes only if every "
        f"seed has p <= {level:g}; each null refits the readout on {n_perm} permutations, "
        f"p = (1 + #null >= observed) / {n_perm + 1}. Protocol G0.7 "
        f"v{result.get('protocol_version', '?')} "
        f"(sha256 {str(result.get('protocol_sha256', ''))[:12]}).",
        fontsize=_FONT_TICK, ha='left', va='bottom', color='0.25',
    )
    return fig


def _observed_vs_null(ax, seeds, colors, tags, observed_key, null_key, q95_key) -> None:
    """Per seed: the null draws as a grey strip, their 95th percentile, and the observed value."""
    jitter_rng = np.random.default_rng(0)  # fixed, so the figure is reproducible
    scored = 0
    for x, (s, color) in enumerate(zip(seeds, colors)):
        if s['degenerate']:
            ax.text(x, 0.3, 'not\nscored', transform=ax.get_xaxis_transform(),
                    ha='center', va='center', fontsize=_FONT_TICK, color='0.35')
            continue
        scored += 1
        null = np.asarray(s[null_key], dtype=float)
        ax.scatter(x + jitter_rng.uniform(-0.2, 0.2, null.size), null, s=5, lw=0,
                   color=_NULL_COLOR, alpha=0.55, zorder=1)
        ax.hlines(s[q95_key], x - 0.3, x + 0.3, color='0.25', lw=1.2, zorder=2)
        ax.scatter([x], [s[observed_key]], s=60, marker='D', color=color,
                   edgecolor='white', lw=0.8, zorder=3)
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels(tags)
    ax.set_xlim(-0.6, len(seeds) - 0.4)
    if scored:
        ax.margins(y=0.08)
    else:
        ax.set_ylim(0.0, 1.0)


def _clause_title(name: str, clause: dict, result: dict) -> str:
    n, k = clause['n_seeds'], clause['n_passed']
    if clause['result'] == 'INSUFFICIENT_EVIDENCE':
        return f"{name}: too few seeds\n{n} given, {result.get('min_seeds', '?')} needed"
    return f"{name}: {clause['result']}\n{k} of {n} seeds beat the null"


def _degenerate_note(seeds, tags) -> str:
    """One line per distinct degeneracy reason, naming the seeds it applies to."""
    import textwrap

    reasons: Dict[str, List[str]] = {}
    for s, tag in zip(seeds, tags):
        if s['degenerate']:
            reasons.setdefault(s['degenerate_reason'], []).append(tag)
    lines = []
    for reason, which in reasons.items():
        lines.append(f"Not scored (degenerate), seeds {', '.join(which)}:")
        lines.extend(textwrap.wrap(reason, width=60))
    return '\n'.join(lines)
