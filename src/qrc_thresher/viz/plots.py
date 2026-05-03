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
