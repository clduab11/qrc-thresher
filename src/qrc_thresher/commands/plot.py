"""Plot command implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def plot_handler(run_id: str, out_path: Optional[str]) -> int:
    """Handle plot command. Returns exit code."""
    out_dir = Path(out_path) if out_path else Path('results') / 'figures' / run_id
    print(f'Figures would be written to {out_dir}')
    print('Use plot functions from qrc_thresher.viz.plots directly for now.')
    return 0
