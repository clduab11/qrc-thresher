"""Summary command implementation."""

from __future__ import annotations

from pathlib import Path


def summary_handler(phase: str) -> int:
    """Handle summary command. Returns exit code."""
    import pandas as pd

    summaries_dir = Path('results') / 'summaries'
    summaries_dir.mkdir(parents=True, exist_ok=True)
    runs_csv = Path('results') / 'runs.csv'

    if not runs_csv.exists():
        print('No runs.csv found. Run some experiments first.')
        return 1

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

    print(f'Summary written to {out_file}')
    return 0
