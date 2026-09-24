"""Summary command implementation (per-arm table via task_names; PI ruling 2)."""

from __future__ import annotations

from pathlib import Path


def _markdown_table(frame) -> str:
    """A GitHub-flavoured markdown table without the optional ``tabulate`` dependency."""
    columns = [str(c) for c in frame.columns]

    def cell(value) -> str:
        if isinstance(value, float):
            return f'{value:.4f}'
        return str(value).replace('|', '\\|').replace('\n', ' ')

    lines = ['| ' + ' | '.join(columns) + ' |', '|' + '|'.join(' --- ' for _ in columns) + '|']
    for _, row in frame.iterrows():
        lines.append('| ' + ' | '.join(cell(row[c]) for c in frame.columns) + ' |')
    return '\n'.join(lines) + '\n'


def _arm_table(df) -> str:
    """Mean +/- std of the primary metric per (task_name, design), with the parsed arm."""
    import pandas as pd

    from qrc_thresher.task_names import parse_task_name

    ok = df[df['success'].astype(str).str.lower() == 'true'].copy()
    if ok.empty:
        return '_no successful runs_\n'
    ok['primary_metric_value'] = pd.to_numeric(ok['primary_metric_value'], errors='coerce')
    if 'design' not in ok.columns:
        ok['design'] = ''
    rows = []
    for (task_name, design, metric), group in ok.groupby(
        ['task_name', 'design', 'primary_metric_name'], dropna=False, sort=True
    ):
        parsed = parse_task_name(task_name)
        values = group['primary_metric_value'].dropna()
        rows.append({
            'task_name': task_name,
            'kind': parsed['kind'],
            'model': parsed['model'],
            'task': parsed['task'] or metric,
            'design': design,
            'metric': metric,
            'n': int(len(values)),
            'mean': float(values.mean()) if len(values) else float('nan'),
            'std': float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        })
    return _markdown_table(pd.DataFrame(rows))


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
    with out_file.open('w', encoding='utf-8') as f:
        f.write(f'# {phase} Summary\n\n')
        f.write(f'- Total runs: {n_total}\n')
        f.write(f'- Successful runs: {n_success}\n')
        f.write(f'- Failed runs: {n_total - n_success}\n\n')
        f.write('## Arms (mean +/- std of the primary metric, n rows)\n\n')
        f.write(_arm_table(df))
        f.write('\n## Runs\n\n')
        f.write(_markdown_table(df))

    print(f'Summary written to {out_file}')
    return 0
