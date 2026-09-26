"""Summary command implementation (per-deployment table via task_names; PI ruling 2, D019 A.3).

`summary_table` groups the successful rows by (config_hash, sweep_id, task_name, design,
deployment, primary_metric_name), where the deployment is read from each row's circuit_hash
against the tuning records (``gates.comparative.deployment_label``), and takes one row per
pair through ``metrics.paired.collapse_exact_reruns``. It is callable in process without
writing a file; `summary_handler` writes results/summaries/<phase>_summary.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

SUMMARY_COLUMNS = ['task_name', 'kind', 'model', 'task', 'design', 'deployment', 'metric',
                   'config', 'sweep_id', 'n_rows', 'n', 'mean', 'std']


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


def summary_table(
    runs, config_path: Optional[Path] = None, *, resolver: Optional[Callable] = None
) -> Tuple['object', List[str]]:
    """The per-deployment table of the successful rows and its note lines (D019, item A.3).

    Args:
        runs: Manifest rows (``gates.comparative.read_runs_csv``), any configs.
        config_path: A config file to try first when resolving a config_hash's designs.
        resolver: ``(config_hash, config_path) -> (designs, {record_sha256: task}, reason)``;
            defaults to ``gates.comparative.resolve_config``, looked up at call time.

    Returns:
        (frame with SUMMARY_COLUMNS, notes): n, mean and std are over one row per pair; a group
        whose duplicates are not exact reruns shows 'refused' with a note naming the pairs; a
        config whose records cannot be resolved gets a note and '<design>:unresolved' labels.
    """
    import pandas as pd

    from qrc_thresher.gates import comparative
    from qrc_thresher.metrics.paired import collapse_exact_reruns
    from qrc_thresher.task_names import parse_task_name, task_of_metric

    resolve = resolver if resolver is not None else comparative.resolve_config
    notes: List[str] = []
    ok = runs[runs['success'].astype(str).str.lower() == 'true'].copy()
    if ok.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS), notes
    ok['primary_metric_value'] = pd.to_numeric(ok['primary_metric_value'], errors='coerce')
    for column in ('design', 'sweep_id', 'config_hash', 'tuning_record_sha'):
        if column not in ok.columns:
            ok[column] = ''
    resolved: dict = {}
    labels = []
    # A config is resolved when any of its successful rows carries a sweep_id (CP5b C8); a config
    # none of whose rows has one needs no designs (no tuning block).
    has_sweep = ok.groupby(ok['config_hash'].astype(str))['sweep_id'].apply(
        lambda col: any(str(v or '') != '' for v in col)
    ).to_dict()
    for _, row in ok.iterrows():
        config_hash = str(row['config_hash'])
        if config_hash not in resolved:
            if not has_sweep.get(config_hash, False):
                resolved[config_hash] = (None, {})  # no tuning block: designs are not needed
            else:
                designs, record_task, reason = resolve(config_hash, config_path)
                if reason:
                    notes.append(f'config {config_hash[:8]}: {reason}; its tuned and inherited '
                                 'deployments are unresolved')
                resolved[config_hash] = (designs, record_task or {})
        labels.append(comparative.deployment_label(row, *resolved[config_hash]))
    ok['deployment'] = labels
    rows = []
    keys = ['config_hash', 'sweep_id', 'task_name', 'design', 'deployment', 'primary_metric_name']
    for (config_hash, sweep_id, task_name, design, deployment, metric), group in ok.groupby(
        keys, dropna=False, sort=True
    ):
        try:
            parsed = parse_task_name(task_name)
        except ValueError:
            parsed = {'kind': 'unknown', 'model': None, 'task': None}
        collapsed, _, reason = collapse_exact_reruns(group, f'{task_name}/{design}/{deployment}')
        if collapsed is None:
            n, mean, std = 0, 'refused', 'refused'
            notes.append(f'{task_name} {design} {deployment} {metric}: {reason}')
        else:
            values = pd.Series(
                [collapsed[p]['primary_metric_value'] for p in sorted(collapsed)], dtype='float64'
            ).dropna()
            n = int(len(values))
            mean = float(values.mean()) if n else 'n/a'
            std = float(values.std(ddof=1)) if n > 1 else (0.0 if n else 'n/a')
        rows.append({
            'task_name': task_name,
            'kind': parsed['kind'],
            'model': parsed['model'],
            'task': parsed['task'] or task_of_metric(str(metric)) or 'n/a',
            'design': design,
            'deployment': deployment,
            'metric': metric,
            'config': str(config_hash)[:8],
            'sweep_id': sweep_id,
            'n_rows': int(len(group)),
            'n': n,
            'mean': mean,
            'std': std,
        })
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS), notes


def _arm_table(df, config_path: Optional[Path] = None) -> str:
    """Mean +/- std of the primary metric per deployment (one row per pair), with the notes."""
    table, notes = summary_table(df, config_path)
    if table.empty:
        return '_no successful runs_\n'
    text = _markdown_table(table)
    if notes:
        text += '\n' + '\n'.join(f'- note: {note}' for note in notes) + '\n'
    return text


def summary_handler(phase: str, config_path: Optional[str] = None) -> int:
    """Handle summary command. Returns exit code."""
    from qrc_thresher.gates.comparative import read_runs_csv

    summaries_dir = Path('results') / 'summaries'
    summaries_dir.mkdir(parents=True, exist_ok=True)
    runs_csv = Path('results') / 'runs.csv'

    if not runs_csv.exists():
        print('No runs.csv found. Run some experiments first.')
        return 1

    df = read_runs_csv(runs_csv)
    n_total = len(df)
    n_success = int(df['success'].astype(str).str.lower().eq('true').sum())

    out_file = summaries_dir / f'{phase}_summary.md'
    with out_file.open('w', encoding='utf-8') as f:
        f.write(f'# {phase} Summary\n\n')
        f.write(f'- Total runs: {n_total}\n')
        f.write(f'- Successful runs: {n_success}\n')
        f.write(f'- Failed runs: {n_total - n_success}\n\n')
        f.write('## Arms (mean +/- std of the primary metric over one row per pair; '
                'n_rows before collapsing)\n\n')
        f.write(_arm_table(df, Path(config_path) if config_path else None))
        f.write('\n## Runs\n\n')
        f.write(_markdown_table(df))

    print(f'Summary written to {out_file}')
    return 0
