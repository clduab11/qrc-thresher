"""Compare two or more CP4c runs row for row (referee tool; read-only).

    uv run python ../qrc-tools/diff_runs.py results/runs.csv.archived_A results/runs.csv.archived_B results/runs.csv
    uv run python ../qrc-tools/diff_runs.py --family results/gates/COMPARATIVE.v1.<stampA>.json results/gates/COMPARATIVE.v1.<stampB>.json

Rows are joined on (task_name, design, circuit_hash, primary_metric_name, task_seed,
reservoir_seed). The metric is part of the key because an ablation row carries no task: the tuned
design's no-entangle ablation run on STM and on parity is the same circuit (same hash, same design
label, same pair) scored with a different metric (D015). The values that must agree between
deterministic replicates are compared exactly: primary_metric_value, every secondary metric,
n_configs, n_validation_evals, success. The columns
that legitimately differ between runs (run_id, timestamps, git commit, sweep id, record sha,
runtimes, platform, versions, artifact paths) are summarised per file instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

KEY = ['task_name', 'design', 'circuit_hash', 'primary_metric_name', 'task_seed', 'reservoir_seed']
EXACT = ['primary_metric_value', 'n_configs', 'n_validation_evals', 'success']
PER_FILE = ['git_commit_hash', 'git_branch', 'sweep_id', 'python_version', 'platform', 'config_hash',
            'measurement_model']


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, dtype=str)
    df['task_seed'] = df['task_seed'].astype(int)
    df['reservoir_seed'] = df['reservoir_seed'].astype(int)
    dup = df.duplicated(KEY, keep=False)
    if dup.any():
        print(f'  NOTE {path}: {int(dup.sum())} rows share a key (reruns?); the first is used')
        df = df[~df.duplicated(KEY, keep='first')]
    return df.set_index(KEY).sort_index()


def secondary(text: str) -> dict:
    try:
        value = json.loads(text) if text else {}
    except ValueError:
        return {'_unparsable': text}
    return value if isinstance(value, dict) else {'_value': value}


def compare_runs(paths: list[str]) -> int:
    frames = {p: load(p) for p in paths}
    for p, df in frames.items():
        print(f'{p}: {len(df)} rows')
        for col in PER_FILE:
            if col in df.columns:
                vals = sorted(set(df[col].tolist()))
                print(f'    {col:18s} {vals if len(vals) <= 3 else f"{len(vals)} distinct values"}')
    worst = 0
    base_path, base = paths[0], frames[paths[0]]
    for other_path in paths[1:]:
        other = frames[other_path]
        only_a = base.index.difference(other.index)
        only_b = other.index.difference(base.index)
        common = base.index.intersection(other.index)
        print(f'\n{base_path}  vs  {other_path}')
        print(f'  common rows {len(common)}; only in first {len(only_a)}; only in second {len(only_b)}')
        for idx in list(only_a)[:5]:
            print(f'    only in first : {idx}')
        for idx in list(only_b)[:5]:
            print(f'    only in second: {idx}')
        mismatches = []
        max_abs = 0.0
        for idx in common:
            a, b = base.loc[idx], other.loc[idx]
            for col in EXACT:
                if str(a[col]) != str(b[col]):
                    mismatches.append((idx, col, a[col], b[col]))
                    if col == 'primary_metric_value':
                        try:
                            max_abs = max(max_abs, abs(float(a[col]) - float(b[col])))
                        except ValueError:
                            pass
            sa, sb = secondary(a.get('secondary_metrics', '')), secondary(b.get('secondary_metrics', ''))
            if sa != sb:
                mismatches.append((idx, 'secondary_metrics', sa, sb))
        if not mismatches and not len(only_a) and not len(only_b):
            print(f'  IDENTICAL: every one of {len(common)} rows agrees exactly on '
                  f'{", ".join(EXACT)} and secondary_metrics')
        else:
            worst = max(worst, 1)
            print(f'  {len(mismatches)} mismatching cells; max |Δ primary_metric_value| = {max_abs:.3e}')
            for idx, col, va, vb in mismatches[:20]:
                print(f'    {idx} {col}: {va!r} != {vb!r}')
    return worst


def compare_family(paths: list[str]) -> int:
    docs = {p: json.loads(Path(p).read_text(encoding='utf-8')) for p in paths}
    for p, d in docs.items():
        print(f'{p}: protocol {d["protocol_sha256"][:16]}  commit {d["git_commit"]}  sweep {d["sweep_id"]}  '
              f'rows {d["n_rows"]}')
    fields = ['result', 'raw_p', 'adjusted_p', 'p_two_sided', 'baseline_better']
    cfields = ['values_a', 'values_b', 'mean_a', 'mean_b', 'mean_diff', 'd_z', 'ci_low', 'ci_high',
               'p_one_sided', 'p_wilcoxon', 'n_pairs', 'status']
    worst = 0
    base = docs[paths[0]]
    for p in paths[1:]:
        other = docs[p]
        diffs = []
        for m in base['members']:
            a, b = base['members'][m], other['members'][m]
            for f in fields:
                if a.get(f) != b.get(f):
                    diffs.append((m, f, a.get(f), b.get(f)))
            for f in cfields:
                if a['comparison'].get(f) != b['comparison'].get(f):
                    diffs.append((m, f'comparison.{f}', a['comparison'].get(f), b['comparison'].get(f)))
            if (a.get('floor') or {}).get('passed') != (b.get('floor') or {}).get('passed'):
                diffs.append((m, 'floor.passed', a.get('floor'), b.get('floor')))
        print(f'\n{paths[0]}  vs  {p}')
        if not diffs:
            print('  IDENTICAL: every member agrees on verdict, p-values, values, CI, d_z and floor')
        else:
            worst = 1
            for m, f, va, vb in diffs[:30]:
                print(f'    {m} {f}: {va!r} != {vb!r}')
    return worst


if __name__ == '__main__':
    args = sys.argv[1:]
    if not args or len(args) < 2 + (args[0] == '--family'):
        sys.exit(__doc__)
    if args[0] == '--family':
        sys.exit(compare_family(args[1:]))
    sys.exit(compare_runs(args))
