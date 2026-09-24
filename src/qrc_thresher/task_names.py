"""The one place that spells runs.csv task names (docs/DECISIONS.md D011, D013; PI ruling 2).

- QRC rows carry the task: 'stm', 'parity', 'narma'.
- Matched ablations carry 'ablation:<name>'; their task is read from primary_metric_name
  (``task_metric``), which is part of the same suffix debt.
- Baseline rows carry the model and, except for STM (D009's names), the task: 'esn',
  'esn_parity', 'esn_narma', 'rks', 'rks_parity', 'rks_narma'.

Nothing else parses these suffixes: the family evaluator and `summary` use ``parse_task_name``
and the builders here. The suffix convention itself is logged as debt (a structured column
would be cleaner than a name).
"""

from __future__ import annotations

from typing import Dict, Optional

TASKS = ('stm', 'parity', 'narma')
BASELINE_MODELS = {'esn': 'esn', 'random_features': 'rks'}  # config name -> row prefix
ABLATIONS = ('phase_random', 'no_entangle', 'haar')
TASK_METRICS = {'stm': 'stm_memory', 'parity': 'accuracy', 'narma': 'nrmse'}  # D011, D014


def task_metric(task: str) -> str:
    """The primary metric name of a task's rows (ablation rows are told apart by it)."""
    _check_task(task)
    return TASK_METRICS[task]


def qrc_task_name(task: str) -> str:
    """The task_name of a QRC row."""
    _check_task(task)
    return task


def ablation_task_name(name: str) -> str:
    """The task_name of a matched-ablation row."""
    if name not in ABLATIONS:
        raise ValueError(f'unknown ablation {name!r}; choose from {list(ABLATIONS)}')
    return f'ablation:{name}'


def baseline_task_name(model: str, task: str) -> str:
    """The task_name of a baseline row: 'esn' / 'rks' for STM, '<prefix>_<task>' otherwise."""
    _check_task(task)
    prefix = BASELINE_MODELS.get(model, model)
    if prefix not in BASELINE_MODELS.values():
        raise ValueError(f'unknown baseline model {model!r}; choose from {sorted(BASELINE_MODELS)}')
    return prefix if task == 'stm' else f'{prefix}_{task}'


def parse_task_name(task_name: str) -> Dict[str, Optional[str]]:
    """Inverse of the builders: {'kind': 'qrc'|'ablation'|'baseline', 'model', 'task'}.

    Raises:
        ValueError: For a name none of the builders could have written (CP4b.1 item C9).
    """
    name = str(task_name)
    if name in TASKS:
        return {'kind': 'qrc', 'model': 'qrc', 'task': name}
    if name.startswith('ablation:'):
        ablation = name.split(':', 1)[1]
        return {'kind': 'ablation', 'model': ablation, 'task': None}
    prefix, _, suffix = name.partition('_')
    if prefix in BASELINE_MODELS.values() and (suffix == '' or suffix in TASKS):
        return {'kind': 'baseline', 'model': prefix, 'task': suffix or 'stm'}
    raise ValueError(
        f'unknown task_name {task_name!r}: not a task, an ablation:<name> or a baseline row'
    )


def _check_task(task: str) -> None:
    if task not in TASKS:
        raise ValueError(f'unknown task {task!r}; choose from {list(TASKS)}')


__all__ = [
    'ABLATIONS',
    'BASELINE_MODELS',
    'TASKS',
    'TASK_METRICS',
    'ablation_task_name',
    'baseline_task_name',
    'parse_task_name',
    'qrc_task_name',
    'task_metric',
]
