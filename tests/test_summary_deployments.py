"""`summary` groups rows by deployment, names tasks, and reads runs.csv through one pinned reader
(docs/DECISIONS.md D019, items A.3, A.4 and A.6).

- ``gates.comparative.deployment_label`` reads a row's deployment from its circuit_hash against
  the deploying record's design and the two defaults; ``commands.summary.summary_table`` groups
  by (config_hash, sweep_id, task_name, design, deployment, primary_metric_name) and takes one
  row per pair through ``metrics.paired.collapse_exact_reruns``.
- ``task_names.task_of_metric`` is the inverse of TASK_METRICS.
- ``gates.comparative.read_runs_csv`` pins the parse: an empty primary_metric_value reads as NaN
  and the column stays float64 whatever other rows the file holds.

Everything runs in process or with the working directory in tmp_path; nothing is written under
the repository's results/. The CP5 modules are imported inside each test so that before they
exist every test fails on its own.
"""

from __future__ import annotations

import csv
import hashlib
import importlib

import numpy as np
import pandas as pd
from synthetic_rows import (
    COLUMNS,
    CONFIG_HASH,
    PAIRS,
    SWEEP_ID,
    ablation_hash,
    classical_arm,
    default_hash,
    default_inherited_arm,
    default_qrc_arm,
    design_hash,
    inherited_arm,
    qrc_arm,
    row,
)

SHA = {task: hashlib.sha256(f'tuning record {task}'.encode()).hexdigest()
       for task in ('stm', 'parity', 'narma')}
RECORD_TASK = {sha: task for task, sha in SHA.items()}
SUMMARY_COLUMNS = ['task_name', 'kind', 'model', 'task', 'design', 'deployment', 'metric',
                   'config', 'sweep_id', 'n_rows', 'n', 'mean', 'std']


def _comparative():
    return importlib.import_module('qrc_thresher.gates.comparative')


def _summary():
    return importlib.import_module('qrc_thresher.commands.summary')


def _designs():
    fam = _comparative()
    return fam.Designs(
        tuned={task: {p: design_hash(task, p) for p in PAIRS} for task in ('stm', 'parity',
                                                                          'narma')},
        default={'default': {p: default_hash(2, p) for p in PAIRS},
                 'default_w1': {p: default_hash(1, p) for p in PAIRS}},
        ablation_hash=ablation_hash,
    )


def _sweep() -> pd.DataFrame:
    """Rows whose deployments a (task_name, design, metric) grouping would pool."""
    rng = np.random.default_rng(3)
    n = len(PAIRS)
    stm = rng.uniform(2.0, 3.0, size=n)
    par = rng.uniform(0.8, 1.0, size=n)
    frames = [
        # parity tuned: design_parity and design_STM deployments (24 rows of one task_name)
        qrc_arm('parity', 'parity', par, metric='accuracy', tuning_record_sha=SHA['parity']),
        qrc_arm('parity', 'stm', par - 0.3, metric='accuracy', tuning_record_sha=SHA['stm']),
        qrc_arm('stm', 'stm', stm, metric='stm_memory', tuning_record_sha=SHA['stm']),
        # haar on STM memory: three parents (design_STM, default, default_w1), 36 rows
        inherited_arm('haar', 'stm', 'stm', stm - 0.5, metric='stm_memory',
                      tuning_record_sha=SHA['stm']),
        default_inherited_arm('haar', 'stm', 2, stm - 2.0, metric='stm_memory',
                              tuning_record_sha=SHA['stm']),
        default_inherited_arm('haar', 'stm', 1, stm - 2.2, metric='stm_memory',
                              tuning_record_sha=SHA['stm']),
        # no_entangle on parity accuracy, inheriting design_STM
        inherited_arm('no_entangle', 'parity', 'stm', par - 0.4, metric='accuracy',
                      tuning_record_sha=SHA['stm']),
        default_qrc_arm('stm', 2, stm - 1.9, metric='stm_memory', tuning_record_sha=SHA['stm']),
        default_qrc_arm('stm', 1, stm - 2.1, metric='stm_memory', tuning_record_sha=SHA['stm']),
        classical_arm('esn', stm - 0.2, metric='stm_memory', tuning_record_sha=SHA['stm']),
        classical_arm('esn', stm - 1.0, metric='stm_memory', design='default', budget=(1, 0),
                      tuning_record_sha=SHA['stm']),
    ]
    return pd.concat(frames, ignore_index=True)


def _resolved(monkeypatch, designs=None, record_task=None, reason=None):
    fam = _comparative()
    designs = _designs() if designs is None and reason is None else designs
    record_task = dict(RECORD_TASK) if record_task is None and reason is None else (record_task
                                                                                     or {})
    monkeypatch.setattr(fam, 'resolve_config',
                        lambda config_hash, config_path=None: (designs, record_task, reason))


def _table(runs):
    table, notes = _summary().summary_table(runs)
    assert list(table.columns) == SUMMARY_COLUMNS
    return table, notes


def _groups(table) -> dict:
    return {(r['task_name'], r['design'], r['deployment'], r['metric']): r
            for _, r in table.iterrows()}


class TestDeployments:
    def test_deployments_split_into_groups_of_twelve(self, monkeypatch) -> None:
        _resolved(monkeypatch)
        table, notes = _table(_sweep())
        groups = _groups(table)
        expected = {
            ('parity', 'tuned', 'tuned:design_parity', 'accuracy'),
            ('parity', 'tuned', 'tuned:design_stm', 'accuracy'),
            ('stm', 'tuned', 'tuned:design_stm', 'stm_memory'),
            ('ablation:haar', 'inherited', 'inherited:design_stm', 'stm_memory'),
            ('ablation:haar', 'inherited', 'inherited:default', 'stm_memory'),
            ('ablation:haar', 'inherited', 'inherited:default_w1', 'stm_memory'),
            ('ablation:no_entangle', 'inherited', 'inherited:design_stm', 'accuracy'),
            ('stm', 'default', 'default', 'stm_memory'),
            ('stm', 'default_w1', 'default_w1', 'stm_memory'),
            ('esn', 'tuned', 'tuned', 'stm_memory'),
            ('esn', 'default', 'default', 'stm_memory'),
        }
        assert set(groups) == expected
        assert all(r['n_rows'] == 12 and r['n'] == 12 for r in groups.values())
        assert table['n_rows'].max() == 12  # no 36-row or 24-row group
        assert not any('unresolved' in str(d) for d in table['deployment'])
        assert (table['config'] == CONFIG_HASH[:8]).all()
        assert (table['sweep_id'] == SWEEP_ID).all()
        assert notes == []

    def test_deployment_label_of_single_rows(self, monkeypatch) -> None:
        fam = _comparative()
        designs = _designs()
        pair = PAIRS[0]
        tuned = qrc_arm('stm', 'stm', [2.5], metric='stm_memory', pairs=[pair],
                        tuning_record_sha=SHA['stm']).iloc[0]
        assert fam.deployment_label(tuned, designs, RECORD_TASK) == 'tuned:design_stm'
        inherited = inherited_arm('haar', 'stm', 'stm', [2.0], metric='stm_memory', pairs=[pair],
                                  tuning_record_sha=SHA['stm']).iloc[0]
        assert fam.deployment_label(inherited, designs, RECORD_TASK) == 'inherited:design_stm'
        default = default_qrc_arm('stm', 1, [0.3], metric='stm_memory', pairs=[pair]).iloc[0]
        assert fam.deployment_label(default, designs, RECORD_TASK) == 'default_w1'
        baseline = classical_arm('esn', [2.3], metric='stm_memory', pairs=[pair]).iloc[0]
        assert fam.deployment_label(baseline, designs, RECORD_TASK) == 'tuned'
        untuned = qrc_arm('stm', 'stm', [2.5], metric='stm_memory', pairs=[pair], sweep_id='',
                          tuning_record_sha='', design='default').iloc[0]
        assert fam.deployment_label(untuned, designs, RECORD_TASK) == 'default'
        stray = tuned.copy()
        stray['circuit_hash'] = 'f' * 64
        assert fam.deployment_label(stray, designs, RECORD_TASK) == 'tuned:unresolved'

    def test_parents_come_from_the_deploying_record_only(self, monkeypatch) -> None:
        designs = _designs()
        pair = (45, 140)
        designs.tuned['narma'][pair] = designs.tuned['stm'][pair]  # the designs coincide
        _resolved(monkeypatch, designs=designs)
        table, _ = _table(_sweep())
        labels = set(table['deployment'])
        assert not any('=' in str(d) for d in labels), labels
        groups = _groups(table)
        haar = groups[('ablation:haar', 'inherited', 'inherited:design_stm', 'stm_memory')]
        assert haar['n_rows'] == haar['n'] == 12

    def test_an_inherited_row_without_a_tuning_block_labels_inherited(self) -> None:
        # TF11 (a): a config with no tuning block writes design 'inherited' with an empty
        # sweep_id; the label is the design.
        fam = _comparative()
        pair = PAIRS[0]
        row_ = inherited_arm('no_entangle', 'stm', 'stm', [1.0], metric='stm_memory', pairs=[pair],
                             sweep_id='', tuning_record_sha='').iloc[0]
        assert fam.deployment_label(row_, _designs(), RECORD_TASK) == 'inherited'
        assert fam.deployment_label(row_, None, {}) == 'inherited'  # no records needed

    def test_a_positive_join_when_two_candidates_match(self) -> None:
        # TF11 (b): when design_STM(pair) and the default coincide, both candidates' ablated
        # hashes match and the label joins them with '=' in candidate order.
        fam = _comparative()
        designs = _designs()
        pair = (45, 140)
        designs.default['default'][pair] = designs.tuned['stm'][pair]
        row_ = inherited_arm('haar', 'stm', 'stm', [1.0], metric='stm_memory', pairs=[pair],
                             tuning_record_sha=SHA['stm']).iloc[0]
        assert fam.deployment_label(row_, designs, RECORD_TASK) == 'inherited:design_stm=default'
        other = inherited_arm('haar', 'stm', 'stm', [1.0], metric='stm_memory', pairs=[PAIRS[0]],
                              tuning_record_sha=SHA['stm']).iloc[0]
        assert fam.deployment_label(other, designs, RECORD_TASK) == 'inherited:design_stm'

    def test_the_parent_is_the_deploying_records_design_not_the_first_match(self) -> None:
        # TF11 (c): rows deployed by the parity record whose design equals design_STM at one
        # pair label 'inherited:design_parity'. A search over every task starting with stm
        # would label them design_stm and fail this test.
        fam = _comparative()
        designs = _designs()
        pair = (45, 140)
        designs.tuned['parity'][pair] = designs.tuned['stm'][pair]
        row_ = inherited_arm('no_entangle', 'parity', 'stm', [0.9], metric='accuracy',
                             pairs=[pair], tuning_record_sha=SHA['parity']).iloc[0]
        assert str(row_['circuit_hash']) == ablation_hash(designs.tuned['parity'][pair],
                                                          'no_entangle', pair)
        assert fam.deployment_label(row_, designs, RECORD_TASK) == 'inherited:design_parity'

    def test_the_task_column_shows_tasks_not_metrics(self, monkeypatch) -> None:
        _resolved(monkeypatch)
        table, _ = _table(_sweep())
        tasks = set(table['task'])
        assert tasks == {'stm', 'parity'}
        assert not tasks & {'stm_memory', 'accuracy', 'nrmse'}
        by_name = {r['task_name']: r for _, r in table.iterrows()}
        assert by_name['ablation:haar']['task'] == 'stm'
        assert by_name['ablation:no_entangle']['task'] == 'parity'
        assert (by_name['ablation:haar']['kind'], by_name['ablation:haar']['model']) == (
            'ablation', 'haar')

    def test_without_tuning_records_rows_are_unresolved_with_a_note(self, monkeypatch) -> None:
        _resolved(monkeypatch, reason='no tuning record for this config_hash')
        runs = _sweep()
        table, notes = _table(runs)
        assert notes and any('no tuning record' in n for n in notes)
        labels = set(table['deployment'])
        assert 'tuned:unresolved' in labels and 'inherited:unresolved' in labels
        assert 'default' in labels and 'default_w1' in labels  # defaults need no record
        rendered = _summary()._arm_table(runs)
        assert 'nan' not in rendered.lower().replace('n/a', '')

    def test_exact_reruns_collapse_and_differing_duplicates_are_refused(self, monkeypatch) -> None:
        _resolved(monkeypatch)
        runs = _sweep()
        key = ('stm', 'tuned', 'tuned:design_stm', 'stm_memory')
        base = _groups(_table(runs)[0])[key]
        picked = runs[(runs['task_name'] == 'stm') & (runs['design'] == 'tuned')
                      & (runs['task_seed'] == 44)]
        rerun = picked.copy()
        rerun['run_id'] = 'rerun-id'
        table, notes = _table(pd.concat([runs, rerun], ignore_index=True))
        group = _groups(table)[key]
        assert (group['n_rows'], group['n']) == (13, 12)
        assert group['mean'] == base['mean'] and group['std'] == base['std']
        assert notes == []
        differing = picked.copy()
        differing['run_id'] = 'differing-id'
        differing['primary_metric_value'] = differing['primary_metric_value'] + 1e-6
        table, notes = _table(pd.concat([runs, differing], ignore_index=True))
        group = _groups(table)[key]
        assert group['mean'] == 'refused' and group['std'] == 'refused'
        assert any('(44, 139)' in n for n in notes)
        other = _groups(table)[('esn', 'tuned', 'tuned', 'stm_memory')]
        assert other['n'] == 12  # only the refused group is refused

    def test_summary_command_accepts_a_config_and_writes_the_table(self, tmp_path,
                                                                    monkeypatch) -> None:
        from click.testing import CliRunner

        from qrc_thresher.cli import cli

        _resolved(monkeypatch)
        monkeypatch.chdir(tmp_path)
        (tmp_path / 'results').mkdir()
        _sweep().to_csv(tmp_path / 'results' / 'runs.csv', index=False)
        result = CliRunner().invoke(cli, ['summary', '--phase', 'cp5', '--config',
                                          'configs/comparative.yaml'])
        assert result.exit_code == 0, result.output
        text = (tmp_path / 'results' / 'summaries' / 'cp5_summary.md').read_text(encoding='utf-8')
        header = text.split('## Runs')[0]
        assert '| ' + ' | '.join(SUMMARY_COLUMNS) + ' |' in header
        assert 'inherited:default_w1' in header and 'tuned:design_parity' in header


class TestTaskOfMetric:
    def test_round_trips_task_metrics_and_returns_none_for_an_unknown_metric(self) -> None:
        from qrc_thresher.task_names import TASK_METRICS, task_of_metric

        for task, metric in TASK_METRICS.items():
            assert task_of_metric(metric) == task
        assert task_of_metric('mc') is None
        assert task_of_metric('') is None


class TestReadRunsCsv:
    METRIC_TEXT = ['1.8106150706836674', '0.1', '2.9073000000000002', '1.5']

    def _write(self, path, rows) -> None:
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: ('' if v is None else v) for k, v in r.items()})

    def test_an_empty_value_reads_as_nan_and_the_column_stays_float64(self, tmp_path) -> None:
        fam = _comparative()
        good = [row('stm', PAIRS[i], text, metric='stm_memory', circuit_hash='a' * 64)
                for i, text in enumerate(self.METRIC_TEXT)]
        failed = row('stm', PAIRS[5], None, metric='', circuit_hash='b' * 64, success=False,
                     config_hash='c' * 64)
        with_failed, without = tmp_path / 'with.csv', tmp_path / 'without.csv'
        self._write(with_failed, good + [failed])
        self._write(without, good)
        a, b = fam.read_runs_csv(with_failed), fam.read_runs_csv(without)
        assert a['primary_metric_value'].dtype == np.float64
        assert b['primary_metric_value'].dtype == np.float64
        assert np.isnan(a['primary_metric_value'].iloc[-1])
        assert np.array_equal(a['primary_metric_value'].to_numpy()[:-1],
                              b['primary_metric_value'].to_numpy())
        assert a['primary_metric_value'].to_numpy()[:-1].tobytes() == \
            b['primary_metric_value'].to_numpy().tobytes()
        for column in ('sweep_id', 'tuning_record_sha', 'circuit_hash', 'config_hash'):
            assert a[column].dtype == object and all(isinstance(v, str) for v in a[column])
        assert a['primary_metric_name'].iloc[-1] == ''  # keep_default_na=False elsewhere
        assert a['failure_reason'].iloc[0] == ''
