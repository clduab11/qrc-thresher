"""One config file per comparison (docs/DECISIONS.md D013): `run TASK` runs every seed pair,
serial and parallel, and has no --seed; `ablation NAME TASK` and `baseline TASK` (parity
included) take the task as an argument; every row the three commands write from one file shares
its config_hash. RKS is a baseline, not an ablation (D011). The commands are exercised on a tiny
config without a tuning block, so every QRC row is design=default with an empty sweep_id.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from qrc_thresher.cli import cli
from qrc_thresher.config import load_config
from qrc_thresher.proof.run_manifest import _config_hash

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG = REPO_ROOT / 'configs' / 'alpha_lite.yaml'
PAIRS = [(42, 137), (43, 138), (44, 139)]
TASKS = ['stm', 'parity', 'narma']
TINY_GRID = {'spectral_radius': [0.9], 'input_scaling': [0.1, 1.0], 'leak_rate': [1.0]}


def _tiny(tmp_path: Path) -> Path:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding='utf-8'))
    raw['experiment_name'] = 'tiny_commands'
    raw['task'].update({'length': 100, 'delay_max': 5, 'parity_window': 2})
    raw['reservoir'].update({'n_qubits': 2, 'depth': 1, 'window': 2})
    raw['training']['washout'] = 20
    raw['seeds']['n_seeds'] = len(PAIRS)
    raw['baseline']['enabled'] = ['esn', 'random_features']
    raw['baseline']['esn_grid'] = TINY_GRID
    path = tmp_path / 'configs' / 'tiny.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw), encoding='utf-8')
    return path


def _rows(path: Path = Path('results') / 'runs.csv') -> list:
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def _pairs(rows) -> list:
    return sorted((int(r['task_seed']), int(r['reservoir_seed'])) for r in rows)


class TestRun:
    @pytest.mark.parametrize('task', TASKS)
    def test_run_runs_every_seed_pair(self, task, tmp_path, monkeypatch) -> None:
        cfg_path = _tiny(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(cli, ['run', task, '--config', str(cfg_path)])
        assert result.exit_code == 0, result.output
        rows = _rows()
        assert [r['task_name'] for r in rows] == [task] * len(PAIRS)
        assert _pairs(rows) == PAIRS
        assert all(r['success'] == 'True' for r in rows)
        assert all(r['design'] == 'default' and r['sweep_id'] == '' for r in rows)
        assert len({r['circuit_hash'] for r in rows}) == len(PAIRS)

    @pytest.mark.slow  # spawns worker processes that import PennyLane
    def test_run_in_parallel_writes_the_same_pairs(self, tmp_path, monkeypatch) -> None:
        cfg_path = _tiny(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(cli, ['run', 'parity', '--config', str(cfg_path),
                                          '--workers', '2'])
        assert result.exit_code == 0, result.output
        assert _pairs(_rows()) == PAIRS

    def test_run_has_no_seed_option(self, tmp_path, monkeypatch) -> None:
        cfg_path = _tiny(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(cli, ['run', 'stm', '--config', str(cfg_path), '--seed', '1'])
        assert result.exit_code == 2, result.output
        assert not (tmp_path / 'results' / 'runs.csv').exists()

    def test_run_handler_takes_task_and_config_only(self, tmp_path, monkeypatch) -> None:
        import inspect

        from qrc_thresher.commands.run import run_handler, run_parallel_handler

        assert 'seed' not in inspect.signature(run_handler).parameters
        assert 'seed' not in inspect.signature(run_parallel_handler).parameters
        cfg_path = _tiny(tmp_path)
        monkeypatch.chdir(tmp_path)
        assert run_handler('stm', str(cfg_path)) == 0
        assert _pairs(_rows()) == PAIRS


class TestAblationAndBaseline:
    @pytest.mark.parametrize('task', TASKS)
    def test_ablation_takes_the_task_as_an_argument(self, task, tmp_path, monkeypatch) -> None:
        cfg_path = _tiny(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(cli, ['ablation', 'no_entangle', task, '--config',
                                          str(cfg_path)])
        assert result.exit_code == 0, result.output
        rows = _rows()
        assert [r['task_name'] for r in rows] == ['ablation:no_entangle'] * len(PAIRS)
        assert _pairs(rows) == PAIRS
        metric = {'stm': 'stm_memory', 'parity': 'accuracy', 'narma': 'nrmse'}[task]
        assert all(r['primary_metric_name'] == metric for r in rows)
        assert all(r['design'] == 'inherited' for r in rows)

    def test_ablation_refuses_random_features_and_a_missing_task(self, tmp_path,
                                                                  monkeypatch) -> None:
        from qrc_thresher.commands.ablation import ABLATION_NAMES

        assert set(ABLATION_NAMES) == {'phase_random', 'no_entangle', 'haar'}
        cfg_path = _tiny(tmp_path)
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        assert runner.invoke(cli, ['ablation', 'random_features', 'parity', '--config',
                                   str(cfg_path)]).exit_code == 2
        assert runner.invoke(cli, ['ablation', 'no_entangle', '--config',
                                   str(cfg_path)]).exit_code == 2
        assert not (tmp_path / 'results' / 'runs.csv').exists()

    def test_baseline_takes_parity_and_runs_both_baselines(self, tmp_path, monkeypatch) -> None:
        cfg_path = _tiny(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(cli, ['baseline', 'parity', '--config', str(cfg_path)])
        assert result.exit_code == 0, result.output
        rows = _rows()
        by_name = {}
        for r in rows:
            by_name.setdefault(r['task_name'], []).append(r)
        assert set(by_name) == {'esn_parity', 'rks_parity'}
        for name, group in by_name.items():
            assert _pairs(group) == PAIRS, name
            assert all(r['primary_metric_name'] == 'accuracy' for r in group)
            assert all(0.0 <= float(r['primary_metric_value']) <= 1.0 for r in group)
        # In-line ESN tuning over baseline.esn_grid (no tuning block): design tuned, empty sweep.
        assert all(r['design'] == 'tuned' and r['sweep_id'] == '' for r in by_name['esn_parity'])
        assert all((r['n_configs'], r['n_validation_evals']) == ('2', '10')
                   for r in by_name['esn_parity'])
        # RKS at D010's default (sigma 1, d = 1) without a tuning block: design default.
        assert all(r['design'] == 'default' for r in by_name['rks_parity'])
        assert all((r['n_configs'], r['n_validation_evals']) == ('1', '0')
                   for r in by_name['rks_parity'])

    @pytest.mark.parametrize('task, names', [('stm', {'esn', 'rks'}),
                                             ('narma', {'esn_narma', 'rks_narma'})])
    def test_baseline_task_names(self, task, names, tmp_path, monkeypatch) -> None:
        from qrc_thresher.commands.baseline import run_baselines

        cfg_path = _tiny(tmp_path)
        monkeypatch.chdir(tmp_path)
        manifests = run_baselines(load_config(cfg_path), task, cfg_path)
        assert {m.task_name for m in manifests} == names
        assert all(m.success for m in manifests), [m.failure_reason for m in manifests]
        metric = 'stm_memory' if task == 'stm' else 'nrmse'
        assert {m.primary_metric_name for m in manifests} == {metric}


class TestOneConfigHash:
    def test_every_arm_written_from_one_file_shares_its_hash(self, tmp_path, monkeypatch) -> None:
        cfg_path = _tiny(tmp_path)
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        for args in (['run', 'parity'], ['ablation', 'no_entangle', 'parity'],
                     ['baseline', 'parity']):
            result = runner.invoke(cli, args + ['--config', str(cfg_path)])
            assert result.exit_code == 0, (args, result.output)
        rows = _rows()
        assert {r['task_name'] for r in rows} == {'parity', 'ablation:no_entangle', 'esn_parity',
                                                  'rks_parity'}
        assert {r['config_hash'] for r in rows} == {_config_hash(cfg_path)}
        assert {r['sweep_id'] for r in rows} == {''}
        assert {r['tuning_record_sha'] for r in rows} == {''}
        for name in ('parity', 'ablation:no_entangle', 'esn_parity', 'rks_parity'):
            assert _pairs([r for r in rows if r['task_name'] == name]) == PAIRS, name
