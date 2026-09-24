"""CP4b.1 integrity, sweep-id and hygiene tests (docs/DECISIONS.md D011, D015, D016).

Item numbers refer to the CP4b.1 list: A1 (the deployed QRC equals the validated one, by hash),
A2 (selection is structurally train-only), A3 (a lost runs.csv row fails the run), A4 (a
missing record or pair aborts before any row), A5 (family git_commit carries -dirty), B6 (one
sweep_id across the three records), C8 (config models forbid unknown keys; no silent
delay_max), C9 (parse_task_name raises; design flags need a tuning block; gate family needs
--config; summary writes n/a).
"""

from __future__ import annotations

import copy
import csv
import dataclasses
import importlib
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
import test_tuning as T
import yaml
from click.testing import CliRunner

from qrc_thresher.config import AlphaLiteConfig, load_config

REPO_ROOT = Path(__file__).parent.parent
CONFIGS = REPO_ROOT / 'configs'


def _tuning():
    return importlib.import_module('qrc_thresher.tuning')


def _rows(path: Path) -> list:
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def _rewrite(record: dict, cfg_path: Path, task: str) -> None:
    """Write a (tampered) record back under its own, recomputed hash so load_record accepts it."""
    tuning = _tuning()
    from qrc_thresher.proof.run_manifest import _config_hash

    record = copy.deepcopy(record)
    record.pop('record_sha256', None)
    record['record_sha256'] = tuning.record_sha256(record)
    path = tuning.record_path(_config_hash(cfg_path), task)
    path.write_text(json.dumps(record, indent=2), encoding='utf-8')


@pytest.fixture(scope='module')
def one_sweep(tmp_path_factory):
    """The tiny config tuned by ``tune_all``: three records under one sweep_id (item B6)."""
    tmp = tmp_path_factory.mktemp('one_sweep')
    cfg_path = T._write(tmp)
    cfg = load_config(cfg_path)
    cwd = os.getcwd()
    os.chdir(tmp)
    try:
        records = _tuning().tune_all(cfg, cfg_path)
    finally:
        os.chdir(cwd)
    return {'dir': tmp, 'cfg_path': cfg_path, 'cfg': cfg, 'records': records}


@pytest.fixture
def tampered(one_sweep, tmp_path, monkeypatch):
    """A copy of the one-sweep directory whose STM record carries a wrong hash for pair 43/138
    and whose parity record lacks pair 43/138; both re-signed so only the deploy checks fire."""
    import shutil

    shutil.copytree(one_sweep['dir'], tmp_path / 'work')
    work = tmp_path / 'work'
    cfg_path = work / 'configs' / 'tiny.yaml'
    monkeypatch.chdir(work)
    stm = copy.deepcopy(one_sweep['records']['stm'])
    stm['qrc']['43/138']['circuit_hash'] = '0' * 64
    _rewrite(stm, cfg_path, 'stm')
    parity = copy.deepcopy(one_sweep['records']['parity'])
    del parity['qrc']['43/138']
    _rewrite(parity, cfg_path, 'parity')
    (work / 'results' / 'runs.csv').unlink(missing_ok=True)
    return {'dir': work, 'cfg_path': cfg_path, 'cfg': load_config(cfg_path)}


# --- A1: the deployed QRC equals the validated one --------------------------------------------

class TestDesignHashVerified:
    def test_run_refuses_a_tampered_record_and_writes_nothing(self, tampered, capsys) -> None:
        from qrc_thresher.commands.run import run_handler

        assert run_handler('stm', str(tampered['cfg_path'])) == 1
        out = capsys.readouterr().out
        assert '43/138' in out and '0' * 64 in out and 'D011' in out
        assert not (tampered['dir'] / 'results' / 'runs.csv').exists()

    def test_the_engine_refuses_too(self, tampered) -> None:
        from qrc_thresher.deploy import DesignHashMismatch
        from qrc_thresher.engine import ParallelRunner

        with pytest.raises(DesignHashMismatch, match='43/138'):
            ParallelRunner(config=tampered['cfg'], max_workers=1).run_seeds(
                'stm', config_path=tampered['cfg_path']
            )
        assert not (tampered['dir'] / 'results' / 'runs.csv').exists()

    def test_ablation_checks_the_design_hash_before_the_suffix(self, tampered, capsys) -> None:
        from qrc_thresher.commands.ablation import ablation_handler

        assert ablation_handler('no_entangle', 'stm', str(tampered['cfg_path'])) == 1
        assert '43/138' in capsys.readouterr().out
        assert not (tampered['dir'] / 'results' / 'runs.csv').exists()
        # design_STM on parity (G1(b)) goes through the same check.
        assert ablation_handler('no_entangle', 'parity', str(tampered['cfg_path']),
                                design_task='stm') == 1
        assert not (tampered['dir'] / 'results' / 'runs.csv').exists()

    def test_g07_tuned_qrc_refuses_and_writes_nothing(self, tampered) -> None:
        from qrc_thresher.cli import cli
        from qrc_thresher.gates import g07

        record = _tuning().load_record(tampered['cfg_path'], 'stm')
        with pytest.raises(ValueError, match='43/138'):
            g07.evaluate_config(tampered['cfg'], model='tuned_qrc', tuning_record=record)
        result = CliRunner().invoke(cli, ['gate', 'G0.7', '--config', str(tampered['cfg_path']),
                                          '--model', 'tuned_qrc', '--tuning-config',
                                          str(tampered['cfg_path'])])
        assert result.exit_code == 2, result.output
        assert '43/138' in result.output and 'INSUFFICIENT_EVIDENCE' in result.output
        gates = tampered['dir'] / 'results' / 'gates'
        assert not gates.exists() or not list(gates.glob('G0.7.tuned_qrc.*'))

    def test_a_verified_deployment_says_so(self, one_sweep, monkeypatch) -> None:
        from qrc_thresher.deploy import qrc_deployments
        from qrc_thresher.gates import g07

        monkeypatch.chdir(one_sweep['dir'])
        cfg, cfg_path = one_sweep['cfg'], one_sweep['cfg_path']
        (dep,) = qrc_deployments(cfg, 'stm', 137, 42, cfg_path)
        entry = one_sweep['records']['stm']['qrc']['42/137']
        assert dep.details['hash_verified'] is True
        assert dep.details['design_hash'] == dep.circuit_hash == entry['circuit_hash']
        (abl,) = qrc_deployments(cfg, 'stm', 137, 42, cfg_path, ablation='no_entangle')
        assert abl.details['design_hash'] == entry['circuit_hash'] != abl.circuit_hash
        _, details = g07.tuned_feature_map(cfg, one_sweep['records']['stm'])
        for key, built in details['designs'].items():
            assert built['hash_verified'] is True
            assert built['circuit_hash'] == one_sweep['records']['stm']['qrc'][key]['circuit_hash']


# --- A2: selection is train-only by construction ---------------------------------------------

class TestSelectionScope:
    def test_poisoned_test_rows_change_nothing(self, tmp_path, monkeypatch) -> None:
        tuning = _tuning()
        cfg_path = T._write(tmp_path)
        cfg = load_config(cfg_path)
        monkeypatch.chdir(tmp_path)
        clean = tuning.tune_config(cfg, 'stm', cfg_path, sweep_id='S')
        original = tuning.task_data

        def poisoned(cfg_, task, task_seed):
            ds = original(cfg_, task, task_seed)
            u, targets = ds.u.copy(), ds.targets.copy()
            u[ds.train_end:] = np.nan
            targets[ds.train_end:] = np.nan
            return dataclasses.replace(ds, u=u, targets=targets)

        monkeypatch.setattr(tuning, 'task_data', poisoned)
        dirty = tuning.tune_config(cfg, 'stm', cfg_path, sweep_id='S')
        assert dirty == clean  # winners, every score, and record_sha256
        assert dirty['record_sha256'] == clean['record_sha256']
        assert clean['selection_scope'] == 'train_cv'
        assert clean['selection_rows'] == [T.WASHOUT, int(100 * 0.7)]

    def test_the_selection_routine_has_no_test_rows_to_touch(self) -> None:
        import inspect

        tuning = _tuning()
        params = inspect.signature(tuning.select_configuration).parameters
        assert 'train_end' not in params
        source = inspect.getsource(tuning.select_configuration)
        assert 'assert ' not in source  # structural, not asserted (vanishes under -O)
        rng = np.random.default_rng(0)
        n_train = 60
        targets = rng.normal(size=(n_train, 2))
        good = tuning.Candidate('a', 'h1', lambda: rng.normal(size=(n_train, 3)))
        too_long = tuning.Candidate('b', 'h2', lambda: rng.normal(size=(n_train + 30, 3)))
        with pytest.raises(ValueError, match='n_train'):
            tuning.select_configuration([good, too_long], targets, 10, [1.0], 5, 'stm')


# --- A3: a lost runs.csv row fails the run ----------------------------------------------------

class TestRunsCsvFailuresAreLoud:
    @staticmethod
    def _block_runs_csv(root: Path) -> Path:
        path = root / 'results' / 'runs.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.mkdir()  # a directory where the file should be: every open() fails
        return path

    def test_run_exits_one_and_leaves_no_row_anywhere(self, one_sweep, tmp_path, monkeypatch,
                                                      capsys) -> None:
        import shutil

        from qrc_thresher.commands.run import run_handler
        from qrc_thresher.db import ExperimentDB

        shutil.copytree(one_sweep['dir'], tmp_path / 'work')
        work = tmp_path / 'work'
        monkeypatch.chdir(work)
        (work / 'results' / 'runs.csv').unlink(missing_ok=True)
        blocked = self._block_runs_csv(work)
        assert run_handler('stm', str(work / 'configs' / 'tiny.yaml')) == 1
        out = capsys.readouterr().out
        assert 'results/runs.csv' in out and 'aborted' in out
        db = ExperimentDB()
        try:
            assert db.query('SELECT COUNT(*) AS n FROM runs')[0]['n'] == 0
        finally:
            db.close()
        assert blocked.is_dir() and not any(blocked.iterdir())

    def test_ablation_and_baseline_exit_one(self, one_sweep, tmp_path, monkeypatch,
                                            capsys) -> None:
        import shutil

        from qrc_thresher.commands.ablation import ablation_handler
        from qrc_thresher.commands.baseline import baseline_handler

        shutil.copytree(one_sweep['dir'], tmp_path / 'work')
        work = tmp_path / 'work'
        monkeypatch.chdir(work)
        (work / 'results' / 'runs.csv').unlink(missing_ok=True)
        self._block_runs_csv(work)
        cfg_path = str(work / 'configs' / 'tiny.yaml')
        assert ablation_handler('haar', 'stm', cfg_path) == 1
        assert baseline_handler('stm', cfg_path) == 1
        out = capsys.readouterr().out
        assert out.count('results/runs.csv') >= 2

    def test_append_to_csv_names_the_path(self, tmp_path) -> None:
        from synthetic_rows import row

        from qrc_thresher.proof.run_manifest import RunsCsvWriteError, append_to_csv

        blocked = tmp_path / 'runs.csv'
        blocked.mkdir()
        manifest = _manifest_from_row(row('stm', (42, 137), 1.0, metric='stm_memory',
                                          circuit_hash='h' * 64))
        with pytest.raises(RunsCsvWriteError, match=re.escape(blocked.as_posix())):
            append_to_csv(manifest, blocked)


def _manifest_from_row(r: dict):
    from qrc_thresher.proof.run_manifest import RunManifest

    fields = {f.name for f in dataclasses.fields(RunManifest)}
    data = {k: v for k, v in r.items() if k in fields}
    for key in ('runtime_per_stage_seconds', 'package_versions', 'secondary_metrics'):
        value = data.get(key)
        data[key] = json.loads(value) if isinstance(value, str) else (value or {})
    data['artifact_paths'] = json.loads(data['artifact_paths'])
    data['success'] = bool(data['success'])
    return RunManifest(**data)


# --- A4: a missing record or pair aborts before any row ---------------------------------------

class TestMissingDesignAbortsFirst:
    def test_a_record_missing_one_pair_writes_no_row(self, tampered, capsys) -> None:
        from qrc_thresher.commands.ablation import ablation_handler
        from qrc_thresher.commands.run import run_handler
        from qrc_thresher.engine import ParallelRunner

        assert run_handler('parity', str(tampered['cfg_path'])) == 1
        out = capsys.readouterr().out
        assert '43/138' in out and 'nothing run' in out
        assert ablation_handler('no_entangle', 'parity', str(tampered['cfg_path'])) == 1
        with pytest.raises(KeyError, match='43/138'):
            ParallelRunner(config=tampered['cfg'], max_workers=1).run_seeds(
                'parity', config_path=tampered['cfg_path']
            )
        assert not (tampered['dir'] / 'results' / 'runs.csv').exists()

    def test_a_missing_record_writes_no_row(self, tmp_path, monkeypatch, capsys) -> None:
        from qrc_thresher.commands.ablation import ablation_handler
        from qrc_thresher.commands.run import run_handler

        cfg_path = T._write(tmp_path)
        monkeypatch.chdir(tmp_path)
        assert run_handler('stm', str(cfg_path)) == 1
        assert ablation_handler('haar', 'stm', str(cfg_path)) == 1
        out = capsys.readouterr().out
        assert out.count('tuning record') >= 2
        assert not (tmp_path / 'results' / 'runs.csv').exists()


# --- A5: the family JSON's git_commit is the manifests' ---------------------------------------

class TestFamilyGitCommit:
    def test_dirty_suffix_comes_from_the_manifest_helper(self, monkeypatch) -> None:
        from qrc_thresher.gates import comparative
        from qrc_thresher.proof import run_manifest

        def fake_run(args, **kwargs):
            out = 'abc123\n' if args[:3] == ['git', 'rev-parse', 'HEAD'] else ' M src/x.py\n'
            return subprocess.CompletedProcess(args, 0, stdout=out, stderr='')

        monkeypatch.setattr(run_manifest.subprocess, 'run', fake_run)
        assert run_manifest._git_commit_hash() == 'abc123-dirty'
        assert comparative._git_commit() == 'abc123-dirty'
        assert not hasattr(comparative, 'subprocess')  # no second git implementation


# --- B6: one sweep id ---------------------------------------------------------------------------

class TestOneSweepId:
    def test_three_records_share_the_stamp_and_their_own_hashes(self, one_sweep,
                                                                 monkeypatch) -> None:
        monkeypatch.chdir(one_sweep['dir'])
        records = one_sweep['records']
        stamps = {records[t]['sweep_id'] for t in T.TASKS}
        assert len(stamps) == 1
        assert re.fullmatch(r'\d{8}T\d{6}\d*Z', stamps.pop())
        assert len({records[t]['record_sha256'] for t in T.TASKS}) == 3
        tuning = _tuning()
        for task in T.TASKS:
            on_disk = tuning.load_record(one_sweep['cfg_path'], task)
            assert on_disk['sweep_id'] == records['stm']['sweep_id']

    def test_the_tune_command_without_a_task_tunes_all_three(self, tmp_path, monkeypatch) -> None:
        from qrc_thresher.cli import cli
        from qrc_thresher.proof.run_manifest import _config_hash

        cfg_path = T._write(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(cli, ['tune', '--config', str(cfg_path)])
        assert result.exit_code == 0, result.output
        h = _config_hash(cfg_path)
        stamps = set()
        for task in T.TASKS:
            record = json.loads((tmp_path / 'results' / 'tuning' / h / f'{task}.json').read_text())
            stamps.add(record['sweep_id'])
        assert len(stamps) == 1
        assert result.output.count('tuning record:') == 3
        # `tune TASK` still rewrites one record under its own stamp (a rerun).
        again = CliRunner().invoke(cli, ['tune', 'parity', '--config', str(cfg_path)])
        assert again.exit_code == 0, again.output
        record = json.loads((tmp_path / 'results' / 'tuning' / h / 'parity.json').read_text())
        assert record['sweep_id'] not in stamps

    def test_rows_and_the_family_json_carry_the_one_stamp(self, one_sweep, tmp_path,
                                                          monkeypatch) -> None:
        import shutil

        from qrc_thresher.commands.baseline import baseline_handler
        from qrc_thresher.commands.run import run_handler
        from qrc_thresher.gates import comparative

        shutil.copytree(one_sweep['dir'], tmp_path / 'work')
        work = tmp_path / 'work'
        monkeypatch.chdir(work)
        cfg_path = work / 'configs' / 'tiny.yaml'
        (work / 'results' / 'runs.csv').unlink(missing_ok=True)
        stamp = one_sweep['records']['stm']['sweep_id']
        assert run_handler('stm', str(cfg_path)) == 0
        assert run_handler('parity', str(cfg_path), design_task='stm') == 0
        assert baseline_handler('narma', str(cfg_path)) == 0
        rows = _rows(work / 'results' / 'runs.csv')
        assert rows and all(r['sweep_id'] == stamp for r in rows)
        result, error = comparative.evaluate_config(
            cfg_path, work / 'results' / 'runs.csv', work / 'results' / 'gates'
        )
        assert error is None
        assert result['sweep_id'] == stamp  # one string, not a {task: stamp} map
        assert isinstance(result['tuning_record_sha'], dict)  # each record keeps its own hash
        assert set(result['tuning_record_sha']) == set(T.TASKS)


# --- C8: configs forbid unknown keys and silent defaults -------------------------------------

class TestConfigStrictness:
    @pytest.mark.parametrize('where, key', [
        (('training',), 'wash_out'), (('reservoir',), 'encoding_scal'), (('task',), 'delay'),
        (('seeds',), 'n_seed'), (('proof',), 'log_everything'), ((), 'gates'),
        (('measurement',), 'shots'),
    ])
    def test_a_misspelled_key_fails(self, where, key) -> None:
        raw = yaml.safe_load((CONFIGS / 'alpha_lite.yaml').read_text(encoding='utf-8'))
        target = raw
        for part in where:
            target = target[part]
        target[key] = 1
        with pytest.raises(ValueError, match=key):
            AlphaLiteConfig.model_validate(raw)

    def test_stm_needs_delay_max_and_parity_needs_its_window(self) -> None:
        from qrc_thresher.tuning import task_data

        raw = yaml.safe_load((CONFIGS / 'alpha_lite.yaml').read_text(encoding='utf-8'))
        raw['task'].pop('delay_max')
        with pytest.raises(ValueError, match='delay_max'):
            AlphaLiteConfig.model_validate(raw)
        raw['task']['name'] = 'narma'
        cfg = AlphaLiteConfig.model_validate(raw)  # a NARMA config may omit it ...
        with pytest.raises(ValueError, match='delay_max'):
            task_data(cfg, 'stm', 42)  # ... but then cannot run STM with a silent K = 20
        raw = yaml.safe_load((CONFIGS / 'alpha_lite.yaml').read_text(encoding='utf-8'))
        raw['task'].pop('parity_window')
        raw['task']['name'] = 'parity'
        with pytest.raises(ValueError, match='parity_window'):
            AlphaLiteConfig.model_validate(raw)

    def test_random_features_is_not_an_ablation(self) -> None:
        from qrc_thresher.config import AblationConfig

        with pytest.raises(ValueError):
            AblationConfig(name='random_features')
        assert AblationConfig(name='haar').name == 'haar'


# --- C9: hygiene ---------------------------------------------------------------------------------

class TestHygiene:
    def test_parse_task_name_raises_on_an_unknown_name(self) -> None:
        from qrc_thresher.task_names import parse_task_name

        assert parse_task_name('esn_parity') == {'kind': 'baseline', 'model': 'esn',
                                                 'task': 'parity'}
        for bad in ('bogus', 'esn_bogus', 'ablation', 'qrc'):
            with pytest.raises(ValueError, match='unknown task_name'):
                parse_task_name(bad)

    def test_design_flags_need_a_tuning_block(self, tmp_path, monkeypatch, capsys) -> None:
        from qrc_thresher.commands.ablation import ablation_handler
        from qrc_thresher.commands.run import run_handler
        from qrc_thresher.deploy import DesignUnavailable, qrc_deployments

        cfg_path = T._write(tmp_path, tuning=False)
        monkeypatch.chdir(tmp_path)
        cfg = load_config(cfg_path)
        assert run_handler('stm', str(cfg_path), design='default') == 1
        assert run_handler('parity', str(cfg_path), design_task='stm') == 1
        assert ablation_handler('haar', 'stm', str(cfg_path), design='default') == 1
        out = capsys.readouterr().out
        assert out.count('no tuning block') == 3
        assert not (tmp_path / 'results' / 'runs.csv').exists()
        with pytest.raises(DesignUnavailable):
            qrc_deployments(cfg, 'stm', 137, 42, cfg_path, design='default')
        (dep,) = qrc_deployments(cfg, 'stm', 137, 42, cfg_path)  # the config as is still runs
        assert dep.design == 'default'

    @pytest.mark.parametrize('name', ['family', 'G1', 'G2', 'G2.5', 'G3', 'G4'])
    def test_gate_family_needs_an_explicit_config(self, name, tmp_path, monkeypatch) -> None:
        from qrc_thresher.cli import cli

        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(cli, ['gate', name])
        assert result.exit_code == 2, result.output
        assert '--config' in result.output
        assert not (tmp_path / 'results').exists()

    def test_summary_writes_na_for_a_group_without_values(self, tmp_path, monkeypatch) -> None:
        from synthetic_rows import frame, row

        from qrc_thresher.commands.summary import summary_handler

        monkeypatch.chdir(tmp_path)
        rows = [row('stm', (42, 137), 1.5, metric='stm_memory', circuit_hash='a' * 64),
                row('esn', (42, 137), None, metric='stm_memory', circuit_hash='b' * 64)]
        (tmp_path / 'results').mkdir()
        frame(rows).to_csv(tmp_path / 'results' / 'runs.csv', index=False)
        assert summary_handler('t') == 0
        text = (tmp_path / 'results' / 'summaries' / 't_summary.md').read_text(encoding='utf-8')
        table = text.split('## Runs')[0]
        assert 'nan' not in table.lower().replace('n/a', '')
        assert re.search(r'\| esn \|.*\| n/a \| n/a \|', table)
