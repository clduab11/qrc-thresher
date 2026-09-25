"""Measurement labels on rows, reports and console output, and gate-JSON provenance (CP5 item B;
docs/DECISIONS.md D018).

- ``measurement_label('classical')`` is 'classical, no measurement cost'; MeasurementConfig keeps
  Literal['exact'].
- ESN and RKS rows written by `baseline` carry 'classical'; QRC and ablation rows keep 'exact'.
- The family's per-arm labels are tested beside the family fixture in test_comparative_family.py
  (TestTablesNameTheirRows).
- Legacy gate JSONs (G0.5 through the CLI, or ``_write_gate_result`` directly) carry
  git_commit_hash, config_hash or null plus a reason, and the measurement label; contain no NaN;
  are timestamped and never overwritten.
- The console output of run, ablation, baseline, tune, gate family, G0.5 and G0.7 carries the
  label (CliRunner on tiny configs, working directory in tmp_path).
"""

from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path

import pytest
import test_comparative_family as F
import yaml
from click.testing import CliRunner
from pydantic import ValidationError

from qrc_thresher.cli import cli
from qrc_thresher.config import MEASUREMENT_LABELS, MeasurementConfig, measurement_label

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG = REPO_ROOT / 'configs' / 'alpha_lite.yaml'
EXACT = 'exact (oracle upper bound)'
CLASSICAL = 'classical, no measurement cost'
TINY_GRID = {'spectral_radius': [0.9], 'input_scaling': [0.1, 1.0], 'leak_rate': [1.0]}
TINY_TUNING = {
    'qrc': {'depth': [1, 2], 'window': [2], 'encoding_scale': [math.pi / 2]},
    'esn': {'spectral_radius': [0.9], 'input_scaling': [0.1, 1.0], 'leak_rate': [1.0]},
    'rks': {'sigma': [0.5, 2.0], 'window': [2]},
}


def _tiny(tmp_path: Path, *, tuning: bool = False) -> Path:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding='utf-8'))
    raw['experiment_name'] = 'tiny_labels'
    raw['task'].update({'length': 100, 'delay_max': 5, 'parity_window': 2})
    raw['reservoir'].update({'n_qubits': 2, 'depth': 1, 'window': 2})
    raw['training']['washout'] = 20
    raw['seeds']['n_seeds'] = 1  # one pair: the labels, not the statistics, are under test
    raw['baseline']['enabled'] = ['esn', 'random_features']
    if tuning:
        raw['tuning'] = copy.deepcopy(TINY_TUNING)
        raw['baseline'].pop('esn_grid')
    else:
        raw['baseline']['esn_grid'] = TINY_GRID
    path = tmp_path / 'configs' / 'tiny.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw), encoding='utf-8')
    return path


def _rows(path: Path = Path('results') / 'runs.csv') -> list:
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


class TestLabels:
    def test_the_classical_label(self) -> None:
        assert measurement_label('classical') == CLASSICAL
        assert MEASUREMENT_LABELS == {'exact': EXACT, 'classical': CLASSICAL}
        assert measurement_label('exact') == EXACT

    def test_measurement_config_stays_exact_only(self) -> None:
        assert MeasurementConfig().model == 'exact'
        with pytest.raises(ValidationError):
            MeasurementConfig(model='classical')

    def test_create_manifest_accepts_the_classical_label(self) -> None:
        from qrc_thresher.proof.run_manifest import create_manifest

        manifest = create_manifest(
            config_path=DEFAULT_CONFIG, circuit_hash='abc', task_seed=42, reservoir_seed=137,
            backend_device='numpy_esn', runtime_per_stage_seconds={}, entanglement_metric=None,
            success=True, failure_reason=None, artifact_paths=[], measurement_model='classical',
        )
        assert manifest.measurement_model == 'classical'
        with pytest.raises(ValueError, match='measurement_model'):
            create_manifest(
                config_path=DEFAULT_CONFIG, circuit_hash='abc', task_seed=42, reservoir_seed=137,
                backend_device='numpy_esn', runtime_per_stage_seconds={},
                entanglement_metric=None, success=True, failure_reason=None, artifact_paths=[],
                measurement_model='shots',
            )

    def test_synthetic_rows_stand_for_rows_written_before_d018(self) -> None:
        # A green guard: synthetic_rows.py keeps 'exact' on every row, classical arms included.
        from synthetic_rows import classical_arm

        arm = classical_arm('esn', [1.0, 2.0], metric='stm_memory', pairs=F.PAIRS[:2])
        assert set(arm['measurement_model']) == {'exact'}


class TestRowsAndConsole:
    def test_baseline_rows_are_classical_and_qrc_and_ablation_rows_stay_exact(
        self, tmp_path, monkeypatch
    ) -> None:
        cfg_path = _tiny(tmp_path)
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        out = {}
        for name, args in (
            ('run', ['run', 'stm', '--config', str(cfg_path)]),
            ('ablation', ['ablation', 'no_entangle', 'stm', '--config', str(cfg_path)]),
            ('baseline', ['baseline', 'stm', '--config', str(cfg_path)]),
        ):
            result = runner.invoke(cli, args)
            assert result.exit_code == 0, (name, result.output)
            out[name] = result.output
        rows = _rows()
        by_task = {}
        for r in rows:
            by_task.setdefault(r['task_name'], set()).add(r['measurement_model'])
        assert by_task['stm'] == {'exact'}
        assert by_task['ablation:no_entangle'] == {'exact'}
        assert by_task['esn'] == {'classical'} and by_task['rks'] == {'classical'}
        # Console: every block that prints a metric value carries the label.
        assert EXACT in out['run'] and EXACT in out['ablation']
        assert CLASSICAL in out['baseline']
        assert EXACT not in out['baseline']

    def test_tune_prints_the_label(self, tmp_path, monkeypatch) -> None:
        cfg_path = _tiny(tmp_path, tuning=True)
        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(cli, ['tune', 'stm', '--config', str(cfg_path)])
        assert result.exit_code == 0, result.output
        assert 'tuning record:' in result.output
        assert EXACT in result.output and CLASSICAL in result.output
        records = list((tmp_path / 'results' / 'tuning').rglob('stm.json'))
        assert len(records) == 1
        assert b'\r' not in records[0].read_bytes()  # the tuning writer uses LF (ruling 12a)

    def test_gate_family_prints_the_label(self, tmp_path, monkeypatch) -> None:
        from qrc_thresher.gates import comparative

        runs = F._sweep()
        result = F._evaluate(runs)
        monkeypatch.setattr(comparative, 'evaluate_config', lambda *a, **k: (result, None))
        monkeypatch.chdir(tmp_path)
        cfg_path = _tiny(tmp_path)
        output = CliRunner().invoke(cli, ['gate', 'family', '--config', str(cfg_path)])
        assert output.exit_code == 0, output.output
        assert EXACT in output.output
        assert output.output.count('Gate G') == 5

    def test_g07_console_prints_the_label(self, monkeypatch, tmp_path) -> None:
        from qrc_thresher.commands import gate

        monkeypatch.chdir(tmp_path)
        fake = ('PASS', {'message': 'both clauses pass', 'stm_clause': 'PASS',
                         'parity_clause': 'PASS', 'measurement_label': EXACT,
                         'json': 'results/gates/G0.7.x.json', 'figure': 'results/gates/x.png'},
                [])
        monkeypatch.setattr(gate, '_evaluate_gate_g07', lambda **kwargs: fake)
        result = CliRunner().invoke(cli, ['gate', 'G0.7', '--config', str(DEFAULT_CONFIG)])
        assert result.exit_code == 0, result.output
        assert EXACT in result.output


class TestLegacyGateJson:
    def test_write_gate_result_carries_provenance_and_never_overwrites(self, tmp_path) -> None:
        from qrc_thresher.commands.gate import _write_gate_result

        first = _write_gate_result(tmp_path, 'G5', 'PASS', {'x': 1.0}, ['r1', 'r2'],
                                   config_path=DEFAULT_CONFIG, measurement_model='exact')
        second = _write_gate_result(tmp_path, 'G5', 'PASS', {'x': 1.0}, ['r1', 'r2'],
                                    config_path=DEFAULT_CONFIG, measurement_model='exact')
        assert first != second and first.exists() and second.exists()
        assert first.name.startswith('G5.') and first.name.endswith('.json')
        assert not (tmp_path / 'G5.json').exists()
        data = json.loads(first.read_bytes().decode('ascii'))
        assert (data['gate'], data['result'], data['run_ids']) == ('G5', 'PASS', ['r1', 'r2'])
        assert isinstance(data['git_commit_hash'], str) and data['git_commit_hash']
        assert len(data['config_hash']) == 64 and data['config_hash_reason'] is None
        assert data['measurement_model'] == 'exact' and data['measurement_label'] == EXACT
        assert b'\r' not in first.read_bytes()  # LF on every platform

    def test_write_gate_result_without_a_config_or_a_reservoir(self, tmp_path) -> None:
        from qrc_thresher.commands.gate import _write_gate_result

        path = _write_gate_result(tmp_path, 'G6', 'INSUFFICIENT_EVIDENCE', {'message': 'x'}, [])
        data = json.loads(path.read_text(encoding='utf-8'))
        assert data['config_hash'] is None and data['config_hash_reason']
        assert data['measurement_model'] is None and data['measurement_label'] is None
        assert 'git_commit_hash' in data

    def test_write_gate_result_refuses_nan(self, tmp_path) -> None:
        from qrc_thresher.commands.gate import _write_gate_result

        with pytest.raises(ValueError, match='NaN|nan|allow_nan|finite'):
            _write_gate_result(tmp_path, 'G6', 'FAIL', {'value': float('nan')}, [])
        assert list(tmp_path.glob('*.json')) == []

    def test_g05_writes_a_timestamped_json_with_the_label(self, tmp_path, monkeypatch) -> None:
        from qrc_thresher.commands import gate

        monkeypatch.setattr(gate, 'G05_TRIPLES', ((2, 1, 2026),))
        monkeypatch.setattr(gate, 'G05_SCALE_CASES', ())
        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(cli, ['gate', 'G0.5'])
        assert result.exit_code in (0, 1), result.output
        assert EXACT in result.output
        files = sorted((tmp_path / 'results' / 'gates').glob('G0.5.*.json'))
        assert len(files) == 1 and not (tmp_path / 'results' / 'gates' / 'G0.5.json').exists()
        data = json.loads(files[0].read_text(encoding='utf-8'))
        assert data['gate'] == 'G0.5' and data['evidence']['n_cases'] == 4
        assert data['measurement_model'] == 'exact' and data['measurement_label'] == EXACT
        assert 'git_commit_hash' in data and 'config_hash' in data
        assert 'NaN' not in files[0].read_text(encoding='utf-8')


class TestFailedRowsKeepTheirTask:
    """CP5a ruling 5: a failed ablation or baseline row writes primary_metric_name = its task's
    metric and an empty value."""

    def _boom(self, *args, **kwargs):
        raise RuntimeError('forced failure for the test')

    def test_a_failed_ablation_row(self, tmp_path, monkeypatch) -> None:
        from qrc_thresher import deploy

        cfg_path = _tiny(tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(deploy, 'fit_and_score', self._boom)
        result = CliRunner().invoke(cli, ['ablation', 'no_entangle', 'parity', '--config',
                                          str(cfg_path)])
        assert result.exit_code == 1, result.output
        rows = _rows()
        assert rows and all(r['success'] == 'False' for r in rows)
        assert all(r['primary_metric_name'] == 'accuracy' for r in rows)
        assert all(r['primary_metric_value'] == '' for r in rows)
        assert all(r['measurement_model'] == 'exact' for r in rows)

    def test_a_failed_baseline_row(self, tmp_path, monkeypatch) -> None:
        from qrc_thresher import deploy

        cfg_path = _tiny(tmp_path)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(deploy, 'fit_and_score', self._boom)
        result = CliRunner().invoke(cli, ['baseline', 'narma', '--config', str(cfg_path)])
        assert result.exit_code == 1, result.output
        rows = _rows()
        assert {r['task_name'] for r in rows} == {'esn_narma', 'rks_narma'}
        assert all(r['success'] == 'False' for r in rows)
        assert all(r['primary_metric_name'] == 'nrmse' for r in rows)
        assert all(r['primary_metric_value'] == '' for r in rows)
        assert all(r['measurement_model'] == 'classical' for r in rows)


class TestG07TopLevelKeys:
    """CP5a ruling 7: every G0.7 JSON written after D018 carries top-level config_hash and
    git_commit_hash; environment and model_details are unchanged. Ruling 12a: LF endings."""

    def test_the_written_json_carries_the_keys(self, tmp_path, monkeypatch) -> None:
        import test_gate_g07 as G

        from qrc_thresher.commands import gate
        from qrc_thresher.gates import g07
        from qrc_thresher.proof.run_manifest import _config_hash

        result = G._evaluate('delay_line')  # cached across the suite by test_gate_g07
        monkeypatch.setattr(g07, 'evaluate_config', lambda *a, **k: result)
        verdict, evidence, _ = gate._evaluate_gate_g07(config_path=DEFAULT_CONFIG,
                                                       out_dir=tmp_path)
        assert verdict == result['result']
        path = Path(evidence['json'])
        raw = path.read_bytes()
        assert b'\r' not in raw  # the G0.7 writer uses LF (ruling 12a)
        data = json.loads(raw.decode('utf-8'))
        assert data['config_hash'] == _config_hash(DEFAULT_CONFIG)
        assert data['git_commit_hash'] == data['environment']['git_commit_hash']
        assert data['environment'] == result['environment']
        assert data['model_details'] == result['model_details']
        for key in ('gate', 'result', 'clauses', 'figure', 'measurement_label'):
            assert key in data, key


class TestE3Hardening:
    """CP5a ruling 12 (b) and (c): cli_command records the program's basename; config_path is
    repo-relative POSIX when the config lies under the repository root."""

    def test_cli_command_records_the_basename_plus_the_arguments(self, tmp_path,
                                                                  monkeypatch) -> None:
        import sys

        from qrc_thresher.proof import run_manifest

        program = str(tmp_path / 'bin' / 'qrc-thresher.exe')  # the host's own separator
        monkeypatch.setattr(sys, 'argv', [program, 'run', 'stm',
                                          '--config', 'configs/comparative.yaml'])
        assert run_manifest._cli_command() == \
            'qrc-thresher.exe run stm --config configs/comparative.yaml'
        monkeypatch.setattr(sys, 'argv', ['/opt/venv/bin/qrc-thresher', 'tune'])
        assert run_manifest._cli_command() == 'qrc-thresher tune'

    def test_config_path_is_repo_relative_posix_under_the_root(self, tmp_path, monkeypatch):
        from qrc_thresher.proof import run_manifest

        root = tmp_path / 'repo'
        (root / 'configs').mkdir(parents=True)
        inside = root / 'configs' / 'tiny.yaml'
        inside.write_text(DEFAULT_CONFIG.read_text(encoding='utf-8'), encoding='utf-8')
        outside = tmp_path / 'elsewhere' / 'tiny.yaml'
        outside.parent.mkdir()
        outside.write_text(DEFAULT_CONFIG.read_text(encoding='utf-8'), encoding='utf-8')
        monkeypatch.setattr(run_manifest, '_REPO_ROOT', root)  # never by writing under the repo

        def manifest(path):
            return run_manifest.create_manifest(
                config_path=path, circuit_hash='abc', task_seed=42, reservoir_seed=137,
                backend_device='default.qubit', runtime_per_stage_seconds={},
                entanglement_metric=None, success=True, failure_reason=None, artifact_paths=[],
            )

        assert manifest(inside).config_path == 'configs/tiny.yaml'
        assert manifest(inside.resolve()).config_path == 'configs/tiny.yaml'
        assert manifest(outside).config_path == str(outside)  # outside the root: as given
