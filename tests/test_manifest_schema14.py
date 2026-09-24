"""Manifest schema 1.4 (docs/DECISIONS.md D013): secondary_metrics, device, precision, design
and sweep_id join runs.csv; STM rows carry stm_memory (k >= 1) as the primary metric with
mc_total and mc_k0 as secondary metrics; G5 reads stm_memory; the header guard refuses a 1.3
file; db.py builds its rows through run_manifest.manifest_row.
"""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from qrc_thresher.proof import run_manifest
from qrc_thresher.proof.run_manifest import (
    CSV_FIELDNAMES,
    SCHEMA_VERSION,
    RunsCsvSchemaError,
    append_to_csv,
    create_manifest,
)

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG = REPO_ROOT / 'configs' / 'alpha_lite.yaml'
NEW_COLUMNS = ['secondary_metrics', 'device', 'precision', 'design', 'sweep_id',
               'tuning_record_sha']  # tuning_record_sha: PI ruling 3
SCHEMA_1_3_HEADER = [c for c in CSV_FIELDNAMES if c not in NEW_COLUMNS]


def _manifest(**kwargs):
    return create_manifest(
        config_path=DEFAULT_CONFIG, circuit_hash='abc', task_seed=42, reservoir_seed=137,
        backend_device='default.qubit', runtime_per_stage_seconds={}, entanglement_metric=None,
        success=True, failure_reason=None, artifact_paths=[], **kwargs,
    )


class TestSchema:
    def test_version_and_columns(self) -> None:
        assert SCHEMA_VERSION == '1.4'
        assert CSV_FIELDNAMES[-6:] == NEW_COLUMNS
        assert len(CSV_FIELDNAMES) == len(SCHEMA_1_3_HEADER) + 6 == 31

    def test_the_constants_and_defaults(self) -> None:
        m = _manifest()
        assert (m.device, m.precision) == ('cpu', 'float64')
        assert (run_manifest.DEVICE, run_manifest.PRECISION) == ('cpu', 'float64')
        assert m.design == 'default' and m.sweep_id == '' and m.secondary_metrics == {}
        assert m.tuning_record_sha == ''

    def test_design_labels_are_the_registered_three(self) -> None:
        # PI ruling 1: the w = 1 default row carries its own label and is reporting-only.
        assert set(run_manifest.DESIGNS) == {'tuned', 'default', 'default_w1', 'inherited'}
        for design in run_manifest.DESIGNS:
            m = _manifest(design=design, sweep_id='20260923T000000Z', tuning_record_sha='a' * 64)
            assert (m.design, m.tuning_record_sha) == (design, 'a' * 64)
        with pytest.raises(ValueError, match='design'):
            _manifest(design='best')

    def test_secondary_metrics_are_written_as_json(self, tmp_path) -> None:
        csv_path = tmp_path / 'runs.csv'
        m = _manifest(task_name='stm', primary_metric_name='stm_memory', primary_metric_value=2.5,
                      secondary_metrics={'mc_total': 3.2, 'mc_k0': 0.7}, design='tuned',
                      sweep_id='20260923T000000Z')
        append_to_csv(m, csv_path=csv_path)
        with csv_path.open(newline='') as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert reader.fieldnames == CSV_FIELDNAMES
        assert json.loads(row['secondary_metrics']) == {'mc_total': 3.2, 'mc_k0': 0.7}
        assert (row['device'], row['precision'], row['design'], row['sweep_id']) == (
            'cpu', 'float64', 'tuned', '20260923T000000Z'
        )
        assert row['tuning_record_sha'] == ''

    def test_the_header_guard_refuses_a_1_3_file(self, tmp_path) -> None:
        csv_path = tmp_path / 'runs.csv'
        with csv_path.open('w', newline='') as f:
            csv.writer(f).writerow(SCHEMA_1_3_HEADER)
        before = csv_path.read_bytes()
        with pytest.raises(RunsCsvSchemaError) as excinfo:
            append_to_csv(_manifest(), csv_path=csv_path)
        message = str(excinfo.value)
        assert 'design' in message and 'sweep_id' in message and 'schema 1.4' in message
        assert 'tuning_record_sha' in message
        assert csv_path.read_bytes() == before


class TestOneRowBuilder:
    def test_db_writes_the_same_row_as_append_to_csv(self, tmp_path, monkeypatch) -> None:
        from qrc_thresher.db import ExperimentDB

        m = _manifest(task_name='stm', primary_metric_name='stm_memory', primary_metric_value=2.5,
                      secondary_metrics={'mc_total': 3.2, 'mc_k0': 0.7}, design='tuned',
                      sweep_id='s')
        direct = tmp_path / 'direct.csv'
        append_to_csv(m, csv_path=direct)
        monkeypatch.chdir(tmp_path)
        db = ExperimentDB(db_path=str(tmp_path / 'results' / 'experiments.db'))
        try:
            db.insert(m)
        finally:
            db.close()
        via_db = (tmp_path / 'results' / 'runs.csv').read_text(encoding='utf-8')
        assert via_db == direct.read_text(encoding='utf-8')
        assert run_manifest.manifest_row(m)['secondary_metrics'] == json.dumps(m.secondary_metrics)
        assert list(run_manifest.manifest_row(m)) == CSV_FIELDNAMES

    def test_db_has_no_row_builder_of_its_own(self) -> None:
        from qrc_thresher import db

        tree = ast.parse(Path(db.__file__).read_text(encoding='utf-8'))
        literal_rows = [
            node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Dict)
            and any(isinstance(k, ast.Constant) and k.value == 'primary_metric_value'
                    for k in node.keys)
        ]
        assert literal_rows == [], literal_rows
        assert db._CSV_FIELDNAMES == CSV_FIELDNAMES


class TestSTMMetrics:
    def test_stm_rows_carry_stm_memory_with_mc_total_and_mc_k0(self, tmp_path,
                                                                 monkeypatch) -> None:
        from qrc_thresher.config import load_config
        from qrc_thresher.engine import ParallelRunner

        raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding='utf-8'))
        raw['experiment_name'] = 'tiny_stm_metrics'
        raw['task'].update({'length': 100, 'delay_max': 5})
        raw['reservoir'].update({'n_qubits': 2, 'depth': 1, 'window': 2})
        raw['training']['washout'] = 20
        raw['seeds']['n_seeds'] = 1
        cfg_path = tmp_path / 'configs' / 'tiny.yaml'
        cfg_path.parent.mkdir(parents=True)
        cfg_path.write_text(yaml.safe_dump(raw), encoding='utf-8')
        monkeypatch.chdir(tmp_path)
        manifests = ParallelRunner(config=load_config(cfg_path), max_workers=1).run_seeds(
            'stm', config_path=cfg_path
        )
        m = manifests[0]
        assert m.success, m.failure_reason
        assert m.primary_metric_name == 'stm_memory'
        assert set(m.secondary_metrics) == {'mc_total', 'mc_k0'}
        assert 0.0 <= m.secondary_metrics['mc_k0'] <= 1.0
        assert m.secondary_metrics['mc_total'] == pytest.approx(
            m.primary_metric_value + m.secondary_metrics['mc_k0']
        )
        with (Path('results') / 'runs.csv').open(newline='') as f:
            row = next(csv.DictReader(f))
        assert row['primary_metric_name'] == 'stm_memory'
        assert json.loads(row['secondary_metrics']) == pytest.approx(m.secondary_metrics)

    def test_g5_reads_stm_memory(self) -> None:
        from qrc_thresher.commands.gate import _evaluate_gate_g5

        def rows(metric):
            return pd.DataFrame([
                {'run_id': f'r{i}', 'task_name': 'stm', 'backend_device': 'default.qubit',
                 'circuit_hash': 'h', 'primary_metric_name': metric,
                 'primary_metric_value': 1.0 + i, 'success': True}
                for i in range(3)
            ])

        result, evidence, _ = _evaluate_gate_g5(rows('stm_memory'))
        assert result == 'PASS', evidence
        assert evidence['stm_memory_by_backend']['default.qubit']['n'] == 3
        assert _evaluate_gate_g5(rows('mc'))[0] == 'INSUFFICIENT_EVIDENCE'
