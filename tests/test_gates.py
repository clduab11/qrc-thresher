"""Fixture-based unit tests for gate evaluators."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

from qrc_thresher.commands.gate import (
    _evaluate_gate_g0,
    _evaluate_gate_g6,
    _evaluate_gate_g7,
    _write_gate_result,
)

# G1, G2, G2.5, G3 and G4 are the members of the comparative family (docs/DECISIONS.md D013,
# D014); their tests are in tests/test_comparative_family.py and tests/test_paired_stats.py.


def make_runs_df(rows: list[dict]) -> pd.DataFrame:
    """Create a DataFrame from list of dicts matching runs.csv schema."""
    columns = [
        'run_id',
        'timestamp_utc',
        'git_commit_hash',
        'git_branch',
        'config_path',
        'config_hash',
        'circuit_hash',
        'task_seed',
        'reservoir_seed',
        'python_version',
        'backend_device',
        'task_name',
        'primary_metric_name',
        'primary_metric_value',
        'success',
    ]
    return pd.DataFrame(rows, columns=columns)


class TestG0:
    """Tests for G0: health report overall = PASS."""

    def test_g0_pass(self, monkeypatch):
        mock_report = {
            'overall': 'PASS',
            'timestamp_utc': '2026-05-03T12:00:00Z',
        }
        monkeypatch.setattr(
            'qrc_thresher.commands.gate._latest_health_report',
            lambda: mock_report,
        )
        result, evidence, run_ids = _evaluate_gate_g0()
        assert result == 'PASS'
        assert evidence['overall'] == 'PASS'

    def test_g0_fail(self, monkeypatch):
        mock_report = {
            'overall': 'FAIL',
            'timestamp_utc': '2026-05-03T12:00:00Z',
        }
        monkeypatch.setattr(
            'qrc_thresher.commands.gate._latest_health_report',
            lambda: mock_report,
        )
        result, evidence, run_ids = _evaluate_gate_g0()
        assert result == 'FAIL'

    def test_g0_no_report(self, monkeypatch):
        monkeypatch.setattr(
            'qrc_thresher.commands.gate._latest_health_report',
            lambda: None,
        )
        result, evidence, run_ids = _evaluate_gate_g0()
        assert result == 'INSUFFICIENT_EVIDENCE'


class TestG05:
    """Tests for G0.5: basic execution without error.

    G0.5 is not yet fully implemented; these tests verify it returns
    a valid verdict string and does not raise.
    """

    def test_g05_returns_expected_verbs(self):
        try:
            from qrc_thresher.commands.gate import _evaluate_gate_g05
        except ImportError:
            pytest.skip('_evaluate_gate_g05 not defined in commands.gate')
        try:
            result, evidence, run_ids = _evaluate_gate_g05()
        except ImportError as exc:
            if 'pennylane' in str(exc):
                pytest.skip('pennylane not available')
            raise
        assert result in ('PASS', 'FAIL', 'INSUFFICIENT_EVIDENCE')
        assert isinstance(evidence, dict)
        assert isinstance(run_ids, list)


class TestG5:
    """Tests for G5: basic execution without error.

    G5 is not yet fully implemented; these tests verify it returns
    a valid verdict string and does not raise.
    """

    def test_g5_returns_expected_verdict(self):
        try:
            from qrc_thresher.commands.gate import _evaluate_gate_g5
        except ImportError:
            pytest.skip('_evaluate_gate_g5 not defined in commands.gate')
        qrc_runs = [
            {'run_id': f'qrc_{i}', 'task_name': 'stm',
             'primary_metric_name': 'mc', 'primary_metric_value': 1.5,
             'success': True}
            for i in range(5)
        ]
        df = make_runs_df(qrc_runs)
        result, evidence, run_ids = _evaluate_gate_g5(df)
        assert result in ('PASS', 'FAIL', 'INSUFFICIENT_EVIDENCE')
        assert isinstance(evidence, dict)
        assert isinstance(run_ids, list)


class TestG6:
    """Tests for G6 readiness checklist gate."""

    def test_g6_returns_valid_verdict(self):
        df = make_runs_df([])
        result, evidence, run_ids = _evaluate_gate_g6(df)
        assert result in ('PASS', 'FAIL', 'INSUFFICIENT_EVIDENCE')
        assert isinstance(evidence, dict)
        assert isinstance(run_ids, list)


class TestG7:
    """Tests for G7 calibration validator gate."""

    def test_g7_insufficient_without_dir(self, tmp_path):
        missing = tmp_path / 'no_calibration_here'
        result, evidence, run_ids = _evaluate_gate_g7(calibration_dir=missing)
        assert result == 'INSUFFICIENT_EVIDENCE'

    def test_g7_pass_with_valid_calibration(self, tmp_path):
        cal_dir = tmp_path / 'calibration'
        cal_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            'backend': 'ibm_fake',
            'timestamp_utc': (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
            't1': 45000,
            't2': 60000,
            'readout_error': 0.02,
        }
        (cal_dir / 'calibration.json').write_text(json.dumps(payload), encoding='utf-8')
        result, evidence, run_ids = _evaluate_gate_g7(calibration_dir=cal_dir, max_age_days=30)
        assert result == 'PASS'


class TestWriteGateResult:
    """Tests for _write_gate_result: ensure no actual disk I/O occurs."""

    def test_write_gate_result_does_not_raise(self, monkeypatch, tmp_path):
        def mock_open(path, mode):
            return MagicMock()

        monkeypatch.setattr('qrc_thresher.commands.gate.Path.open', mock_open)

        _write_gate_result(
            gates_dir=tmp_path,
            name='G1',
            result='PASS',
            evidence={'test': 'data'},
            run_ids=['run_1', 'run_2'],
        )
