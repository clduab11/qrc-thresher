"""results/runs.csv is never appended under a mismatched header.

Every writer of results/runs.csv (append_to_csv and ExperimentDB) must refuse a file whose
header differs from CSV_FIELDNAMES, name both schemas, and tell the user to move or migrate
the file, instead of appending misaligned rows.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from qrc_thresher.proof.run_manifest import (
    CSV_FIELDNAMES,
    SCHEMA_VERSION,
    RunsCsvSchemaError,
    append_to_csv,
    create_manifest,
)

# The schema 1.1 header: everything before measurement_model was added.
SCHEMA_1_1_HEADER = [c for c in CSV_FIELDNAMES if c != 'measurement_model']


def _manifest():
    return create_manifest(
        config_path=Path('configs/alpha_lite.yaml'),
        circuit_hash='abc',
        task_seed=42,
        reservoir_seed=137,
        backend_device='default.qubit',
        runtime_per_stage_seconds={},
        entanglement_metric=None,
        success=True,
        failure_reason=None,
        artifact_paths=[],
    )


def _write_header(path: Path, header: list) -> None:
    with path.open('w', newline='') as f:
        csv.writer(f).writerow(header)


def _assert_names_both_schemas(message: str) -> None:
    assert 'measurement_model' in message  # the column that differs
    assert f'schema {SCHEMA_VERSION}' in message
    assert 'existing header' in message
    assert 'move' in message.lower()
    assert 'migrate' in message.lower()


def test_schema_error_is_a_value_error() -> None:
    assert issubclass(RunsCsvSchemaError, ValueError)


def test_append_refuses_a_mismatched_header(tmp_path: Path) -> None:
    csv_path = tmp_path / 'runs.csv'
    _write_header(csv_path, SCHEMA_1_1_HEADER)
    before = csv_path.read_bytes()
    with pytest.raises(RunsCsvSchemaError) as excinfo:
        append_to_csv(_manifest(), csv_path=csv_path)
    _assert_names_both_schemas(str(excinfo.value))
    assert csv_path.read_bytes() == before  # nothing appended


def test_append_accepts_a_matching_header(tmp_path: Path) -> None:
    csv_path = tmp_path / 'runs.csv'
    _write_header(csv_path, CSV_FIELDNAMES)
    append_to_csv(_manifest(), csv_path=csv_path)
    with csv_path.open(newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == CSV_FIELDNAMES
    assert len(rows) == 1


def test_db_refuses_a_mismatched_header(tmp_path: Path, monkeypatch) -> None:
    from qrc_thresher.db import ExperimentDB

    monkeypatch.chdir(tmp_path)  # ExperimentDB writes results/runs.csv under the cwd
    csv_path = tmp_path / 'results' / 'runs.csv'
    csv_path.parent.mkdir()
    _write_header(csv_path, SCHEMA_1_1_HEADER)
    before = csv_path.read_bytes()
    db = ExperimentDB(db_path=str(tmp_path / 'results' / 'experiments.db'))
    try:
        with pytest.raises(RunsCsvSchemaError) as excinfo:
            db._append_to_csv(_manifest())
        _assert_names_both_schemas(str(excinfo.value))
        with pytest.raises(RunsCsvSchemaError):
            db.insert(_manifest())
        # The refusal comes before the database write, so the two stores stay consistent.
        assert db.query('SELECT COUNT(*) AS n FROM runs')[0]['n'] == 0
    finally:
        db.close()
    assert csv_path.read_bytes() == before
