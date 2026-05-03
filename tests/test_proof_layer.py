"""Tests for proof layer: manifest schema, CSV append, compute tracker."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from qrc_thresher.proof.run_manifest import (
    CSV_FIELDNAMES,
    RunManifest,
    append_to_csv,
    create_manifest,
    update_cumulative_compute,
)


class TestRunManifest:
    """Tests for run manifest creation and schema."""

    def _make_manifest(self) -> RunManifest:
        return create_manifest(
            config_path=Path('configs/alpha_lite.yaml'),
            circuit_hash='abc123def456',
            task_seed=42,
            reservoir_seed=137,
            backend_device='default.qubit',
            runtime_per_stage_seconds={'feature_extraction': 1.5, 'readout_training': 0.3},
            entanglement_metric=None,
            success=True,
            failure_reason=None,
            artifact_paths=['results/run1/stm_mc.png'],
        )

    def test_manifest_run_id_is_uuid(self) -> None:
        manifest = self._make_manifest()
        import uuid
        uuid.UUID(manifest.run_id)  # raises if not valid UUID

    def test_manifest_fields_present(self) -> None:
        manifest = self._make_manifest()
        assert manifest.task_seed == 42
        assert manifest.reservoir_seed == 137
        assert manifest.backend_device == 'default.qubit'
        assert manifest.success is True
        assert manifest.failure_reason is None
        assert 'feature_extraction' in manifest.runtime_per_stage_seconds

    def test_manifest_python_version(self) -> None:
        import sys
        manifest = self._make_manifest()
        major, minor, _ = manifest.python_version.split('.')
        assert int(major) == sys.version_info.major
        assert int(minor) == sys.version_info.minor

    def test_manifest_package_versions_dict(self) -> None:
        manifest = self._make_manifest()
        assert isinstance(manifest.package_versions, dict)
        assert 'numpy' in manifest.package_versions


class TestAppendToCSV:
    """Tests for CSV append function."""

    def test_creates_file_with_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / 'runs.csv'
            manifest = create_manifest(
                config_path=Path('configs/alpha_lite.yaml'),
                circuit_hash='abc',
                task_seed=1,
                reservoir_seed=2,
                backend_device='default.qubit',
                runtime_per_stage_seconds={},
                entanglement_metric=None,
                success=True,
                failure_reason=None,
                artifact_paths=[],
            )
            append_to_csv(manifest, csv_path=csv_path)
            assert csv_path.exists()
            with csv_path.open() as f:
                reader = csv.DictReader(f)
                assert reader.fieldnames == CSV_FIELDNAMES

    def test_appends_multiple_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / 'runs.csv'
            for i in range(3):
                manifest = create_manifest(
                    config_path=Path('configs/alpha_lite.yaml'),
                    circuit_hash=f'hash{i}',
                    task_seed=i,
                    reservoir_seed=i + 100,
                    backend_device='default.qubit',
                    runtime_per_stage_seconds={'total': float(i)},
                    entanglement_metric=None,
                    success=True,
                    failure_reason=None,
                    artifact_paths=[],
                )
                append_to_csv(manifest, csv_path=csv_path)
            with csv_path.open() as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 3

    def test_row_values_serialized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / 'runs.csv'
            manifest = create_manifest(
                config_path=Path('configs/alpha_lite.yaml'),
                circuit_hash='xyz',
                task_seed=99,
                reservoir_seed=200,
                backend_device='default.qubit',
                runtime_per_stage_seconds={'stage1': 2.5},
                entanglement_metric=0.42,
                success=False,
                failure_reason='test error',
                artifact_paths=['path/to/file.png'],
            )
            append_to_csv(manifest, csv_path=csv_path)
            with csv_path.open() as f:
                row = next(csv.DictReader(f))
            assert row['task_seed'] == '99'
            assert row['failure_reason'] == 'test error'
            assert row['entanglement_metric'] == '0.42'
            timing = json.loads(row['runtime_per_stage_seconds'])
            assert timing['stage1'] == 2.5


class TestUpdateCumulativeCompute:
    """Tests for cumulative compute tracker."""

    def test_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / 'cumulative_compute.json'
            update_cumulative_compute(10.0, json_path=json_path)
            assert json_path.exists()

    def test_increments_correctly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / 'cumulative_compute.json'
            update_cumulative_compute(10.0, json_path=json_path)
            update_cumulative_compute(20.0, json_path=json_path)
            with json_path.open() as f:
                data = json.load(f)
            assert data['total_runs'] == 2
            assert abs(data['total_compute_seconds'] - 30.0) < 1e-9

    def test_estimated_kwh_positive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            json_path = Path(tmp) / 'cumulative_compute.json'
            update_cumulative_compute(3600.0, json_path=json_path)
            with json_path.open() as f:
                data = json.load(f)
            assert data['estimated_kwh'] > 0.0
