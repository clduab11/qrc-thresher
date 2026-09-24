"""The measurement model is a config field and a manifest field (docs/DECISIONS.md D004).

Exact expectation values are the only model until the finite-shot path exists, and every
report labels them "exact (oracle upper bound)".
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from qrc_thresher.config import (
    AlphaLiteConfig,
    MeasurementConfig,
    load_config,
    measurement_label,
)
from qrc_thresher.proof.run_manifest import (
    CSV_FIELDNAMES,
    SCHEMA_VERSION,
    append_to_csv,
    create_manifest,
)

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG = REPO_ROOT / 'configs' / 'alpha_lite.yaml'


def _manifest(**kwargs):
    return create_manifest(
        config_path=DEFAULT_CONFIG,
        circuit_hash='abc',
        task_seed=42,
        reservoir_seed=137,
        backend_device='default.qubit',
        runtime_per_stage_seconds={},
        entanglement_metric=None,
        success=True,
        failure_reason=None,
        artifact_paths=[],
        **kwargs,
    )


class TestMeasurementConfig:
    def test_default_config_declares_exact(self) -> None:
        raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding='utf-8'))
        assert raw['measurement']['model'] == 'exact'
        assert load_config(DEFAULT_CONFIG).measurement.model == 'exact'

    def test_omitted_measurement_block_means_exact(self) -> None:
        raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding='utf-8'))
        raw.pop('measurement')
        assert AlphaLiteConfig.model_validate(raw).measurement.model == 'exact'

    def test_unknown_measurement_model_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MeasurementConfig(model='shots')

    def test_exact_is_labelled_oracle_upper_bound(self) -> None:
        assert measurement_label('exact') == 'exact (oracle upper bound)'


class TestMeasurementManifest:
    def test_schema_version_includes_measurement_model(self) -> None:
        assert tuple(int(p) for p in SCHEMA_VERSION.split('.')) >= (1, 4)  # 1.4 since D013

    def test_manifest_records_measurement_model(self) -> None:
        assert _manifest().measurement_model == 'exact'

    def test_manifest_rejects_unknown_measurement_model(self) -> None:
        with pytest.raises(ValueError, match='measurement_model'):
            _manifest(measurement_model='shots')

    def test_runs_csv_has_measurement_model_column(self, tmp_path: Path) -> None:
        csv_path = tmp_path / 'runs.csv'
        append_to_csv(_manifest(), csv_path=csv_path)
        with csv_path.open(newline='') as f:
            row = next(csv.DictReader(f))
        assert 'measurement_model' in CSV_FIELDNAMES
        assert row['measurement_model'] == 'exact'

    def test_db_writer_uses_the_manifest_csv_columns(self) -> None:
        from qrc_thresher import db

        assert db._CSV_FIELDNAMES == CSV_FIELDNAMES
