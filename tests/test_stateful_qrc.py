"""Tests for stateful QRC feature scaffolding."""

from __future__ import annotations

import pytest


def test_stateful_import_and_validation_without_pennylane() -> None:
    try:
        from qrc_thresher.reservoirs.stateful_qrc import extract_features_stateful
    except ImportError:
        pytest.skip('stateful_qrc dependencies unavailable')

    with pytest.raises(ValueError, match='carry_depth'):
        extract_features_stateful(u=[], params=None, carry_depth=0)  # type: ignore[arg-type]
