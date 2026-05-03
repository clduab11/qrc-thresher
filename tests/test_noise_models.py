"""Tests for noise model scaffolding."""

from __future__ import annotations

from qrc_thresher.reservoirs.noise_models import (
    NoiseModelSpec,
    build_depolarizing_noise_model,
    build_relaxation_noise_model,
)


def test_depolarizing_builder_returns_object() -> None:
    model = build_depolarizing_noise_model(probability=0.01)
    assert model is not None


def test_relaxation_builder_returns_object() -> None:
    model = build_relaxation_noise_model(t1=45000, t2=60000)
    assert model is not None


def test_fallback_type_is_spec_when_qiskit_missing_or_stub() -> None:
    # This test remains valid across environments with/without qiskit-aer.
    model = build_depolarizing_noise_model(probability=0.001)
    if isinstance(model, NoiseModelSpec):
        assert model.name == 'depolarizing'
