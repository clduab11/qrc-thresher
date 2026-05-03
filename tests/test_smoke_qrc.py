"""Smoke tests for the QRC reservoir (2-qubit, default.qubit)."""

from __future__ import annotations

import numpy as np
import pytest

from qrc_thresher.reservoirs.pennylane_qrc import (
    build_reservoir_params,
    compute_circuit_hash,
    extract_features,
    train_readout,
)


class TestQRCSmoke:
    """Smoke tests for QRC feature extraction."""

    def test_z_only_shape(self) -> None:
        rng = np.random.default_rng(137)
        params = build_reservoir_params(
            n_qubits=2, depth=1, readout='z_only', backend='default.qubit', rng=rng
        )
        u = np.linspace(-1, 1, 10)
        X = extract_features(u, params)
        assert X.shape == (10, 2)

    def test_z_and_zz_shape(self) -> None:
        rng = np.random.default_rng(137)
        params = build_reservoir_params(
            n_qubits=3, depth=1, readout='z_and_zz', backend='default.qubit', rng=rng
        )
        u = np.linspace(-1, 1, 5)
        X = extract_features(u, params)
        # 3 qubits z_only + 3 pairs zz = 6
        assert X.shape == (5, 6)

    def test_features_finite(self) -> None:
        rng = np.random.default_rng(42)
        params = build_reservoir_params(
            n_qubits=2, depth=2, readout='z_only', backend='default.qubit', rng=rng
        )
        u = np.random.default_rng(0).uniform(-1, 1, size=20)
        X = extract_features(u, params)
        assert np.isfinite(X).all()

    def test_features_bounded(self) -> None:
        """Pauli-Z expectation values are in [-1, 1]."""
        rng = np.random.default_rng(42)
        params = build_reservoir_params(
            n_qubits=2, depth=1, readout='z_only', backend='default.qubit', rng=rng
        )
        u = np.linspace(-1, 1, 20)
        X = extract_features(u, params)
        assert X.min() >= -1.0 - 1e-9
        assert X.max() <= 1.0 + 1e-9

    def test_circuit_hash_deterministic(self) -> None:
        rng = np.random.default_rng(137)
        params = build_reservoir_params(
            n_qubits=2, depth=1, readout='z_only', backend='default.qubit', rng=rng
        )
        h1 = compute_circuit_hash(params)
        h2 = compute_circuit_hash(params)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_circuit_hash_changes_with_params(self) -> None:
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(2)
        p1 = build_reservoir_params(
            n_qubits=2, depth=1, readout='z_only', backend='default.qubit', rng=rng1
        )
        p2 = build_reservoir_params(
            n_qubits=2, depth=1, readout='z_only', backend='default.qubit', rng=rng2
        )
        assert compute_circuit_hash(p1) != compute_circuit_hash(p2)

    def test_readout_training(self) -> None:
        rng = np.random.default_rng(42)
        params = build_reservoir_params(
            n_qubits=2, depth=1, readout='z_only', backend='default.qubit', rng=rng
        )
        u = rng.uniform(-1, 1, size=50)
        X = extract_features(u, params)
        y = u[:50]  # trivial target
        model = train_readout(X[:40], y[:40], ridge_alphas=[1e-4, 1e-2, 1.0], cv_folds=3)
        preds = model.predict(X[40:])
        assert preds.shape == (10,)
        assert np.isfinite(preds).all()

    def test_deterministic_same_params(self) -> None:
        rng = np.random.default_rng(137)
        params = build_reservoir_params(
            n_qubits=2, depth=1, readout='z_only', backend='default.qubit', rng=rng
        )
        u = np.array([0.3, -0.5, 0.7])
        X1 = extract_features(u, params)
        X2 = extract_features(u, params)
        assert np.allclose(X1, X2)

    @pytest.mark.slow
    def test_4qubit_depth3(self) -> None:
        rng = np.random.default_rng(42)
        params = build_reservoir_params(
            n_qubits=4, depth=3, readout='z_only', backend='default.qubit', rng=rng
        )
        u = rng.uniform(-1, 1, size=30)
        X = extract_features(u, params)
        assert X.shape == (30, 4)
        assert np.isfinite(X).all()
