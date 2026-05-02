"""Tests for full pipeline reproducibility."""

from __future__ import annotations

import numpy as np

from qrc_thresher.metrics.scoring import memory_capacity
from qrc_thresher.reservoirs.pennylane_qrc import (
    build_reservoir_params,
    compute_circuit_hash,
    extract_features,
    train_readout,
)
from qrc_thresher.tasks.stm import generate_stm


def _run_stm_pipeline(task_seed: int, reservoir_seed: int) -> dict:
    """Run a full STM pipeline and return results dict."""
    rng_task = np.random.default_rng(task_seed)
    rng_res = np.random.default_rng(reservoir_seed)

    ds = generate_stm(length=100, delay_max=5, train_frac=0.7, rng=rng_task)
    params = build_reservoir_params(
        n_qubits=2, depth=1, readout='z_only', backend='default.qubit', rng=rng_res
    )
    X = extract_features(ds.u, params)
    model = train_readout(
        X[: ds.train_end],
        ds.targets[: ds.train_end],
        ridge_alphas=[1e-4, 1e-2],
        cv_folds=3,
    )
    y_pred = model.predict(X[ds.train_end :])
    mc = memory_capacity(y_pred, ds.targets[ds.train_end :])

    return {
        'u': ds.u,
        'X': X,
        'y_pred': y_pred,
        'mc': mc,
        'circuit_hash': compute_circuit_hash(params),
    }


class TestReproducibility:
    """Tests that same seeds produce identical results."""

    def test_task_deterministic(self) -> None:
        r1 = _run_stm_pipeline(42, 137)
        r2 = _run_stm_pipeline(42, 137)
        assert np.array_equal(r1['u'], r2['u'])

    def test_features_deterministic(self) -> None:
        r1 = _run_stm_pipeline(42, 137)
        r2 = _run_stm_pipeline(42, 137)
        assert np.allclose(r1['X'], r2['X'])

    def test_predictions_deterministic(self) -> None:
        r1 = _run_stm_pipeline(42, 137)
        r2 = _run_stm_pipeline(42, 137)
        assert np.allclose(r1['y_pred'], r2['y_pred'])

    def test_mc_deterministic(self) -> None:
        r1 = _run_stm_pipeline(42, 137)
        r2 = _run_stm_pipeline(42, 137)
        assert r1['mc'] == r2['mc']

    def test_circuit_hash_deterministic(self) -> None:
        r1 = _run_stm_pipeline(42, 137)
        r2 = _run_stm_pipeline(42, 137)
        assert r1['circuit_hash'] == r2['circuit_hash']

    def test_different_seeds_differ(self) -> None:
        r1 = _run_stm_pipeline(42, 137)
        r2 = _run_stm_pipeline(99, 200)
        assert not np.array_equal(r1['u'], r2['u'])
        assert r1['circuit_hash'] != r2['circuit_hash']

    def test_mc_finite(self) -> None:
        r = _run_stm_pipeline(42, 137)
        assert np.isfinite(r['mc'])

    def test_mc_nonnegative(self) -> None:
        r = _run_stm_pipeline(42, 137)
        assert r['mc'] >= 0.0
