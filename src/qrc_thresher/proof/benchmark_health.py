"""Functional benchmark health checks.

Tests config loading, task determinism, QRC smoke circuit,
ESN baseline, metric computations, manifest writer, and reproducibility.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path('configs') / 'alpha_lite.yaml'


def check_config() -> Dict[str, Any]:
    """Check that alpha_lite.yaml validates against config schema."""
    try:
        from qrc_thresher.config import load_config

        cfg = load_config(_CONFIG_PATH)
        return {'status': 'PASS', 'experiment_name': cfg.experiment_name}
    except Exception as exc:
        return {'status': f'FAIL: {exc}'}


def check_tasks() -> Dict[str, Any]:
    """Check task determinism for STM and parity."""
    try:
        from qrc_thresher.tasks.stm import generate_stm
        from qrc_thresher.tasks.temporal_parity import generate_parity

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        ds1 = generate_stm(length=100, delay_max=5, train_frac=0.7, rng=rng1)
        ds2 = generate_stm(length=100, delay_max=5, train_frac=0.7, rng=rng2)
        stm_ok = np.array_equal(ds1.u, ds2.u)

        rng3 = np.random.default_rng(99)
        rng4 = np.random.default_rng(99)
        p1 = generate_parity(length=100, window=3, train_frac=0.7, rng=rng3)
        p2 = generate_parity(length=100, window=3, train_frac=0.7, rng=rng4)
        parity_ok = np.array_equal(p1.u, p2.u)

        return {
            'status': 'PASS' if (stm_ok and parity_ok) else 'FAIL',
            'stm_deterministic': stm_ok,
            'parity_deterministic': parity_ok,
        }
    except Exception as exc:
        return {'status': f'FAIL: {exc}'}


def check_qrc_smoke() -> Dict[str, Any]:
    """Check 2-qubit QRC smoke circuit on default.qubit."""
    try:
        from qrc_thresher.reservoirs.pennylane_qrc import build_reservoir_params, extract_features

        rng = np.random.default_rng(137)
        params = build_reservoir_params(
            n_qubits=2, depth=1, readout='z_only', backend='default.qubit', rng=rng
        )
        u = np.array([0.5])
        feats = extract_features(u, params)
        finite_ok = np.isfinite(feats).all()
        return {
            'status': 'PASS' if finite_ok else 'FAIL',
            'shape': list(feats.shape),
            'finite': bool(finite_ok),
        }
    except Exception as exc:
        return {'status': f'FAIL: {exc}'}


def check_esn_smoke() -> Dict[str, Any]:
    """Check ESN baseline trains and predicts on a small STM dataset."""
    try:
        from qrc_thresher.baselines.esn import grid_search_esn, train_predict_esn
        from qrc_thresher.tasks.stm import generate_stm

        rng = np.random.default_rng(42)
        ds = generate_stm(length=100, delay_max=3, train_frac=0.7, rng=rng)
        u_train = ds.u[: ds.train_end]
        y_train = ds.targets[: ds.train_end, 1]  # delay=1 target

        grid = {
            'spectral_radius': [0.9],
            'input_scaling': [0.5],
            'leak_rate': [0.3],
        }
        result = grid_search_esn(
            u_train,
            y_train,
            grid=grid,
            n_features=4,
            cv_folds=2,
            ridge_alphas=[1e-4, 1e-2],
            rng=np.random.default_rng(0),
        )
        preds = train_predict_esn(
            u_train,
            y_train,
            ds.u[ds.train_end :],
            result,
            n_features=4,
            rng=np.random.default_rng(0),
        )
        return {
            'status': 'PASS',
            'pred_shape': list(preds.shape),
        }
    except Exception as exc:
        return {'status': f'FAIL: {exc}'}


def check_metrics() -> Dict[str, Any]:
    """Check metric computations return finite values on dummy data."""
    try:
        from qrc_thresher.metrics.scoring import classification_accuracy, memory_capacity, nrmse

        rng = np.random.default_rng(0)
        y_pred = rng.normal(0, 1, size=(50, 5))
        y_true = rng.normal(0, 1, size=(50, 5))
        mc = memory_capacity(y_pred, y_true)

        yp = rng.normal(0, 1, size=50)
        yt = rng.normal(0, 1, size=50)
        nr = nrmse(yp, yt)

        yp_bin = rng.uniform(0, 1, size=50)
        yt_bin = rng.integers(0, 2, size=50)
        acc = classification_accuracy(yp_bin, yt_bin)

        all_finite = all(np.isfinite([mc, nr, acc]))
        return {
            'status': 'PASS' if all_finite else 'FAIL',
            'mc': mc,
            'nrmse': nr,
            'accuracy': acc,
        }
    except Exception as exc:
        return {'status': f'FAIL: {exc}'}


def check_manifest() -> Dict[str, Any]:
    """Check run manifest writer produces a valid record."""
    try:

        from qrc_thresher.proof.run_manifest import append_to_csv, create_manifest

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / 'runs.csv'
            manifest = create_manifest(
                config_path=_CONFIG_PATH,
                circuit_hash='abc123',
                task_seed=42,
                reservoir_seed=137,
                backend_device='default.qubit',
                runtime_per_stage_seconds={'total': 0.1},
                entanglement_metric=None,
                success=True,
                failure_reason=None,
                artifact_paths=[],
            )
            append_to_csv(manifest, csv_path=csv_path)
            assert csv_path.exists()
            return {'status': 'PASS', 'run_id': manifest.run_id}
    except Exception as exc:
        return {'status': f'FAIL: {exc}'}


def check_reproducibility() -> Dict[str, Any]:
    """Check same seed produces identical task sequence and metric values."""
    try:
        from qrc_thresher.metrics.scoring import memory_capacity
        from qrc_thresher.tasks.stm import generate_stm

        def run_once(seed: int) -> float:
            rng = np.random.default_rng(seed)
            ds = generate_stm(length=100, delay_max=3, train_frac=0.7, rng=rng)
            # Trivial "predictor": predict mean
            y_pred = np.tile(
                np.mean(ds.targets[: ds.train_end], axis=0), (100 - ds.train_end, 1)
            )
            y_true = ds.targets[ds.train_end :]
            return memory_capacity(y_pred, y_true)

        mc1 = run_once(42)
        mc2 = run_once(42)
        ok = mc1 == mc2
        return {
            'status': 'PASS' if ok else 'FAIL',
            'mc_run1': mc1,
            'mc_run2': mc2,
            'identical': ok,
        }
    except Exception as exc:
        return {'status': f'FAIL: {exc}'}


def run_benchmark_health() -> Dict[str, Any]:
    """Run all benchmark health checks.

    Returns:
        Health report dict with per-check status and overall status.
    """
    checks = {
        'config': check_config(),
        'tasks': check_tasks(),
        'qrc_smoke': check_qrc_smoke(),
        'esn_smoke': check_esn_smoke(),
        'metrics': check_metrics(),
        'manifest': check_manifest(),
        'reproducibility': check_reproducibility(),
    }

    def _status(v: Any) -> str:
        if isinstance(v, dict):
            return v.get('status', 'FAIL')
        return str(v)

    overall = all(_status(v) == 'PASS' for v in checks.values())
    return {
        'checks': {k: _status(v) for k, v in checks.items()},
        'details': checks,
        'overall': 'PASS' if overall else 'FAIL',
    }
