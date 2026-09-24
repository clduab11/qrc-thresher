"""Functional benchmark health checks.

Tests config loading, task determinism, QRC smoke circuit,
ESN baseline, metric computations, manifest writer, and reproducibility.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path('configs') / 'alpha_lite.yaml'
# The ESN smoke check passes only if held-out memory over delays k = 1..4 reaches this
# (docs/DECISIONS.md D009). A memoryless or disconnected model cannot.
_ESN_SMOKE_MIN_MEMORY = 1.0


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


def check_esn_smoke(params: Optional[Any] = None) -> Dict[str, Any]:
    """Check the ESN baseline learns, not just that it returns an array.

    A 4-unit ESN (the linear preset unless ``params`` is given) is fitted with the harness
    readout on a small STM task (T = 300, K = 4) after the config's washout
    (``training.washout``, D012). PASS requires a held-out ``stm_memory`` (delays k = 1..4) of
    at least _ESN_SMOKE_MIN_MEMORY. A disconnected ESN fails: its predictions are constant and
    cannot be scored.
    """
    try:
        from qrc_thresher.baselines.esn import ESN_PRESETS, draw_reservoir, fit_predict_esn
        from qrc_thresher.config import load_config
        from qrc_thresher.metrics.scoring import stm_memory
        from qrc_thresher.tasks.stm import generate_stm

        params = params or ESN_PRESETS['esn_linear']
        cfg = load_config(_CONFIG_PATH)
        washout = int(cfg.training.washout)
        ds = generate_stm(length=300, delay_max=4, train_frac=0.7, rng=np.random.default_rng(42))
        pred, esn, _ = fit_predict_esn(
            ds.u, ds.targets, ds.train_end, draw_reservoir(4, 137), params, washout,
            cfg.training.ridge_alphas, cfg.training.cv_folds,
        )
        memory = float(stm_memory(pred, ds.targets[ds.train_end:]))
        learned = memory >= _ESN_SMOKE_MIN_MEMORY
        return {
            'status': 'PASS' if learned else 'FAIL',
            'memory_k1_to_4': memory,
            'threshold': _ESN_SMOKE_MIN_MEMORY,
            'washout': washout,
            'weight_hash': esn.weight_hash(),
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
            # Deterministic, non-constant "predictor": the targets plus seeded noise.
            # (A constant predictor can no longer be scored: defect D9.)
            y_true = ds.targets[ds.train_end :]
            noise = np.random.default_rng(seed + 1).normal(0.0, 0.5, size=y_true.shape)
            return memory_capacity(y_true + noise, y_true)

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
