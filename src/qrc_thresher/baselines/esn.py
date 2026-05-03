"""Echo State Network (ESN) baseline using ReservoirPy.

Feature dimension is matched to QRC: N_ESN = N_quantum_features.
NEVER use 2^N as the ESN dimension — see docstring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional

import numpy as np
from numpy.random import Generator

logger = logging.getLogger(__name__)

# Feature-matching formula: N_ESN = N_quantum_features
# For z_only: N_quantum_features = n_qubits
# For z_and_zz: N_quantum_features = n_qubits + n_qubits*(n_qubits-1)/2
# Do NOT set N_ESN = 2^n_qubits. That would be an unfair comparison.


@dataclass(frozen=True, slots=True)
class ESNResult:
    """Best ESN hyperparameters and fitted model."""

    spectral_radius: float
    input_scaling: float
    leak_rate: float
    ridge_alpha: float
    val_score: float


def _n_features(n_qubits: int, readout: str) -> int:
    """Compute number of QRC features for dimension matching.

    Args:
        n_qubits: Number of qubits.
        readout: 'z_only' or 'z_and_zz'.

    Returns:
        Number of features (N_ESN matched to N_quantum_features).
    """
    if readout == 'z_only':
        return n_qubits
    return n_qubits + n_qubits * (n_qubits - 1) // 2


def grid_search_esn(
    u_train: np.ndarray,
    y_train: np.ndarray,
    grid: Dict[str, List[float]],
    n_features: int,
    cv_folds: int,
    ridge_alphas: List[float],
    rng: Generator,
) -> ESNResult:
    """Grid search over ESN hyperparameters on training data only.

    Args:
        u_train: Training input of shape (T_train,).
        y_train: Training targets of shape (T_train,) or (T_train, K).
        grid: Hyperparameter grid with keys spectral_radius, input_scaling, leak_rate.
        n_features: ESN reservoir size (must equal N_quantum_features).
        cv_folds: Number of CV folds (train-only; never use test indices).
        ridge_alphas: Ridge regression alpha candidates.
        rng: Seeded generator.

    Returns:
        ESNResult with best hyperparameters.
    """
    import reservoirpy as rpy
    from reservoirpy.nodes import Reservoir, Ridge

    rpy.set_seed(int(rng.integers(0, 2**31)))
    rpy.verbosity(0)

    spectral_radii = grid.get('spectral_radius', [0.9])
    input_scalings = grid.get('input_scaling', [0.5])
    leak_rates = grid.get('leak_rate', [0.3])

    best_score = float('inf')
    best_result: Optional[ESNResult] = None

    u_col = u_train.reshape(-1, 1)
    if y_train.ndim == 1:
        y_col = y_train.reshape(-1, 1)
    else:
        y_col = y_train

    fold_size = len(u_train) // cv_folds

    for sr, isc, lr in product(spectral_radii, input_scalings, leak_rates):
        fold_scores = []
        for fold in range(cv_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size
            u_val_fold = u_col[val_start:val_end]
            y_val_fold = y_col[val_start:val_end]
            u_tr_fold = np.concatenate([u_col[:val_start], u_col[val_end:]], axis=0)
            y_tr_fold = np.concatenate([y_col[:val_start], y_col[val_end:]], axis=0)

            try:
                reservoir = Reservoir(
                    units=n_features,
                    sr=sr,
                    input_scaling=isc,
                    lr=lr,
                    seed=int(rng.integers(0, 2**31)),
                )
                readout = Ridge(ridge=min(ridge_alphas))
                model = reservoir >> readout
                model.fit(u_tr_fold, y_tr_fold, warmup=10)
                preds = model.run(u_val_fold)
                score = float(np.mean((preds - y_val_fold) ** 2))
                fold_scores.append(score)
            except Exception as exc:
                logger.warning(
                    'ESN fold failed: sr=%.2f, isc=%.2f, lr=%.2f: %s', sr, isc, lr, exc
                )
                fold_scores.append(float('inf'))

        mean_score = float(np.mean(fold_scores))
        if mean_score < best_score:
            best_score = mean_score
            best_result = ESNResult(
                spectral_radius=sr,
                input_scaling=isc,
                leak_rate=lr,
                ridge_alpha=min(ridge_alphas),
                val_score=mean_score,
            )

    if best_result is None:
        raise RuntimeError('ESN grid search failed: no valid hyperparameter combination found')

    logger.info(
        'Best ESN: sr=%.2f, isc=%.2f, lr=%.2f, val_score=%.4f',
        best_result.spectral_radius,
        best_result.input_scaling,
        best_result.leak_rate,
        best_result.val_score,
    )
    return best_result


def train_predict_esn(
    u_train: np.ndarray,
    y_train: np.ndarray,
    u_test: np.ndarray,
    result: ESNResult,
    n_features: int,
    rng: Generator,
) -> np.ndarray:
    """Train ESN with best hyperparameters and predict on test set.

    Args:
        u_train: Training input of shape (T_train,).
        y_train: Training targets.
        u_test: Test input of shape (T_test,).
        result: Best ESN hyperparameters from grid search.
        n_features: Reservoir size.
        rng: Seeded generator.

    Returns:
        Predictions of shape (T_test,) or (T_test, K).
    """
    import reservoirpy as rpy
    from reservoirpy.nodes import Reservoir, Ridge

    rpy.set_seed(int(rng.integers(0, 2**31)))
    rpy.verbosity(0)

    reservoir = Reservoir(
        units=n_features,
        sr=result.spectral_radius,
        input_scaling=result.input_scaling,
        lr=result.leak_rate,
        seed=int(rng.integers(0, 2**31)),
    )
    readout = Ridge(ridge=result.ridge_alpha)
    model = reservoir >> readout

    u_col = u_train.reshape(-1, 1)
    y_col = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
    model.fit(u_col, y_col, warmup=10)

    preds = model.run(u_test.reshape(-1, 1))
    if preds.shape[1] == 1:
        return preds.ravel()
    return preds
