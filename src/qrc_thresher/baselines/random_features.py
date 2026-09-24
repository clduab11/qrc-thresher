"""Random Kitchen Sinks / Random Nonlinear Features baseline (docs/DECISIONS.md D011; D13).

Feature: phi(x_t) = cos(W x_t + b) on the zero-padded input window
x_t = (u_t, u_{t-1}, ..., u_{t-d+1}) of dimension d, with W of shape (F, d), W_ij ~ N(0,
(sigma / sqrt(d))^2) drawn row-major from the stream, then b ~ Uniform(0, 2 pi) of shape (F,).
At d = 1 the stream positions of the D010 draw are unchanged and only the scale of W changes
(sigma / F became sigma). F is matched to the QRC feature count.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
from numpy.random import Generator
from sklearn.linear_model import RidgeCV

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RKSParams:
    """Random Kitchen Sinks parameters."""

    W: np.ndarray  # shape (n_features, window)
    b: np.ndarray  # shape (n_features,)
    sigma: float
    n_features: int
    window: int = 1


def build_rks_params(
    n_features: int,
    sigma: float,
    rng: Generator,
    window: int = 1,
) -> RKSParams:
    """Build random projection parameters for the RKS baseline (D011).

    Args:
        n_features: Number of random features F (must equal N_quantum_features).
        sigma: Bandwidth parameter; W_ij ~ N(0, (sigma / sqrt(window))^2).
        rng: Seeded generator (default_rng([reservoir_seed, 3]) in the harness).
        window: Input window dimension d >= 1.

    Returns:
        RKSParams with W of shape (F, d) and b of shape (F,).
    """
    if window < 1:
        raise ValueError(f'window must be >= 1; got {window}')
    W = rng.normal(0.0, sigma / np.sqrt(window), size=(n_features, window))
    b = rng.uniform(0.0, 2.0 * np.pi, size=n_features)
    return RKSParams(W=W, b=b, sigma=float(sigma), n_features=int(n_features), window=int(window))


def window_inputs(u: np.ndarray, window: int) -> np.ndarray:
    """Zero-padded input window: row t, column j is u_{t-j}, or 0 where t - j < 0."""
    u = np.asarray(u, dtype=np.float64).ravel()
    T = len(u)
    out = np.zeros((T, window), dtype=np.float64)
    for j in range(window):
        out[j:, j] = u[: T - j]
    return out


def extract_rks_features(
    u: np.ndarray,
    params: RKSParams,
) -> np.ndarray:
    """Extract RKS feature matrix from input sequence.

    Args:
        u: Input sequence of shape (T,).
        params: RKS parameters.

    Returns:
        Feature matrix of shape (T, n_features).

    Raises:
        ValueError: If features contain non-finite values.
    """
    # phi(x_t) = cos(W x_t + b) on the zero-padded window, vectorised over time.
    W = np.asarray(params.W, dtype=np.float64).reshape(params.n_features, -1)
    X = np.cos(window_inputs(u, W.shape[1]) @ W.T + params.b)  # shape (T, n_features)
    if not np.isfinite(X).all():
        raise ValueError('RKS features contain non-finite values')
    return X


def train_rks(
    u_train: np.ndarray,
    y_train: np.ndarray,
    u_test: np.ndarray,
    n_features: int,
    ridge_alphas: List[float],
    cv_folds: int,
    rng: Generator,
    sigma: float = 1.0,
    window: int = 1,
) -> np.ndarray:
    """Train RKS baseline and predict on test set (plugin entry point).

    Args:
        u_train: Training input of shape (T_train,).
        y_train: Training targets.
        u_test: Test input of shape (T_test,).
        n_features: Number of random features.
        ridge_alphas: Ridge regression alphas.
        cv_folds: Number of CV folds.
        rng: Seeded generator.
        sigma: Bandwidth parameter.

    Returns:
        Predictions of shape (T_test,) or (T_test, K).
    """
    params = build_rks_params(n_features=n_features, sigma=sigma, rng=rng, window=window)
    X_train = extract_rks_features(u_train, params)
    X_test = extract_rks_features(u_test, params)

    model = RidgeCV(alphas=ridge_alphas, cv=cv_folds)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    logger.debug('RKS fitted: best_alpha=%s', getattr(model, 'alpha_', None))
    return preds
