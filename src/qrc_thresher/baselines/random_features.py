"""Random Kitchen Sinks / Random Nonlinear Features baseline.

Feature: phi(u) = cos(W*u + b) where W ~ N(0, sigma^2/d), b ~ Uniform(0, 2*pi).
Dimension matched to QRC features.
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

    W: np.ndarray  # shape (n_features,)
    b: np.ndarray  # shape (n_features,)
    sigma: float
    n_features: int


def build_rks_params(
    n_features: int,
    sigma: float,
    rng: Generator,
) -> RKSParams:
    """Build random projection parameters for RKS baseline.

    Args:
        n_features: Number of random features (must equal N_quantum_features).
        sigma: Bandwidth parameter.
        rng: Seeded generator.

    Returns:
        RKSParams with W and b.
    """
    W = rng.normal(0.0, sigma / max(n_features, 1), size=n_features)
    b = rng.uniform(0.0, 2.0 * np.pi, size=n_features)
    return RKSParams(W=W, b=b, sigma=sigma, n_features=n_features)


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
    # phi(u_t) = cos(W * u_t + b) — vectorized over time
    X = np.cos(np.outer(u, params.W) + params.b)  # shape (T, n_features)
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
) -> np.ndarray:
    """Train RKS baseline and predict on test set.

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
    params = build_rks_params(n_features=n_features, sigma=sigma, rng=rng)
    X_train = extract_rks_features(u_train, params)
    X_test = extract_rks_features(u_test, params)

    model = RidgeCV(alphas=ridge_alphas, cv=cv_folds)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    logger.debug('RKS fitted: best_alpha=%s', getattr(model, 'alpha_', None))
    return preds
