"""Batched ridge readout for permutation nulls.

``fit_ridge_cv_batched`` reproduces ``sklearn.linear_model.RidgeCV(alphas=..., cv=k)``,
the harness readout, for many target sets that share one feature matrix. It follows the
same procedure: a grid search over ``alphas`` with ``k`` contiguous (unshuffled) KFold
splits, R^2 averaged uniformly over the targets of a set, the first best alpha on ties, an
unpenalised intercept, and a Cholesky solve, followed by a refit on all rows.

Permuting targets leaves X^T X unchanged, so each fold's (X^T X + alpha I) is built once
and every target set is solved in a single call. G0.7 uses this to refit its readout on
200 permutations at the cost of a few solves (docs/DECISIONS.md D005, D008).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import linalg
from sklearn.model_selection import KFold


@dataclass(frozen=True)
class BatchedRidgeCV:
    """Ridge readouts fitted to many target sets that share one feature matrix.

    Attributes:
        alphas: Candidate penalties, shape (n_alphas,).
        alpha_: Selected penalty per target set, shape (n_sets,).
        alpha_index_: Index of the selected penalty in ``alphas``, shape (n_sets,).
        cv_score_: Mean cross-validated R^2 per penalty and set, shape (n_alphas, n_sets).
        coef_: Weights, shape (n_features, n_targets, n_sets).
        intercept_: Intercepts, shape (n_targets, n_sets).
    """

    alphas: np.ndarray
    alpha_: np.ndarray
    alpha_index_: np.ndarray
    cv_score_: np.ndarray
    coef_: np.ndarray
    intercept_: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict every target of every set.

        Args:
            X: Features, shape (n_samples, n_features).

        Returns:
            Predictions, shape (n_samples, n_targets, n_sets).
        """
        X = np.asarray(X, dtype=np.float64)
        n_features, n_targets, n_sets = self.coef_.shape
        flat = X @ self.coef_.reshape(n_features, n_targets * n_sets)
        return flat.reshape(len(X), n_targets, n_sets) + self.intercept_


def fit_ridge_cv_batched(
    X: np.ndarray,
    Y: np.ndarray,
    alphas: Sequence[float],
    cv_folds: int,
) -> BatchedRidgeCV:
    """Fit RidgeCV's grid-search readout to every target set at once.

    Args:
        X: Features, shape (n_samples, n_features), shared by every set.
        Y: Targets, shape (n_samples, n_targets, n_sets). Set j is Y[:, :, j].
        alphas: Candidate ridge penalties, all > 0 (the harness's ridge_alphas).
        cv_folds: Number of contiguous KFold splits.

    Returns:
        BatchedRidgeCV whose set j matches RidgeCV(alphas, cv=cv_folds).fit(X, Y[:, :, j]).

    Raises:
        ValueError: If shapes or alphas are invalid.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    alpha_grid = np.asarray(alphas, dtype=np.float64).ravel()
    if X.ndim != 2:
        raise ValueError(f'X must have shape (n_samples, n_features); got {X.shape}')
    if Y.ndim != 3 or Y.shape[0] != X.shape[0]:
        raise ValueError(
            f'Y must have shape (n_samples, n_targets, n_sets) with n_samples={X.shape[0]}; '
            f'got {Y.shape}'
        )
    if alpha_grid.size == 0 or np.any(~np.isfinite(alpha_grid)) or np.any(alpha_grid <= 0):
        raise ValueError(f'alphas must be finite and > 0; got {alpha_grid.tolist()}')

    n_samples, n_targets, n_sets = Y.shape
    Y_flat = Y.reshape(n_samples, n_targets * n_sets)
    folds = list(KFold(n_splits=cv_folds).split(X))

    # scores[a, f, j]: R^2 of penalty a on held-out fold f for set j (RidgeCV's grid).
    scores = np.empty((alpha_grid.size, len(folds), n_sets), dtype=np.float64)
    for f, (train_idx, val_idx) in enumerate(folds):
        solutions = _ridge_solutions(X[train_idx], Y_flat[train_idx], alpha_grid)
        for a, (coef, intercept) in enumerate(solutions):
            pred = X[val_idx] @ coef + intercept
            scores[a, f] = _r2_uniform_average(
                Y_flat[val_idx].reshape(len(val_idx), n_targets, n_sets),
                pred.reshape(len(val_idx), n_targets, n_sets),
            )
    cv_score = np.average(scores, axis=1)
    best = _first_best(cv_score)

    solutions = _ridge_solutions(X, Y_flat, alpha_grid)
    coefs = np.stack([c.reshape(-1, n_targets, n_sets) for c, _ in solutions])
    intercepts = np.stack([b.reshape(n_targets, n_sets) for _, b in solutions])
    sets = np.arange(n_sets)
    return BatchedRidgeCV(
        alphas=alpha_grid,
        alpha_=alpha_grid[best],
        alpha_index_=best,
        cv_score_=cv_score,
        coef_=coefs[best, :, :, sets].transpose(1, 2, 0),
        intercept_=intercepts[best, :, sets].T,
    )


def _ridge_solutions(X: np.ndarray, Y: np.ndarray, alphas: np.ndarray) -> list:
    """Solve centred ridge for every penalty, as sklearn's Ridge(solver='cholesky') does.

    Returns:
        One (coef, intercept) pair per penalty; coef has shape (n_features, n_columns)
        and intercept has shape (n_columns,).
    """
    x_offset = np.average(X, axis=0)
    y_offset = np.average(Y, axis=0)
    Xc = X - x_offset
    Yc = Y - y_offset
    gram = Xc.T @ Xc
    xy = Xc.T @ Yc
    n_features = X.shape[1]
    solutions = []
    for alpha in alphas:
        A = gram.copy()
        A.flat[:: n_features + 1] += alpha
        coef = linalg.solve(A, xy, assume_a='pos', overwrite_a=True)
        solutions.append((coef, y_offset - x_offset @ coef))
    return solutions


def _r2_uniform_average(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """sklearn's r2_score(multioutput='uniform_average', force_finite=True), per set.

    Args:
        y_true: Shape (n_samples, n_targets, n_sets).
        y_pred: Shape (n_samples, n_targets, n_sets).

    Returns:
        Mean R^2 over targets for each set, shape (n_sets,).
    """
    numerator = ((y_true - y_pred) ** 2).sum(axis=0)
    denominator = ((y_true - np.average(y_true, axis=0)) ** 2).sum(axis=0)
    nonzero_num = numerator != 0
    nonzero_den = denominator != 0
    valid = nonzero_num & nonzero_den
    scores = np.ones_like(numerator)
    scores[valid] = 1.0 - numerator[valid] / denominator[valid]
    scores[nonzero_num & ~nonzero_den] = 0.0
    # Average each set's targets as one contiguous row, like sklearn's np.average.
    return np.ascontiguousarray(scores.T).mean(axis=1)


def _first_best(cv_score: np.ndarray) -> np.ndarray:
    """Index of the first highest mean score per set; NaN scores rank last.

    This is GridSearchCV's rule: rank the means (ties share the best rank) and take the
    first candidate with the best rank. If every mean is NaN, the first alpha is taken.
    """
    finite = np.where(np.isnan(cv_score), -np.inf, cv_score)
    return np.argmax(finite, axis=0)
