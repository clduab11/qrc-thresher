"""Scoring metrics: Memory Capacity (MC), NRMSE, classification accuracy.

All functions validate inputs and raise on non-finite values.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def memory_capacity(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Compute Memory Capacity (MC) for STM task.

    MC = sum_k corr(y_hat^{(k)}, y^{(k)})^2

    Args:
        y_pred: Predicted targets of shape (T, K+1).
        y_true: True targets of shape (T, K+1).

    Returns:
        Memory capacity scalar MC >= 0.

    Raises:
        ValueError: If inputs contain non-finite values.
    """
    _check_finite(y_pred, 'y_pred')
    _check_finite(y_true, 'y_true')
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    mc = 0.0
    for k in range(y_true.shape[1]):
        corr = _safe_corrcoef(y_pred[:, k], y_true[:, k])
        mc += corr**2
    logger.debug('MC computed: %.4f (over %d delays)', mc, y_true.shape[1])
    return float(mc)


def nrmse(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Compute Normalized Root Mean Square Error (NRMSE).

    NRMSE = sqrt(MSE) / std(y_true)

    Args:
        y_pred: Predictions of shape (T,).
        y_true: True values of shape (T,).

    Returns:
        NRMSE scalar >= 0.

    Raises:
        ValueError: If inputs contain non-finite values or y_true has zero variance.
    """
    _check_finite(y_pred, 'y_pred')
    _check_finite(y_true, 'y_true')
    y_std = float(np.std(y_true))
    if y_std == 0.0:
        raise ValueError('NRMSE undefined: y_true has zero variance')
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    result = rmse / y_std
    logger.debug('NRMSE: %.4f', result)
    return result


def classification_accuracy(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Compute classification accuracy for parity task.

    Args:
        y_pred: Predicted class probabilities or logits of shape (T,).
        y_true: True class labels of shape (T,) in {0, 1}.

    Returns:
        Accuracy in [0, 1].

    Raises:
        ValueError: If inputs contain non-finite values.
    """
    _check_finite(y_pred, 'y_pred')
    predicted_labels = (y_pred >= 0.5).astype(np.int32)
    acc = float(np.mean(predicted_labels == y_true))
    logger.debug('Accuracy: %.4f', acc)
    return acc


def _check_finite(arr: np.ndarray, name: str) -> None:
    """Check that array contains only finite values.

    Args:
        arr: Array to check.
        name: Variable name for error message.

    Raises:
        ValueError: If any non-finite value is found.
    """
    if not np.isfinite(arr).all():
        raise ValueError(f'{name} contains non-finite values (NaN or inf)')


def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Pearson correlation coefficient, returning 0 for constant arrays.

    Args:
        a: Array 1.
        b: Array 2.

    Returns:
        Pearson r in [-1, 1], or 0.0 if either array is constant.
    """
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    corr_matrix = np.corrcoef(a, b)
    return float(corr_matrix[0, 1])
