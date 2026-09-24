"""Scoring metrics: STM memory (k >= 1), Memory Capacity (MC), NRMSE, classification accuracy.

All functions validate inputs and raise on non-finite values.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# A prediction or target whose standard deviation is at or below this is treated as
# constant: its correlation is undefined, so it is refused rather than scored as 0.
# Same value as the degeneracy tolerance registered for G0.7 v1 (docs/DECISIONS.md D009).
DEGENERATE_STD = 1e-12


class DegeneratePredictionError(ValueError):
    """A correlation-based metric was asked to score a constant series."""


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
        DegeneratePredictionError: If any prediction or target column is constant
            (std <= DEGENERATE_STD). A constant prediction is never scored as 0.
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


def stm_memory(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """The STM memory sum over delays k >= 1 (docs/DECISIONS.md D011, D013; defect D10).

    Column k of ``y_true`` is u_{t-k}; column 0 (k = 0, the present input) is never memory and
    is excluded. This is the one scoring function every model, tuner and ablation uses for STM
    (PI ruling 10, CP4b); ``memory_capacity`` over all columns gives ``mc_total``.

    Args:
        y_pred: Predicted targets of shape (T, K+1).
        y_true: True targets of shape (T, K+1).

    Returns:
        sum_{k=1..K} corr(y_hat^{(k)}, y^{(k)})^2.

    Raises:
        ValueError: If inputs contain non-finite values or have fewer than two columns.
        DegeneratePredictionError: If any prediction or target column k >= 1 is constant.
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    if y_pred.ndim != 2 or y_true.ndim != 2 or y_true.shape[1] < 2:
        raise ValueError(
            f'stm_memory needs (T, K+1) targets with K >= 1; got {y_pred.shape} and {y_true.shape}'
        )
    return memory_capacity(y_pred[:, 1:], y_true[:, 1:])


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
    """Compute the Pearson correlation coefficient, refusing constant arrays.

    A constant prediction carries no information, and its correlation is undefined.
    Scoring it as 0 would hide a dead model, so it raises instead (defect D9).

    Args:
        a: Array 1.
        b: Array 2.

    Returns:
        Pearson r in [-1, 1].

    Raises:
        DegeneratePredictionError: If either array has std <= DEGENERATE_STD.
    """
    std_a, std_b = float(np.std(a)), float(np.std(b))
    if std_a <= DEGENERATE_STD or std_b <= DEGENERATE_STD:
        raise DegeneratePredictionError(
            f'constant input to a correlation (std {std_a:.3g} and {std_b:.3g}, '
            f'tolerance {DEGENERATE_STD:g}): the score is undefined, not 0'
        )
    corr_matrix = np.corrcoef(a, b)
    return float(corr_matrix[0, 1])
