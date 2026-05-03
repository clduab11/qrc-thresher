"""NARMA-10 task generator.

Phase 1.5 ONLY — gated behind G3 pass.
Input: u_t ~ Uniform(0, 0.5).
Recurrence: y_{t+1} = 0.3*y_t + 0.05*y_t*(sum last 10 y) + 1.5*u_{t-9}*u_t + 0.1
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

logger = logging.getLogger(__name__)

_NARMA_A = 0.3
_NARMA_B = 0.05
_NARMA_C = 1.5
_NARMA_D = 0.1
_NARMA_ORDER = 10


@dataclass(frozen=True, slots=True)
class NARMA10Dataset:
    """Container for NARMA-10 task data."""

    u: np.ndarray  # shape (T,)
    targets: np.ndarray  # shape (T,)
    train_end: int


def verify_narma10_recurrence(u: np.ndarray, y: np.ndarray) -> bool:
    """Verify that y satisfies the NARMA-10 recurrence relation.

    Args:
        u: Input array of shape (T,).
        y: Output array of shape (T + 1,).

    Returns:
        True if recurrence relation holds for all valid indices.
    """
    for t in range(_NARMA_ORDER, len(u)):
        y_sum = np.sum(y[t - _NARMA_ORDER + 1 : t + 1])
        expected = (
            _NARMA_A * y[t]
            + _NARMA_B * y[t] * y_sum
            + _NARMA_C * u[t - _NARMA_ORDER + 1] * u[t]
            + _NARMA_D
        )
        if not np.isclose(y[t + 1], expected, rtol=1e-10, atol=1e-12):
            return False
    return True


def generate_narma10(
    length: int,
    train_frac: float,
    rng: Generator,
) -> NARMA10Dataset:
    """Generate NARMA-10 task data (Phase 1.5, gated behind G3).

    Args:
        length: Total sequence length T.
        train_frac: Fraction of data for training (chronological split).
        rng: Seeded random generator.

    Returns:
        NARMA10Dataset with u in [0, 0.5] and targets.

    Raises:
        AssertionError: If generated input falls outside [0, 0.5].
    """
    u = rng.uniform(0.0, 0.5, size=length)
    if not (0 <= u.min() and u.max() <= 0.5):
        raise ValueError(f'NARMA-10 input out of range: [{u.min()}, {u.max()}]')
    y = np.zeros(length + 1, dtype=np.float64)
    for t in range(_NARMA_ORDER, length):
        y_sum = np.sum(y[t - _NARMA_ORDER + 1 : t + 1])
        y[t + 1] = (
            _NARMA_A * y[t]
            + _NARMA_B * y[t] * y_sum
            + _NARMA_C * u[t - _NARMA_ORDER + 1] * u[t]
            + _NARMA_D
        )
    if not verify_narma10_recurrence(u, y):
        raise ValueError('NARMA-10 recurrence relation verification failed')
    targets = y[1:]
    train_end = int(length * train_frac)
    logger.debug('Generated NARMA-10: length=%d, train_end=%d', length, train_end)
    return NARMA10Dataset(u=u, targets=targets, train_end=train_end)
