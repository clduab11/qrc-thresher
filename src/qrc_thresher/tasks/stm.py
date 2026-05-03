"""Short-Term Memory (STM) task generator.

Input: u_t ~ Uniform(-1, 1), length T.
Targets: y_t^{(k)} = u_{t-k} for k in [0, K], K typically 20.
Train/test split: chronological (no shuffling).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class STMDataset:
    """Container for STM task data."""

    u: np.ndarray  # shape (T,)
    targets: np.ndarray  # shape (T, K+1)
    delay_max: int
    train_end: int  # exclusive index


def generate_stm(
    length: int,
    delay_max: int,
    train_frac: float,
    rng: Generator,
) -> STMDataset:
    """Generate Short-Term Memory task data.

    Args:
        length: Total sequence length T.
        delay_max: Maximum delay K. Targets are y_t^{(k)} = u_{t-k} for k in [0, K].
        train_frac: Fraction of data for training (chronological split).
        rng: Seeded random generator.

    Returns:
        STMDataset with u of shape (T,) and targets of shape (T, K+1).
    """
    u = rng.uniform(-1.0, 1.0, size=length)
    K = delay_max
    targets = np.zeros((length, K + 1), dtype=np.float64)
    for k in range(K + 1):
        if k == 0:
            targets[:, k] = u
        else:
            targets[k:, k] = u[:-k]
    train_end = int(length * train_frac)
    logger.debug(
        'Generated STM: length=%d, delay_max=%d, train_end=%d',
        length,
        delay_max,
        train_end,
    )
    return STMDataset(u=u, targets=targets, delay_max=delay_max, train_end=train_end)
