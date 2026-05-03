"""Temporal Parity / XOR task generator.

Input: binary u_t in {0, 1}, length T.
Target: y_t = XOR of u_{t-d+1} ... u_t for window d.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ParityDataset:
    """Container for temporal parity task data."""

    u: np.ndarray  # shape (T,) binary
    targets: np.ndarray  # shape (T,) binary (XOR output)
    window: int
    train_end: int


def generate_parity(
    length: int,
    window: int,
    train_frac: float,
    rng: Generator,
) -> ParityDataset:
    """Generate temporal parity task data.

    Args:
        length: Total sequence length T.
        window: XOR window size d.
        train_frac: Fraction of data for training (chronological split).
        rng: Seeded random generator.

    Returns:
        ParityDataset with binary u and targets (XOR over window).
    """
    u = rng.integers(0, 2, size=length).astype(np.int32)
    targets = np.zeros(length, dtype=np.int32)
    for t in range(window - 1, length):
        targets[t] = int(np.bitwise_xor.reduce(u[t - window + 1 : t + 1]))
    train_end = int(length * train_frac)
    logger.debug(
        'Generated parity: length=%d, window=%d, train_end=%d',
        length,
        window,
        train_end,
    )
    return ParityDataset(u=u, targets=targets, window=window, train_end=train_end)
