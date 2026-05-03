"""GRU baseline stub (Phase 1.5+, gated behind Phase 1 pass).

CPU-only. Only enabled if Phase 1 gates G1 and G2 pass.
"""

from __future__ import annotations


def train_gru(*args, **kwargs) -> None:  # noqa: ANN002, ANN003
    """GRU baseline — gated behind Phase 1 pass.

    Raises:
        NotImplementedError: Always, until Phase 1 gates pass.
    """
    raise NotImplementedError('GRU baseline gated behind Phase 1 pass')
