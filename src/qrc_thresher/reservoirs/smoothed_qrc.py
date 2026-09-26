"""Classical smoothing of the memoryless QRC features (docs/DECISIONS.md D018; defect D2).

``extract_features_smoothed`` blends each feature row with the mean of row t and up to
carry_depth rows before it: features[t] = 0.5 * base[t] + 0.5 * mean(base[t - carry_depth : t + 1]),
where ``base`` is ``pennylane_qrc.extract_features``. This adds classical smoothing of the
feature stream, not quantum memory: no reservoir state is carried between steps. It is the
former ``stateful_qrc.extract_features_stateful``, renamed to say what it does; the old module
stays as a deprecated shim that returns identical values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qrc_thresher.reservoirs.pennylane_qrc import QRCParams, extract_features


@dataclass(frozen=True, slots=True)
class SmoothedQRCResult:
    """Smoothed features and the running mean (``state_trace``) they were blended with."""

    features: np.ndarray
    state_trace: np.ndarray


def extract_features_smoothed(
    u: np.ndarray,
    params: QRCParams,
    carry_depth: int = 1,
) -> SmoothedQRCResult:
    """Smooth the memoryless QRC features over a rolling window of ``carry_depth`` rows.

    Args:
        u: Input sequence of shape (T,).
        params: The reservoir (``build_reservoir_params``).
        carry_depth: Rows before t that enter the running mean (at least 1).

    Returns:
        SmoothedQRCResult with ``features`` (T, F) and ``state_trace`` (T, F), bit-identical
        to the former stateful extractor.

    Raises:
        ValueError: If carry_depth < 1.
    """
    if carry_depth < 1:
        raise ValueError(f'carry_depth must be >= 1, got {carry_depth}')

    base = extract_features(u, params)
    features = np.array(base, copy=True)
    state = np.zeros_like(features)

    for t in range(len(features)):
        start = max(0, t - carry_depth)
        window = base[start : t + 1]
        carried = np.mean(window, axis=0)
        state[t] = carried
        features[t] = 0.5 * base[t] + 0.5 * carried

    return SmoothedQRCResult(features=features, state_trace=state)


__all__ = ['SmoothedQRCResult', 'extract_features_smoothed']
