"""Stateful reservoir helpers for Phase 2 carry-depth experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qrc_thresher.reservoirs.pennylane_qrc import QRCParams, extract_features


@dataclass(frozen=True, slots=True)
class StatefulQRCResult:
    """Result container for stateful extraction."""

    features: np.ndarray
    state_trace: np.ndarray


def extract_features_stateful(
    u: np.ndarray,
    params: QRCParams,
    carry_depth: int = 1,
) -> StatefulQRCResult:
    """Extract features with a simple rolling carry-state approximation.

    This is a lightweight Phase 2 scaffold: it blends current and previous
    feature vectors to emulate carried internal state while preserving the
    existing deterministic QRC feature extractor.
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

    return StatefulQRCResult(features=features, state_trace=state)
