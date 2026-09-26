"""Deprecated shim for ``reservoirs.smoothed_qrc`` (docs/DECISIONS.md D018; defect D2).

The former "stateful" extractor carried no quantum state: it smooths the memoryless features
classically. Use ``extract_features_smoothed``. ``extract_features_stateful`` returns identical
values and warns with a DeprecationWarning on every call (never at import); nothing is deleted.
"""

from __future__ import annotations

import warnings

import numpy as np

from qrc_thresher.reservoirs.pennylane_qrc import QRCParams
from qrc_thresher.reservoirs.smoothed_qrc import SmoothedQRCResult, extract_features_smoothed

StatefulQRCResult = SmoothedQRCResult  # the same class, kept importable


def extract_features_stateful(
    u: np.ndarray,
    params: QRCParams,
    carry_depth: int = 1,
) -> SmoothedQRCResult:
    """Deprecated: identical to ``extract_features_smoothed`` (classical smoothing, not memory)."""
    warnings.warn(
        'extract_features_stateful is deprecated: it adds classical smoothing, not quantum '
        'memory; use qrc_thresher.reservoirs.smoothed_qrc.extract_features_smoothed (D018)',
        DeprecationWarning,
        stacklevel=2,
    )
    return extract_features_smoothed(u, params, carry_depth=carry_depth)


__all__ = ['StatefulQRCResult', 'extract_features_stateful']
