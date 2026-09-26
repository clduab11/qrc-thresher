"""The readout's feature count F, shared by every model (CP5 item D.3).

F = n for the z_only readout (one <Z_i> per qubit) and n + n(n-1)/2 for z_and_zz (the <Z_i Z_j>
pairs beside them). The ESN and RKS baselines are matched to this F, never to 2^n. The Qiskit
cross-check keeps its own copy on purpose (it is G0.5's independent build).
"""

from __future__ import annotations

READOUTS = ('z_only', 'z_and_zz')


def n_features(n_qubits: int, readout: str) -> int:
    """Number of readout features for ``n_qubits`` qubits under ``readout``.

    Raises:
        ValueError: For a readout other than 'z_only' or 'z_and_zz'.
    """
    if readout not in READOUTS:
        raise ValueError(f'unknown readout {readout!r}; choose from {list(READOUTS)}')
    n = int(n_qubits)
    return n if readout == 'z_only' else n + n * (n - 1) // 2


__all__ = ['READOUTS', 'n_features']
