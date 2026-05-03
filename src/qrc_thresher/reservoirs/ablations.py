"""Ablation variants of the quantum reservoir (Section 7.4).

Implements:
- Phase-randomized: random phases at each time step (no coherent dynamics).
- Entanglement-suppressed: single-qubit rotations only, no CNOT.
- Haar-random unitary: apply U ~ Haar(2^N), measure Pauli-Z expectations.
"""

from __future__ import annotations

import logging

import numpy as np
import pennylane as qml
from numpy.random import Generator

from qrc_thresher.reservoirs.pennylane_qrc import _ENCODING_SCALE, QRCParams

logger = logging.getLogger(__name__)


def extract_features_phase_random(
    u: np.ndarray,
    params: QRCParams,
    rng: Generator,
) -> np.ndarray:
    """Phase-randomized ablation: random angles at each time step.

    Replaces fixed coherent dynamics with random phases per step.
    Tests whether QRC reduces to random projection.

    Args:
        u: Input sequence of shape (T,).
        params: Original reservoir parameters (n_qubits, depth, backend used).
        rng: Seeded generator for random phases.

    Returns:
        Feature matrix of shape (T, n_qubits).
    """
    dev = qml.device(params.backend, wires=params.n_qubits)
    T = len(u)
    features = np.zeros((T, params.n_qubits), dtype=np.float64)

    for t in range(T):
        u_t = float(u[t])
        # Draw fresh random phases each time step
        thetas_t = rng.uniform(0.0, 2.0 * np.pi, size=(params.depth, params.n_qubits))
        phis_t = rng.uniform(0.0, 2.0 * np.pi, size=(params.depth, params.n_qubits))

        # Use default argument binding to capture loop variables correctly
        def make_circuit(u_val: float, th: np.ndarray, ph: np.ndarray) -> list:
            @qml.qnode(dev)
            def circuit_t() -> list:
                for d in range(params.depth):
                    for i in range(params.n_qubits):
                        qml.RY(_ENCODING_SCALE * u_val, wires=i)
                    for i in range(params.n_qubits):
                        qml.RZ(th[d, i], wires=i)
                        qml.RX(ph[d, i], wires=i)
                    for i in range(params.n_qubits):
                        qml.CNOT(wires=[i, (i + 1) % params.n_qubits])
                return [qml.expval(qml.PauliZ(i)) for i in range(params.n_qubits)]

            return circuit_t()  # type: ignore[return-value]

        features[t] = np.array(make_circuit(u_t, thetas_t, phis_t), dtype=np.float64)

    if not np.isfinite(features).all():
        raise ValueError('Phase-randomized ablation features contain non-finite values')
    return features


def extract_features_no_entangle(
    u: np.ndarray,
    params: QRCParams,
) -> np.ndarray:
    """Entanglement-suppressed ablation: remove all two-qubit gates.

    Tests whether entanglement contributes to QRC performance.

    Args:
        u: Input sequence of shape (T,).
        params: Reservoir parameters (CNOT gates are omitted).

    Returns:
        Feature matrix of shape (T, n_qubits).
    """
    dev = qml.device(params.backend, wires=params.n_qubits)
    T = len(u)
    features = np.zeros((T, params.n_qubits), dtype=np.float64)

    for t in range(T):
        u_t = float(u[t])

        def make_circuit(u_val: float) -> list:
            @qml.qnode(dev)
            def circuit_t() -> list:
                for d in range(params.depth):
                    for i in range(params.n_qubits):
                        qml.RY(_ENCODING_SCALE * u_val, wires=i)
                    for i in range(params.n_qubits):
                        qml.RZ(params.thetas[d, i], wires=i)
                        qml.RX(params.phis[d, i], wires=i)
                    # No entangling gates
                return [qml.expval(qml.PauliZ(i)) for i in range(params.n_qubits)]

            return circuit_t()  # type: ignore[return-value]

        features[t] = np.array(make_circuit(u_t), dtype=np.float64)

    if not np.isfinite(features).all():
        raise ValueError('No-entangle ablation features contain non-finite values')
    return features


def extract_features_haar(
    u: np.ndarray,
    n_qubits: int,
    backend: str,
    rng: Generator,
) -> np.ndarray:
    """Haar-random unitary ablation (G2.5 mandatory).

    Samples U ~ Haar(2^N), applies to angle-encoded state, measures <Z_i>.

    Args:
        u: Input sequence of shape (T,).
        n_qubits: Number of qubits N.
        backend: PennyLane device string.
        rng: Seeded generator.

    Returns:
        Feature matrix of shape (T, n_qubits).
    """
    from scipy.stats import unitary_group

    # Sample one Haar-random unitary (fixed for the whole sequence)
    U = unitary_group.rvs(2**n_qubits, random_state=int(rng.integers(0, 2**31)))
    dev = qml.device(backend, wires=n_qubits)
    T = len(u)
    features = np.zeros((T, n_qubits), dtype=np.float64)

    for t in range(T):
        u_t = float(u[t])

        def make_circuit(u_val: float) -> list:
            @qml.qnode(dev)
            def circuit_t() -> list:
                for i in range(n_qubits):
                    qml.RY(_ENCODING_SCALE * u_val, wires=i)
                qml.QubitUnitary(U, wires=list(range(n_qubits)))
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

            return circuit_t()  # type: ignore[return-value]

        features[t] = np.array(make_circuit(u_t), dtype=np.float64)

    if not np.isfinite(features).all():
        raise ValueError('Haar ablation features contain non-finite values')
    return features
