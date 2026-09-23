"""Qiskit-based cross-check of the reservoir features, for gate G0.5 (docs/DECISIONS.md D010).

The circuit is built independently from the same angles: this module imports nothing from
PennyLane or from qrc_thresher. It implements D010's schedule directly. With window w, qubit j
re-uploads u_{t-(j mod w)} at every layer through RY(pi * u), with 0 where t - (j mod w) < 0;
then RZ(theta_{d,j}) and RX(phi_{d,j}) on every qubit and the CNOT ring j -> (j + 1) mod n.
The features are the Z expectations, then ZZ in PennyLane's pair order (i < j, lexicographic).

Every circuit of a case runs untranspiled in one Aer statevector job (RY, RZ, RX and CX are Aer
basis gates). CROSSCHECK_TOLERANCE = 1e-6 (float64) is the one tolerance the gate uses; float32
paths get their own pre-registered tolerance (D006).
"""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

CROSSCHECK_TOLERANCE = 1e-6  # float64; the single definition (defect D6)
_ENCODING_SCALE = math.pi


def window_inputs(u: np.ndarray, window: int, n_qubits: int) -> np.ndarray:
    """Per-qubit inputs: row t, column j is u_{t-(j mod w)}, or 0 where that index is negative."""
    if not 1 <= int(window) <= int(n_qubits):
        raise ValueError(f'window must be in 1..n_qubits = 1..{n_qubits}; got {window}')
    u = np.asarray(u, dtype=np.float64).ravel()
    out = np.zeros((len(u), n_qubits), dtype=np.float64)
    for j in range(n_qubits):
        lag = j % int(window)
        out[lag:, j] = u[: len(u) - lag]
    return out


def _circuit(x, thetas, phis, n_qubits: int, depth: int):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n_qubits)
    for d in range(depth):
        for j in range(n_qubits):
            qc.ry(_ENCODING_SCALE * float(x[j]), j)
        for j in range(n_qubits):
            qc.rz(float(thetas[d, j]), j)
            qc.rx(float(phis[d, j]), j)
        for j in range(n_qubits):
            qc.cx(j, (j + 1) % n_qubits)
    qc.save_statevector()
    return qc


def _z_signs(n_qubits: int) -> np.ndarray:
    """z[q, idx] = +1 if qubit q is 0 in basis state idx, else -1 (Qiskit's little-endian order)."""
    idx = np.arange(2**n_qubits)
    return np.array([1.0 - 2.0 * ((idx >> q) & 1) for q in range(n_qubits)], dtype=np.float64)


def _expectations(probabilities: np.ndarray, n_qubits: int, readout: str) -> np.ndarray:
    z = _z_signs(n_qubits)
    values = [float(np.sum(probabilities * z[q])) for q in range(n_qubits)]
    if readout == 'z_and_zz':
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                values.append(float(np.sum(probabilities * z[i] * z[j])))
    elif readout != 'z_only':
        raise ValueError(f"readout must be 'z_only' or 'z_and_zz'; got {readout!r}")
    return np.array(values, dtype=np.float64)


def qiskit_features(
    u: np.ndarray,
    thetas: np.ndarray,
    phis: np.ndarray,
    n_qubits: int,
    depth: int,
    window: int,
    readout: str,
) -> np.ndarray:
    """Feature matrix of shape (T, F) from Qiskit Aer, for D010's windowed schedule.

    Args:
        u: Input sequence, shape (T,).
        thetas: RZ angles, shape (depth, n_qubits).
        phis: RX angles, shape (depth, n_qubits).
        n_qubits: Number of qubits n.
        depth: Number of layers L.
        window: The window w, in 1..n_qubits.
        readout: 'z_only' (F = n) or 'z_and_zz' (F = n + n(n-1)/2).
    """
    from qiskit_aer import AerSimulator

    thetas = np.asarray(thetas, dtype=np.float64)
    phis = np.asarray(phis, dtype=np.float64)
    if thetas.shape != (depth, n_qubits) or phis.shape != (depth, n_qubits):
        raise ValueError(f'angles must have shape {(depth, n_qubits)}')
    inputs = window_inputs(u, window, n_qubits)
    circuits = [_circuit(inputs[t], thetas, phis, n_qubits, depth) for t in range(len(inputs))]
    if not circuits:
        n_features = n_qubits if readout == 'z_only' else n_qubits + n_qubits * (n_qubits - 1) // 2
        return np.zeros((0, n_features), dtype=np.float64)
    sim = AerSimulator(method='statevector')
    result = sim.run(circuits).result()  # one untranspiled job per case (D010)
    rows = []
    for t in range(len(circuits)):
        amplitudes = np.asarray(result.get_statevector(t), dtype=np.complex128)
        rows.append(_expectations(np.abs(amplitudes) ** 2, n_qubits, readout))
    return np.stack(rows, axis=0)


def qiskit_expectation_values(
    u_t: float,
    thetas: np.ndarray,
    phis: np.ndarray,
    n_qubits: int,
    depth: int,
) -> np.ndarray:
    """Z expectations for a single step of the w = 1 circuit (kept for compatibility).

    Returns:
        Array of <Z_i> expectation values of shape (n_qubits,).
    """
    return qiskit_features(np.array([u_t]), thetas, phis, n_qubits, depth, 1, 'z_only')[0]


def verify_crosscheck(
    pennylane_values: np.ndarray,
    qiskit_values: np.ndarray,
    tolerance: float = CROSSCHECK_TOLERANCE,
) -> bool:
    """Verify that PennyLane and Qiskit values match within ``tolerance`` (max |diff|).

    Args:
        pennylane_values: Expectation values from PennyLane.
        qiskit_values: Expectation values from Qiskit.
        tolerance: Maximum allowed absolute difference (default CROSSCHECK_TOLERANCE).

    Returns:
        True if all values match within tolerance, False otherwise.
    """
    pennylane_values = np.asarray(pennylane_values, dtype=np.float64)
    qiskit_values = np.asarray(qiskit_values, dtype=np.float64)
    if pennylane_values.shape != qiskit_values.shape:
        raise ValueError(
            f'shape mismatch: {pennylane_values.shape} vs {qiskit_values.shape}'
        )
    max_diff = float(np.max(np.abs(pennylane_values - qiskit_values)))
    logger.info('Crosscheck max diff: %.2e (tol=%.2e)', max_diff, tolerance)
    return bool(max_diff <= tolerance)
