"""Qiskit-based cross-check for G5 validation.

Verifies that PennyLane and Qiskit produce identical expectation values
(within 1e-6) on the same circuit.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_CROSSCHECK_TOLERANCE = 1e-6


def qiskit_expectation_values(
    u_t: float,
    thetas: np.ndarray,
    phis: np.ndarray,
    n_qubits: int,
    depth: int,
) -> np.ndarray:
    """Compute Pauli-Z expectation values using Qiskit Aer statevector.

    Args:
        u_t: Single time-step input.
        thetas: Fixed R_z angles of shape (depth, n_qubits).
        phis: Fixed R_x angles of shape (depth, n_qubits).
        n_qubits: Number of qubits.
        depth: Circuit depth.

    Returns:
        Array of <Z_i> expectation values of shape (n_qubits,).
    """
    import math

    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator

    qc = QuantumCircuit(n_qubits)
    encoding_scale = math.pi

    for d in range(depth):
        for i in range(n_qubits):
            qc.ry(encoding_scale * u_t, i)
        for i in range(n_qubits):
            qc.rz(thetas[d, i], i)
            qc.rx(phis[d, i], i)
        for i in range(n_qubits):
            qc.cx(i, (i + 1) % n_qubits)

    qc.save_statevector()
    sim = AerSimulator(method='statevector')
    compiled = transpile(qc, sim)
    result = sim.run(compiled).result()
    sv = result.get_statevector(compiled)
    sv_arr = np.array(sv)

    expectations = np.zeros(n_qubits, dtype=np.float64)
    for qubit in range(n_qubits):
        pauli_z_diag = np.array([
            1.0 if ((idx >> qubit) & 1) == 0 else -1.0
            for idx in range(2**n_qubits)
        ])
        expectations[qubit] = np.real(np.sum(pauli_z_diag * np.abs(sv_arr) ** 2))

    return expectations


def verify_crosscheck(
    pennylane_values: np.ndarray,
    qiskit_values: np.ndarray,
    tolerance: float = _CROSSCHECK_TOLERANCE,
) -> bool:
    """Verify that PennyLane and Qiskit expectation values match.

    Args:
        pennylane_values: Expectation values from PennyLane.
        qiskit_values: Expectation values from Qiskit.
        tolerance: Maximum allowed absolute difference.

    Returns:
        True if all values match within tolerance, False otherwise.
    """
    max_diff = np.max(np.abs(pennylane_values - qiskit_values))
    logger.info('Crosscheck max diff: %.2e (tol=%.2e)', max_diff, tolerance)
    return bool(max_diff <= tolerance)
