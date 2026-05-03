"""PennyLane-based quantum reservoir computer (QRC).

Architecture (Section 7.2):
1. Angle-encode input u_t via R_y(alpha * u_t) on each qubit.
2. Fixed random R_z(theta_i), R_x(phi_i) drawn at construction.
3. Fixed ring-topology entangling layer: CNOT between qubit i and (i+1) mod N.
4. Repeat steps 1-3 for depth d.
5. Measure <Z_i> for each qubit (optionally <Z_i Z_j> for i<j).
6. Stack readouts to form feature matrix X of shape (T, F).
7. Ridge regression on X to predict targets.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import pennylane as qml
from numpy.random import Generator
from sklearn.linear_model import RidgeCV

logger = logging.getLogger(__name__)

_ENCODING_SCALE = np.pi  # alpha factor for angle encoding


@dataclass(frozen=True, slots=True)
class QRCParams:
    """Immutable reservoir parameter container."""

    thetas: np.ndarray  # shape (depth, n_qubits) R_z angles
    phis: np.ndarray  # shape (depth, n_qubits) R_x angles
    n_qubits: int
    depth: int
    readout: str
    backend: str


def build_reservoir_params(
    n_qubits: int,
    depth: int,
    readout: str,
    backend: str,
    rng: Generator,
) -> QRCParams:
    """Draw fixed random rotation angles for the reservoir.

    Args:
        n_qubits: Number of qubits N.
        depth: Circuit depth d.
        readout: 'z_only' or 'z_and_zz'.
        backend: PennyLane device string.
        rng: Seeded random generator.

    Returns:
        QRCParams with fixed angles.
    """
    thetas = rng.uniform(0.0, 2.0 * np.pi, size=(depth, n_qubits))
    phis = rng.uniform(0.0, 2.0 * np.pi, size=(depth, n_qubits))
    return QRCParams(
        thetas=thetas,
        phis=phis,
        n_qubits=n_qubits,
        depth=depth,
        readout=readout,
        backend=backend,
    )


def _build_circuit(params: QRCParams) -> qml.QNode:
    """Build a PennyLane QNode for one time-step feature extraction.

    Args:
        params: Reservoir parameters.

    Returns:
        QNode that takes (u_t: float) and returns expectation values.
    """
    dev = qml.device(params.backend, wires=params.n_qubits)

    @qml.qnode(dev)
    def circuit(u_t: float) -> list:
        for d in range(params.depth):
            # Step 1: angle encoding
            for i in range(params.n_qubits):
                qml.RY(_ENCODING_SCALE * u_t, wires=i)
            # Step 2: fixed single-qubit rotations
            for i in range(params.n_qubits):
                qml.RZ(params.thetas[d, i], wires=i)
                qml.RX(params.phis[d, i], wires=i)
            # Step 3: ring entangling layer
            for i in range(params.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % params.n_qubits])
        # Step 5: readout
        obs = [qml.expval(qml.PauliZ(i)) for i in range(params.n_qubits)]
        if params.readout == 'z_and_zz':
            for i in range(params.n_qubits):
                for j in range(i + 1, params.n_qubits):
                    obs.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)))
        return obs  # type: ignore[return-value]

    return circuit


def extract_features(
    u: np.ndarray,
    params: QRCParams,
) -> np.ndarray:
    """Extract reservoir feature matrix from input sequence.

    Args:
        u: Input sequence of shape (T,).
        params: Reservoir parameters.

    Returns:
        Feature matrix of shape (T, F) where F = n_qubits (z_only)
        or n_qubits + n_qubits*(n_qubits-1)/2 (z_and_zz).

    Raises:
        ValueError: If extracted features contain NaN or inf.
    """
    circuit = _build_circuit(params)
    features_list = []
    for u_t in u:
        feats = np.array(circuit(float(u_t)), dtype=np.float64)
        features_list.append(feats)
    X = np.stack(features_list, axis=0)
    if not np.isfinite(X).all():
        raise ValueError(f'QRC features contain non-finite values: {X}')
    logger.debug('Extracted features: shape=%s', X.shape)
    return X


def compute_circuit_hash(params: QRCParams) -> str:
    """Compute a reproducibility hash from reservoir parameters.

    Args:
        params: Reservoir parameters.

    Returns:
        SHA-256 hex digest of canonical parameter representation.
    """
    canonical = (
        f'n_qubits={params.n_qubits},'
        f'depth={params.depth},'
        f'thetas={params.thetas.tobytes().hex()},'
        f'phis={params.phis.tobytes().hex()},'
        f'readout={params.readout}'
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def train_readout(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ridge_alphas: List[float],
    cv_folds: int,
) -> RidgeCV:
    """Train ridge regression readout on reservoir features.

    Args:
        X_train: Feature matrix of shape (T_train, F).
        y_train: Target matrix of shape (T_train,) or (T_train, K).
        ridge_alphas: Candidate regularization strengths.
        cv_folds: Number of CV folds.

    Returns:
        Fitted RidgeCV model.
    """
    model = RidgeCV(alphas=ridge_alphas, cv=cv_folds)
    model.fit(X_train, y_train)
    logger.debug('RidgeCV fitted: best_alpha=%s', getattr(model, 'alpha_', None))
    return model
