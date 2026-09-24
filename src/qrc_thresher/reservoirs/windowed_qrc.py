"""Windowed-input quantum reservoir, design (a), with its matched ablations.

Decisions: docs/DECISIONS.md D003 and D010. Defects D1 and D12.

With window w, qubit j re-uploads u_{t-(j mod w)} at every layer, using 0 where
t - (j mod w) < 0. The rest of each layer is today's: RZ(theta_{d,j}) and RX(phi_{d,j}) on every
qubit, then the CNOT ring j -> (j + 1) mod n. Every row starts from |0...0>, so row t depends on
u_t ... u_{t-w+1} only: the memory cliff sits at delay k = w. At w = 1 this is today's circuit
(reservoirs/pennylane_qrc.py, kept unchanged as the reference), bit for bit.

The window draws no randomness: the angles come from build_reservoir_params with
default_rng(reservoir_seed), so they are the same at every w. The encoding scale alpha of
RY(alpha * u) is a tuned hyperparameter (D011; pi is today's circuit). The circuit hash starts
from compute_circuit_hash(params), appends ",window=<w>" when w > 1, then always
",encoding_scale=<repr(float(alpha))>" (D011, PI ruling 8), then any ablation suffix.

Matched ablations change one factor and inherit everything else (seed, readout, window and the
re-upload schedule):

- no_entangle removes the CNOT ring (``entangle=False``);
- phase_random draws fresh RZ and RX angles at every step from default_rng([reservoir_seed, 1])
  (``random_phases=True``);
- haar replaces each layer's RZ, RX and CNOT ring, after that layer's re-upload, with an
  independent Haar-random unitary from default_rng([reservoir_seed, 2]) (``layer_unitaries``).

An ablation's hash extends the reservoir's: ",entangle=False", ",random_phases=[<seed>,1]" or
",layer_unitaries=<SHA-256 of the unitaries' bytes>". Restoring the factor restores the hash.

RKS (random_features) is not a circuit variant: rks_from_config gives it the reservoir's
feature count F and the stream default_rng([reservoir_seed, 3]); its bandwidth is sigma / sqrt(d)
on the zero-padded input window of dimension d (D011, defect D13), with sigma = 1 and d = 1 as
the untuned default.

Every config-driven build goes through reservoir_from_config, which reads reservoir.window.
Within one features() call, each distinct per-qubit input row is simulated once and reused
(not for phase_random, whose angles change every step); the per-row QNode is the same, so the
result is bit for bit the per-step loop.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pennylane as qml

from qrc_thresher.config import AlphaLiteConfig
from qrc_thresher.reservoirs.pennylane_qrc import (
    QRCParams,
    build_reservoir_params,
    compute_circuit_hash,
)

logger = logging.getLogger(__name__)

ABLATIONS = ('phase_random', 'no_entangle', 'haar')
# Random streams are default_rng([reservoir_seed, tag]) (D010).
ABLATION_TAGS: Dict[str, int] = {'phase_random': 1, 'haar': 2, 'random_features': 3}
# Feeding each layer the unitary of its own RZ, RX and CNOT ring reproduces the reservoir
# within this maximum absolute difference (D010): float64 rounding only, of order 1e-15.
HAAR_SWITCH_BACK_TOLERANCE = 1e-10
RKS_SIGMA = 1.0  # the untuned RKS bandwidth; the window d = 1 (D010 default, D011)
ENCODING_SCALE_DEFAULT = math.pi  # today's circuit; the untuned default (D011)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def check_window(window: int, n_qubits: int) -> int:
    """Return ``window`` if it lies in 1..n_qubits, else raise ValueError."""
    if not isinstance(window, (int, np.integer)) or isinstance(window, bool):
        raise ValueError(f'reservoir.window must be an integer in 1..n_qubits; got {window!r}')
    if not 1 <= int(window) <= int(n_qubits):
        raise ValueError(
            f'reservoir.window must be in 1..n_qubits = 1..{n_qubits}; got {window}'
        )
    return int(window)


def window_inputs(u: np.ndarray, window: int, n_qubits: int) -> np.ndarray:
    """Per-qubit inputs: row t, column j is u_{t-(j mod w)}, or 0 where that index is negative.

    Args:
        u: Input sequence, shape (T,).
        window: The window w, in 1..n_qubits.
        n_qubits: Number of qubits n.

    Returns:
        Array of shape (T, n).
    """
    window = check_window(window, n_qubits)
    u = np.asarray(u, dtype=np.float64).ravel()
    T = len(u)
    out = np.zeros((T, n_qubits), dtype=np.float64)
    for j in range(n_qubits):
        lag = j % window
        out[lag:, j] = u[: T - lag]
    return out


def phase_random_angles(
    T: int, depth: int, n_qubits: int, reservoir_seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Fresh RZ and RX angles for every step, from default_rng([reservoir_seed, 1]).

    For t = 0, 1, ... the stream draws the step's RZ angles and then its RX angles, each
    Uniform[0, 2 pi) of shape (depth, n_qubits).

    Returns:
        (thetas, phis), each of shape (T, depth, n_qubits).
    """
    rng = np.random.default_rng([int(reservoir_seed), ABLATION_TAGS['phase_random']])
    thetas = np.empty((T, depth, n_qubits), dtype=np.float64)
    phis = np.empty((T, depth, n_qubits), dtype=np.float64)
    for t in range(T):
        thetas[t] = rng.uniform(0.0, 2.0 * np.pi, size=(depth, n_qubits))
        phis[t] = rng.uniform(0.0, 2.0 * np.pi, size=(depth, n_qubits))
    return thetas, phis


def haar_layer_unitaries(n_qubits: int, depth: int, reservoir_seed: int) -> Tuple[np.ndarray, ...]:
    """One independent Haar-random unitary on all n qubits per layer, in layer order.

    Drawn with scipy.stats.unitary_group.rvs(2^n, random_state=default_rng([reservoir_seed, 2])).
    """
    from scipy.stats import unitary_group

    rng = np.random.default_rng([int(reservoir_seed), ABLATION_TAGS['haar']])
    return tuple(
        np.ascontiguousarray(unitary_group.rvs(2**n_qubits, random_state=rng), dtype=np.complex128)
        for _ in range(depth)
    )


def _unitaries_digest(unitaries: Tuple[np.ndarray, ...]) -> str:
    """SHA-256 of the L complex128 matrices in C order, concatenated in layer order."""
    h = hashlib.sha256()
    for U in unitaries:
        h.update(np.ascontiguousarray(U, dtype=np.complex128).tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class WindowedReservoir:
    """A windowed reservoir, or one matched ablation of it (docs/DECISIONS.md D010).

    Attributes:
        params: Angles, size, depth, readout and backend (build_reservoir_params).
        window: The window w, in 1..n_qubits.
        reservoir_seed: The seed the angles came from; needed by phase_random and haar.
        ablation: The ablation's name, for the record, or None for the reservoir itself.
        entangle: False removes the CNOT ring (no_entangle).
        random_phases: True draws fresh RZ and RX angles at every step (phase_random).
        layer_unitaries: One unitary per layer replacing that layer's RZ, RX and CNOT ring
            (haar), or None.
        encoding_scale: The angle-encoding scale alpha of RY(alpha * u) at every re-upload
            (D011); pi is today's circuit.

    ``circuit_hash`` is derived from these fields, never stored, so dataclasses.replace keeps
    it consistent.
    """

    params: QRCParams
    window: int = 1
    reservoir_seed: Optional[int] = None
    ablation: Optional[str] = None
    entangle: bool = True
    random_phases: bool = False
    layer_unitaries: Optional[Tuple[np.ndarray, ...]] = None
    encoding_scale: float = ENCODING_SCALE_DEFAULT

    def __post_init__(self) -> None:
        n = self.params.n_qubits
        object.__setattr__(self, 'window', check_window(self.window, n))
        object.__setattr__(self, 'entangle', bool(self.entangle))
        object.__setattr__(self, 'random_phases', bool(self.random_phases))
        scale = float(self.encoding_scale)
        if not (math.isfinite(scale) and scale > 0):
            raise ValueError(f'encoding_scale must be a finite positive number; got {scale!r}')
        object.__setattr__(self, 'encoding_scale', scale)
        if self.random_phases and self.reservoir_seed is None:
            raise ValueError('phase_random needs reservoir_seed for its random stream')
        if self.layer_unitaries is not None:
            unitaries = tuple(
                np.ascontiguousarray(U, dtype=np.complex128) for U in self.layer_unitaries
            )
            if len(unitaries) != self.params.depth:
                raise ValueError(
                    f'layer_unitaries needs one unitary per layer ({self.params.depth}); '
                    f'got {len(unitaries)}'
                )
            for U in unitaries:
                if U.shape != (2**n, 2**n):
                    raise ValueError(
                        f'layer unitaries must have shape {(2**n, 2**n)}; got {U.shape}'
                    )
            object.__setattr__(self, 'layer_unitaries', unitaries)

    @property
    def n_qubits(self) -> int:
        return self.params.n_qubits

    @property
    def depth(self) -> int:
        return self.params.depth

    @property
    def n_features(self) -> int:
        n = self.params.n_qubits
        return n if self.params.readout == 'z_only' else n + n * (n - 1) // 2

    @property
    def circuit_hash(self) -> str:
        """SHA-256 identifying the simulated circuit (D010 recipes, D011 scale suffix)."""
        h = compute_circuit_hash(self.params)
        if self.window > 1:
            h = _sha256(f'{h},window={self.window}')
        h = _sha256(f'{h},encoding_scale={float(self.encoding_scale)!r}')  # always (ruling 8)
        if not self.entangle:
            h = _sha256(f'{h},entangle=False')
        if self.random_phases:
            tag = ABLATION_TAGS['phase_random']
            h = _sha256(f'{h},random_phases=[{self.reservoir_seed},{tag}]')
        if self.layer_unitaries is not None:
            h = _sha256(f'{h},layer_unitaries={_unitaries_digest(self.layer_unitaries)}')
        return h

    def _circuit(self, thetas: np.ndarray, phis: np.ndarray):
        """QNode for one row: takes the n per-qubit inputs, returns the F expectation values."""
        params = self.params
        n, depth = params.n_qubits, params.depth
        entangle, unitaries = self.entangle, self.layer_unitaries
        scale = self.encoding_scale
        dev = qml.device(params.backend, wires=n)

        @qml.qnode(dev)
        def circuit(x: List[float]) -> list:
            for d in range(depth):
                for j in range(n):
                    qml.RY(scale * x[j], wires=j)
                if unitaries is not None:
                    qml.QubitUnitary(unitaries[d], wires=list(range(n)))
                    continue
                for j in range(n):
                    qml.RZ(thetas[d, j], wires=j)
                    qml.RX(phis[d, j], wires=j)
                if entangle:
                    for j in range(n):
                        qml.CNOT(wires=[j, (j + 1) % n])
            obs = [qml.expval(qml.PauliZ(j)) for j in range(n)]
            if params.readout == 'z_and_zz':
                for i in range(n):
                    for j in range(i + 1, n):
                        obs.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)))
            return obs  # type: ignore[return-value]

        return circuit

    def features(self, u: np.ndarray, cache: bool = True) -> np.ndarray:
        """Feature matrix of shape (T, F) for an input sequence of shape (T,).

        Args:
            u: Input sequence.
            cache: Simulate each distinct per-qubit input row once per call and reuse it.
                Ignored for phase_random, whose angles change at every step. The result is bit
                for bit the per-step loop either way.

        Raises:
            ValueError: If the features contain non-finite values.
        """
        u = np.asarray(u, dtype=np.float64).ravel()
        inputs = window_inputs(u, self.window, self.params.n_qubits)
        T = len(u)
        rows: List[np.ndarray] = []
        if self.random_phases:
            thetas, phis = phase_random_angles(
                T, self.params.depth, self.params.n_qubits, self.reservoir_seed
            )
            for t in range(T):
                circuit = self._circuit(thetas[t], phis[t])
                rows.append(np.array(circuit([float(v) for v in inputs[t]]), dtype=np.float64))
        else:
            circuit = self._circuit(self.params.thetas, self.params.phis)
            seen: Dict[tuple, np.ndarray] = {}
            for t in range(T):
                x = [float(v) for v in inputs[t]]
                key = tuple(x)
                if cache and key in seen:
                    rows.append(seen[key])
                    continue
                row = np.array(circuit(x), dtype=np.float64)
                if cache:
                    seen[key] = row
                rows.append(row)
        X = np.stack(rows, axis=0) if rows else np.zeros((0, self.n_features), dtype=np.float64)
        if not np.isfinite(X).all():
            raise ValueError('windowed reservoir features contain non-finite values')
        logger.debug('Windowed features: window=%d shape=%s', self.window, X.shape)
        return X


def _ablation_fields(
    ablation: Optional[str], params: QRCParams, reservoir_seed: Optional[int]
) -> dict:
    if ablation is None:
        return {}
    if ablation == 'no_entangle':
        return {'entangle': False}
    if ablation == 'phase_random':
        if reservoir_seed is None:
            raise ValueError('phase_random needs reservoir_seed for its random stream')
        return {'random_phases': True}
    if ablation == 'haar':
        if reservoir_seed is None:
            raise ValueError('haar needs reservoir_seed for its random stream')
        return {
            'layer_unitaries': haar_layer_unitaries(params.n_qubits, params.depth, reservoir_seed)
        }
    if ablation == 'random_features':
        raise ValueError(
            'random_features is not a circuit ablation; build it with rks_from_config'
        )
    raise ValueError(f'unknown ablation {ablation!r}; choose from {list(ABLATIONS)}')


def reservoir_from_config(
    cfg: AlphaLiteConfig, reservoir_seed: int, ablation: Optional[str] = None
) -> WindowedReservoir:
    """Build the configured reservoir, or one matched ablation of it, for a reservoir seed.

    The angles come from build_reservoir_params with default_rng(reservoir_seed), as every
    harness build draws them; the window is cfg.reservoir.window and the encoding scale
    cfg.reservoir.encoding_scale.

    Raises:
        ValueError: For an unknown ablation name, or 'random_features' (see rks_from_config).
    """
    params = build_reservoir_params(
        n_qubits=cfg.reservoir.n_qubits,
        depth=cfg.reservoir.depth,
        readout=cfg.reservoir.readout,
        backend=cfg.reservoir.backend,
        rng=np.random.default_rng(reservoir_seed),
    )
    fields = _ablation_fields(ablation, params, reservoir_seed)
    return WindowedReservoir(
        params=params,
        window=cfg.reservoir.window,
        reservoir_seed=int(reservoir_seed),
        ablation=ablation,
        encoding_scale=cfg.reservoir.encoding_scale,
        **fields,
    )


def n_features_from_config(cfg: AlphaLiteConfig) -> int:
    n = cfg.reservoir.n_qubits
    return n if cfg.reservoir.readout == 'z_only' else n + n * (n - 1) // 2


def rks_from_config(
    cfg: AlphaLiteConfig, reservoir_seed: int, sigma: float = RKS_SIGMA, window: int = 1
):
    """RKS parameters matched to the reservoir: F features, stream default_rng([seed, 3]),
    bandwidth sigma / sqrt(window) on the zero-padded input window (D011)."""
    from qrc_thresher.baselines.random_features import build_rks_params

    rng = np.random.default_rng([int(reservoir_seed), ABLATION_TAGS['random_features']])
    return build_rks_params(
        n_features=n_features_from_config(cfg), sigma=float(sigma), rng=rng, window=int(window)
    )


def rks_circuit_hash(rks_params) -> str:
    """SHA-256 of F, sigma, the window d, W and b: identifies an RKS row's features (D011)."""
    h = hashlib.sha256()
    h.update(
        f'rks:n_features={rks_params.n_features},sigma={rks_params.sigma!r},'
        f'window={rks_params.window}:'.encode()
    )
    h.update(np.ascontiguousarray(rks_params.W, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(rks_params.b, dtype=np.float64).tobytes())
    return h.hexdigest()


# Builtin reservoir plugins (plugins/builtin.py). All take (u, params, window=1,
# reservoir_seed=None) and return features of shape (T, F).
def plugin_windowed(u, params: QRCParams, window: int = 1, reservoir_seed=None) -> np.ndarray:
    """The windowed reservoir itself."""
    reservoir = WindowedReservoir(params=params, window=window, reservoir_seed=reservoir_seed)
    return reservoir.features(u)


def plugin_no_entangle(u, params: QRCParams, window: int = 1, reservoir_seed=None) -> np.ndarray:
    """Matched no-entangle ablation: the CNOT ring removed."""
    return WindowedReservoir(
        params=params, window=window, reservoir_seed=reservoir_seed, ablation='no_entangle',
        entangle=False,
    ).features(u)


def plugin_phase_random(u, params: QRCParams, window: int = 1, reservoir_seed=None) -> np.ndarray:
    """Matched phase-random ablation: fresh RZ and RX angles at every step."""
    if reservoir_seed is None:
        raise ValueError('the phase_random plugin needs reservoir_seed for its random stream')
    return WindowedReservoir(
        params=params, window=window, reservoir_seed=int(reservoir_seed),
        ablation='phase_random', random_phases=True,
    ).features(u)


def plugin_haar(u, params: QRCParams, window: int = 1, reservoir_seed=None) -> np.ndarray:
    """Matched Haar ablation: an independent Haar unitary per layer after the re-upload."""
    if reservoir_seed is None:
        raise ValueError('the haar plugin needs reservoir_seed for its random stream')
    unitaries = haar_layer_unitaries(params.n_qubits, params.depth, int(reservoir_seed))
    return WindowedReservoir(
        params=params, window=window, reservoir_seed=int(reservoir_seed), ablation='haar',
        layer_unitaries=unitaries,
    ).features(u)


__all__ = [
    'ABLATIONS',
    'ABLATION_TAGS',
    'ENCODING_SCALE_DEFAULT',
    'HAAR_SWITCH_BACK_TOLERANCE',
    'WindowedReservoir',
    'check_window',
    'haar_layer_unitaries',
    'n_features_from_config',
    'phase_random_angles',
    'plugin_haar',
    'plugin_no_entangle',
    'plugin_phase_random',
    'plugin_windowed',
    'reservoir_from_config',
    'rks_circuit_hash',
    'rks_from_config',
    'window_inputs',
]
