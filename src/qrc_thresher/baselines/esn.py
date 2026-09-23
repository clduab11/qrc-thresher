"""Echo state network (ESN) baseline (defects D9 and D5; docs/DECISIONS.md D009).

The ESN is implemented directly in numpy with the standard leaky-integrator update

    x_t = (1 - a) x_{t-1} + a tanh(rho W_hat x_{t-1} + s W_in u_t),   x_{-1} = 0,

where W_hat is the recurrent draw normalised to spectral radius 1, rho the spectral radius,
s the input scaling and a the leak rate. There is no bias term. The update matches
ReservoirPy's Reservoir given the same matrices (tests/test_esn.py cross-checks it).

Wiring rule: every unit receives the input (input connectivity 1.0), and the recurrent
matrix is dense (recurrent connectivity 1.0, self-connections included). A draw whose
spectral radius cannot be scaled (non-finite, or at most 1e-8), or whose scaled weights
are non-finite or larger than 1e3 in magnitude, is refused.

One reservoir draw is made per reservoir_seed. Hyperparameters rescale that same draw, so
a search compares hyperparameters, not random draws, and the deployed ESN is the validated
one (identical weight hashes).

Feature matching: N_ESN = N_quantum_features (n for z_only, n + n(n-1)/2 for z_and_zz),
never 2^n.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from itertools import product
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

INPUT_CONNECTIVITY = 1.0  # every unit sees the input
RECURRENT_CONNECTIVITY = 1.0  # dense recurrent matrix
MIN_SPECTRAL_RADIUS = 1e-8  # a draw below this cannot be rescaled
MAX_ABS_WEIGHT = 1e3  # scaled weights above this are refused

GRID_KEYS = ('spectral_radius', 'input_scaling', 'leak_rate')


class NonFiniteStatesError(ValueError):
    """The reservoir states became non-finite: the configuration is degenerate."""


@dataclass(frozen=True)
class ESNParams:
    """Hyperparameters that rescale one reservoir draw."""

    spectral_radius: float
    input_scaling: float
    leak_rate: float


# Fixed configurations evaluated as G0.7 findings (docs/DECISIONS.md D009).
ESN_PRESETS: Dict[str, ESNParams] = {
    'esn_linear': ESNParams(spectral_radius=0.9, input_scaling=0.1, leak_rate=1.0),
    'esn_nonlinear': ESNParams(spectral_radius=0.9, input_scaling=1.0, leak_rate=1.0),
}


@dataclass(frozen=True)
class ReservoirDraw:
    """The random part of an ESN, drawn once per reservoir_seed.

    Attributes:
        reservoir_seed: Seed of the draw.
        W_unit: Dense recurrent weights normalised to spectral radius 1, shape (N, N).
        Win_unit: Dense input weights in [-1, 1] at input scaling 1, shape (N, 1).
    """

    reservoir_seed: int
    W_unit: np.ndarray
    Win_unit: np.ndarray

    @property
    def n_units(self) -> int:
        return int(self.W_unit.shape[0])


@dataclass(frozen=True)
class ESN:
    """A fully specified ESN: one draw rescaled by one set of hyperparameters."""

    W: np.ndarray
    Win: np.ndarray
    params: ESNParams
    reservoir_seed: int

    @property
    def n_units(self) -> int:
        return int(self.W.shape[0])

    def states(self, u: np.ndarray) -> np.ndarray:
        """Run the reservoir over an input sequence from x = 0.

        Args:
            u: Input sequence, shape (T,).

        Returns:
            States, shape (T, N). Row t depends on u_0 .. u_t only.
        """
        u = np.asarray(u, dtype=np.float64).ravel()
        a = self.params.leak_rate
        drive = u[:, None] * self.Win[:, 0][None, :]  # (T, N)
        x = np.zeros(self.n_units, dtype=np.float64)
        out = np.empty((len(u), self.n_units), dtype=np.float64)
        for t in range(len(u)):
            x = (1.0 - a) * x + a * np.tanh(drive[t] + self.W @ x)
            out[t] = x
        if not np.all(np.isfinite(out)):
            raise NonFiniteStatesError('ESN states became non-finite')
        return out

    def weight_hash(self) -> str:
        """SHA-256 of the scaled weights and leak rate: identifies the deployed ESN."""
        h = hashlib.sha256()
        h.update(f'esn:n={self.n_units}:a={self.params.leak_rate!r}:'.encode())
        h.update(np.ascontiguousarray(self.W, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(self.Win, dtype=np.float64).tobytes())
        return h.hexdigest()


def draw_reservoir(n_units: int, reservoir_seed: int) -> ReservoirDraw:
    """Draw one dense reservoir from default_rng(reservoir_seed): W first, then W_in.

    W has i.i.d. N(0, 1) entries and is normalised to spectral radius 1; W_in has i.i.d.
    Uniform(-1, 1) entries.

    Raises:
        ValueError: If n_units < 1, or the draw cannot be scaled (see scale_recurrent).
    """
    if n_units < 1:
        raise ValueError(f'n_units must be >= 1; got {n_units}')
    rng = np.random.default_rng(reservoir_seed)
    W = rng.normal(0.0, 1.0, size=(n_units, n_units))
    Win = rng.uniform(-1.0, 1.0, size=(n_units, 1))
    return ReservoirDraw(
        reservoir_seed=int(reservoir_seed), W_unit=scale_recurrent(W, 1.0), Win_unit=Win
    )


def scale_recurrent(W: np.ndarray, spectral_radius: float) -> np.ndarray:
    """Rescale a recurrent matrix to a target spectral radius, refusing unsafe draws.

    Raises:
        ValueError: If W is non-finite, its spectral radius is non-finite or at most
            MIN_SPECTRAL_RADIUS, or the scaled matrix fails the weight guard.
    """
    W = np.asarray(W, dtype=np.float64)
    if not np.all(np.isfinite(W)):
        raise ValueError('recurrent matrix has non-finite entries')
    rho = float(np.max(np.abs(np.linalg.eigvals(W)))) if W.size else 0.0
    if not np.isfinite(rho) or rho <= MIN_SPECTRAL_RADIUS:
        raise ValueError(
            f'recurrent matrix spectral radius {rho:.3g} cannot be scaled '
            f'(needs a finite value above {MIN_SPECTRAL_RADIUS:g})'
        )
    scaled = W * (spectral_radius / rho)
    _check_weights(scaled, 'recurrent')
    return scaled


def build_esn(draw: ReservoirDraw, params: ESNParams) -> ESN:
    """Rescale one draw by one set of hyperparameters.

    Raises:
        ValueError: If a hyperparameter is out of range or a weight fails the guard.
    """
    if params.spectral_radius < 0 or params.input_scaling < 0:
        raise ValueError(f'spectral radius and input scaling must be >= 0; got {params}')
    if not 0.0 < params.leak_rate <= 1.0:
        raise ValueError(f'leak rate must be in (0, 1]; got {params.leak_rate}')
    W = draw.W_unit * params.spectral_radius
    Win = draw.Win_unit * params.input_scaling
    _check_weights(W, 'recurrent')
    _check_weights(Win, 'input')
    return ESN(W=W, Win=Win, params=params, reservoir_seed=draw.reservoir_seed)


def esn_feature_map(n_units: int, params: ESNParams) -> Callable[[np.ndarray, int], np.ndarray]:
    """Feature map u, reservoir_seed -> ESN states, for G0.7 and other feature-map users."""

    def feature_map(u: np.ndarray, reservoir_seed: int) -> np.ndarray:
        return build_esn(draw_reservoir(n_units, reservoir_seed), params).states(u)

    return feature_map


@dataclass(frozen=True)
class ESNSearch:
    """Outcome of an ESN hyperparameter search on one reservoir draw.

    Attributes:
        best: Selected hyperparameters (first best in grid order).
        best_hash: Weight hash of the selected ESN, as validated.
        configs: One record per configuration, in grid order: params, mean validation
            score (None if degenerate), degeneracy flag and reason, weight hash.
        n_configs: Number of configurations evaluated.
        n_validation_evals: Configuration x validation-block evaluations.
        score_name: Validation score maximised ('mc', 'neg_nrmse' or 'accuracy').
    """

    best: ESNParams
    best_hash: str
    configs: List[dict]
    n_configs: int
    n_validation_evals: int
    score_name: str


def tune_esn(
    u: np.ndarray,
    targets: np.ndarray,
    train_end: int,
    draw: ReservoirDraw,
    grid: Dict[str, Sequence[float]],
    washout: int,
    ridge_alphas: Sequence[float],
    cv_folds: int,
    task: str,
) -> ESNSearch:
    """Select ESN hyperparameters on contiguous validation blocks of the training rows.

    Every configuration rescales the same draw. Its states come from one run over the
    whole input sequence (inputs only, never targets). The training rows after the
    washout, [washout, train_end), are split into ``cv_folds`` contiguous blocks; each
    block is predicted by the harness readout fitted on the other blocks (computed with
    readout.fit_ridge_cv_batched, which reproduces RidgeCV), and scored with the task's
    primary metric. Test rows are never used. A configuration whose states are non-finite,
    or whose predictions are degenerate (metrics.scoring.DegeneratePredictionError), is
    flagged and never selected; any other error propagates.

    Args:
        u: Input sequence, shape (T,).
        targets: Targets, shape (T,) or (T, K).
        train_end: First test row.
        draw: The reservoir draw for this reservoir_seed.
        grid: Lists of spectral_radius, input_scaling and leak_rate values.
        washout: Leading rows dropped from selection, training and scoring.
        ridge_alphas: Harness readout penalties (training.ridge_alphas).
        cv_folds: Number of validation blocks, and the readout's inner fold count.
        task: 'stm' (maximise MC), 'narma' (minimise NRMSE) or 'parity' (accuracy).

    Returns:
        The search outcome, with its budget.

    Raises:
        ValueError: If the grid is malformed or there are too few training rows.
        RuntimeError: If every configuration is degenerate.
    """
    from sklearn.model_selection import KFold

    from qrc_thresher.metrics.scoring import DegeneratePredictionError
    from qrc_thresher.readout import fit_ridge_cv_batched

    score_name, score = _score_function(task)
    configs = _grid_configs(grid)
    rows = np.arange(washout, train_end)
    if len(rows) < 2 * cv_folds:
        raise ValueError(f'too few training rows after the washout: {len(rows)}')
    blocks = list(KFold(n_splits=cv_folds).split(rows))
    targets = np.asarray(targets, dtype=np.float64)

    records = []
    for params in configs:
        esn = build_esn(draw, params)
        record = {
            'params': params.__dict__.copy(),
            'score': None,
            'degenerate': False,
            'reason': None,
            'weight_hash': esn.weight_hash(),
        }
        try:
            X = esn.states(u)
            block_scores = []
            for fit_idx, val_idx in blocks:
                # The harness readout (RidgeCV over ridge_alphas, contiguous folds), via its
                # batched implementation: same alpha and predictions within 1e-10
                # (tests/test_gate_g07.py), without GridSearchCV's per-fit overhead.
                fit_rows, val_rows = rows[fit_idx], rows[val_idx]
                y_fit = targets[fit_rows].reshape(len(fit_rows), -1, 1)
                fit = fit_ridge_cv_batched(X[fit_rows], y_fit, ridge_alphas, cv_folds)
                pred = fit.predict(X[val_rows])[:, :, 0].reshape(targets[val_rows].shape)
                block_scores.append(score(pred, targets[val_rows]))
            record['score'] = float(np.mean(block_scores))
        except (DegeneratePredictionError, NonFiniteStatesError) as exc:
            record.update({'degenerate': True, 'reason': str(exc)})
        records.append(record)

    scored = [i for i, r in enumerate(records) if not r['degenerate']]
    if not scored:
        raise RuntimeError('every ESN configuration was degenerate; nothing to select')
    best_i = max(scored, key=lambda i: (records[i]['score'], -i))
    best = configs[best_i]
    logger.info('ESN search: best %s (%s = %.4f)', best, score_name, records[best_i]['score'])
    return ESNSearch(
        best=best,
        best_hash=records[best_i]['weight_hash'],
        configs=records,
        n_configs=len(configs),
        n_validation_evals=len(configs) * len(blocks),
        score_name=score_name,
    )


def fit_predict_esn(
    u: np.ndarray,
    targets: np.ndarray,
    train_end: int,
    draw: ReservoirDraw,
    params: ESNParams,
    washout: int,
    ridge_alphas: Sequence[float],
    cv_folds: int,
) -> Tuple[np.ndarray, ESN, object]:
    """Deploy one ESN: fit the harness readout on [washout, train_end), predict the test rows.

    Returns:
        (predictions for rows [train_end, T), the deployed ESN, the fitted readout).
    """
    from qrc_thresher.readout import fit_ridge_cv

    esn = build_esn(draw, params)
    X = esn.states(u)
    targets = np.asarray(targets, dtype=np.float64)
    model = fit_ridge_cv(X[washout:train_end], targets[washout:train_end], ridge_alphas, cv_folds)
    return model.predict(X[train_end:]), esn, model


# Names kept for the plugin registry and pyproject entry points.
grid_search_esn = tune_esn
train_predict_esn = fit_predict_esn


def _n_features(n_qubits: int, readout: str) -> int:
    """Number of QRC features, which sets N_ESN (never 2^n_qubits)."""
    if readout == 'z_only':
        return n_qubits
    return n_qubits + n_qubits * (n_qubits - 1) // 2


def _grid_configs(grid: Dict[str, Sequence[float]]) -> List[ESNParams]:
    unknown = set(grid) - set(GRID_KEYS)
    missing = [k for k in GRID_KEYS if not grid.get(k)]
    if unknown or missing:
        raise ValueError(
            f'ESN grid needs non-empty {list(GRID_KEYS)}; unknown {sorted(unknown)}, '
            f'missing {missing}'
        )
    return [
        ESNParams(spectral_radius=float(s), input_scaling=float(i), leak_rate=float(a))
        for s, i, a in product(grid['spectral_radius'], grid['input_scaling'], grid['leak_rate'])
    ]


def _score_function(task: str):
    from qrc_thresher.metrics.scoring import classification_accuracy, memory_capacity, nrmse

    if task == 'stm':
        return 'mc', lambda pred, y: memory_capacity(pred, y)
    if task == 'narma':
        return 'neg_nrmse', lambda pred, y: -nrmse(np.ravel(pred), np.ravel(y))
    if task == 'parity':
        return 'accuracy', lambda pred, y: classification_accuracy(np.ravel(pred), np.ravel(y))
    raise ValueError(f'unknown task for ESN tuning: {task!r}')


def _check_weights(W: np.ndarray, which: str) -> None:
    if not np.all(np.isfinite(W)):
        raise ValueError(f'{which} weights have non-finite entries')
    if W.size and float(np.max(np.abs(W))) > MAX_ABS_WEIGHT:
        raise ValueError(f'{which} weights exceed {MAX_ABS_WEIGHT:g} in magnitude')
