"""G0.7 memory sanity gate, pre-registered version 1 (option A).

The protocol, thresholds and seeds are fixed in ``configs/gates/G0.7.v1.yaml``
(docs/DECISIONS.md D005, D008). This module evaluates that protocol and never chooses a
threshold of its own.

A model enters as a feature map ``feature_map(u, reservoir_seed) -> X`` with X of shape
(T, F). Every seed pair (task_seed, reservoir_seed) is tested on two clauses:

- STM: S = sum over k = 1..K of r2_k, the held-out squared correlation between the readout's
  prediction and u_{t-k}, against a null that permutes the target rows (k = 0 is reported,
  never counted as memory);
- parity: held-out window-2 parity accuracy, against a shuffled-label null.

The readout is the harness protocol (RidgeCV, the registered alphas, contiguous folds) and
is refit on every permutation. A clause passes only if every seed has p <= its significance
level, with at least the registered number of seeds; G0.7 passes only if both clauses pass.
Results carry the measurement model's label, e.g. "exact (oracle upper bound)".
"""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field
from sklearn.linear_model import RidgeCV

from qrc_thresher.config import AlphaLiteConfig, measurement_label
from qrc_thresher.readout import fit_ridge_cv_batched
from qrc_thresher.tasks.stm import generate_stm
from qrc_thresher.tasks.temporal_parity import generate_parity

FeatureMap = Callable[[np.ndarray, int], np.ndarray]

GATE_NAME = 'G0.7'
PROTOCOL_FILE = Path('configs') / 'gates' / 'G0.7.v1.yaml'
_REPO_ROOT = Path(__file__).resolve().parents[3]

# The batched null must reproduce the harness readout on the observed targets.
_BATCHED_READOUT_ATOL = 1e-10


class _Spec(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)


class MeasurementSpec(_Spec):
    model: Literal['exact']
    label: str


class SeedSpec(_Spec):
    source: Literal['experiment_config']
    min_seeds: int = Field(ge=1)
    rule: Literal['every_seed']


class ReadoutSpec(_Spec):
    method: Literal['ridge_cv']
    alpha_selection: Literal['joint']
    ridge_alphas: List[float]
    cv_folds: int = Field(ge=2)
    cv_scheme: Literal['contiguous']
    fit_intercept: Literal[True]


class PermutationNullSpec(_Spec):
    n_permutations: int = Field(ge=1)
    permute: Literal['train_and_test_independently']
    refit_readout: Literal[True]
    rng_seed: int = Field(ge=0)
    draw_order: Literal['train_then_test']
    p_value: Literal['one_plus_count_ge_over_n_plus_one']


class STMClauseSpec(_Spec):
    clause_index: Literal[0]
    task: Literal['stm']
    length: int
    train_frac: float
    delay_max: int = Field(ge=1)
    washout: int = Field(ge=0)
    statistic: Literal['sum_r2_k1_to_delay_max']
    significance_level: float = Field(gt=0, lt=1)


class ParityClauseSpec(_Spec):
    clause_index: Literal[1]
    task: Literal['parity']
    window: int = Field(ge=1)
    length: int
    train_frac: float
    washout: int = Field(ge=0)
    statistic: Literal['heldout_accuracy']
    decision_threshold: float
    significance_level: float = Field(gt=0, lt=1)


class DegenerateSpec(_Spec):
    std_tolerance: float = Field(ge=0)


class ReportingSpec(_Spec):
    null_quantile: float = Field(gt=0, lt=1)
    figure: Literal['forgetting_curve']


class G07Protocol(_Spec):
    """The G0.7 v1 pre-registration, validated field by field."""

    gate: Literal['G0.7']
    version: Literal[1]
    option: Literal['A']
    decision_records: List[str]
    measurement: MeasurementSpec
    seeds: SeedSpec
    readout: ReadoutSpec
    permutation_null: PermutationNullSpec
    stm: STMClauseSpec
    parity: ParityClauseSpec
    degenerate: DegenerateSpec
    reporting: ReportingSpec
    source_path: str = ''
    sha256: str = ''


def protocol_sha256(raw: dict) -> str:
    """SHA-256 of the canonical JSON of a parsed protocol file (sorted keys)."""
    canonical = json.dumps(raw, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def load_protocol(path: Optional[Path] = None) -> G07Protocol:
    """Load and validate the registered G0.7 protocol.

    Args:
        path: Protocol file. Defaults to configs/gates/G0.7.v1.yaml, looked up under the
            working directory and then under the repository root.

    Returns:
        The validated protocol, with its source path and content hash attached.

    Raises:
        FileNotFoundError: If the protocol file does not exist.
        pydantic.ValidationError: If the file does not match the v1 schema.
    """
    if path is None:
        candidates = [PROTOCOL_FILE, _REPO_ROOT / PROTOCOL_FILE]
        path = next((p for p in candidates if p.exists()), candidates[0])
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'G0.7 protocol file not found: {path}')
    raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    return G07Protocol.model_validate(
        {**raw, 'source_path': _display_path(path), 'sha256': protocol_sha256(raw)}
    )


def _display_path(path: Path) -> str:
    """Repository-relative path when possible, so reports carry no local paths."""
    try:
        return path.resolve().relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def seed_pairs_from_config(cfg: AlphaLiteConfig) -> List[Tuple[int, int]]:
    """The config's seed pairs: (task_seed + i, reservoir_seed + i), i = 0..n_seeds-1."""
    return [
        (cfg.seeds.task_seed + i, cfg.seeds.reservoir_seed + i) for i in range(cfg.seeds.n_seeds)
    ]


def check_config(cfg: AlphaLiteConfig, protocol: G07Protocol) -> None:
    """Refuse an experiment config whose readout or measurement differs from the protocol.

    Raises:
        ValueError: If training.ridge_alphas, training.cv_folds or measurement.model
            differ from the registered values.
    """
    registered = [float(a) for a in protocol.readout.ridge_alphas]
    configured = [float(a) for a in cfg.training.ridge_alphas]
    if configured != registered:
        raise ValueError(
            f'G0.7 v1 registers training.ridge_alphas={registered}; the config has {configured}.'
        )
    if cfg.training.cv_folds != protocol.readout.cv_folds:
        raise ValueError(
            f'G0.7 v1 registers training.cv_folds={protocol.readout.cv_folds}; '
            f'the config has {cfg.training.cv_folds}.'
        )
    if cfg.measurement.model != protocol.measurement.model:
        raise ValueError(
            f'G0.7 v1 registers measurement.model={protocol.measurement.model!r}; '
            f'the config has {cfg.measurement.model!r}.'
        )


def qrc_feature_map(cfg: AlphaLiteConfig, ablation: Optional[str] = None) -> FeatureMap:
    """Feature map of the configured reservoir (or its matched ablation), built as the harness
    builds it: through reservoir_from_config, which reads reservoir.window (D010)."""
    from qrc_thresher.reservoirs.windowed_qrc import reservoir_from_config

    def feature_map(u: np.ndarray, reservoir_seed: int) -> np.ndarray:
        reservoir = reservoir_from_config(cfg, reservoir_seed, ablation=ablation)
        return reservoir.features(np.asarray(u, dtype=np.float64))

    return feature_map


MODELS = ('pennylane_qrc', 'no_entangle', 'tuned_qrc', 'esn_linear', 'esn_nonlinear')
_DESIGN_KEYS = ('depth', 'window', 'encoding_scale', 'circuit_hash')


def tuned_feature_map(cfg: AlphaLiteConfig, tuning_record: dict) -> Tuple[FeatureMap, dict]:
    """design_STM(pair) of a tuning record as a G0.7 feature map (D011; PI ruling 4).

    Every seed pair of ``cfg`` must have an entry in the record's ``qrc`` section, and the
    record must describe this config's reservoir (n_qubits, readout, backend).

    Returns:
        (feature_map, details) where details carry the designs per pair, the tuning config
        hash and the sweep id the family evaluator verifies against the rows.

    Raises:
        ValueError: For a missing pair (named by its task seed) or another reservoir.
    """
    from qrc_thresher.tuning import tuned_reservoir

    reservoir = tuning_record.get('reservoir') or {}
    ours = {'n_qubits': cfg.reservoir.n_qubits, 'readout': cfg.reservoir.readout,
            'backend': cfg.reservoir.backend}
    mismatched = {k: (reservoir.get(k), v) for k, v in ours.items() if reservoir.get(k) != v}
    if mismatched:
        raise ValueError(
            'the tuning record describes another reservoir: '
            + ', '.join(f'{k} {have!r} vs config {want!r}'
                        for k, (have, want) in mismatched.items())
        )
    entries = tuning_record.get('qrc') or {}
    designs: Dict[int, dict] = {}
    for task_seed, reservoir_seed in seed_pairs_from_config(cfg):
        entry = entries.get(f'{task_seed}/{reservoir_seed}')
        if entry is None:
            raise ValueError(
                f'the tuning record has no design_STM for seed pair {task_seed}/{reservoir_seed}'
            )
        designs[int(reservoir_seed)] = entry
    reservoirs = {seed: tuned_reservoir(cfg, entry, seed) for seed, entry in designs.items()}

    def feature_map(u: np.ndarray, reservoir_seed: int) -> np.ndarray:
        return reservoirs[int(reservoir_seed)].features(np.asarray(u, dtype=np.float64))

    details = {
        'reservoir': 'pennylane_qrc',
        'kind': 'quantum',
        'design': 'tuned',
        'backend': cfg.reservoir.backend,
        'n_qubits': cfg.reservoir.n_qubits,
        'readout': cfg.reservoir.readout,
        'ablation': None,
        'tuning_config_hash': tuning_record.get('config_hash'),
        'sweep_id': tuning_record.get('sweep_id'),
        'tuning_record_sha': tuning_record.get('record_sha256'),
        'tuning_task': tuning_record.get('task'),
        'designs': {key: {k: entry[k] for k in _DESIGN_KEYS} for key, entry in entries.items()
                    if int(key.split('/')[1]) in designs},
    }
    return feature_map, details


def evaluate_config(
    cfg: AlphaLiteConfig,
    protocol: Optional[G07Protocol] = None,
    model: str = 'pennylane_qrc',
    tuning_record: Optional[dict] = None,
) -> dict:
    """Evaluate G0.7 on the seed pairs of an experiment config.

    Args:
        cfg: Experiment config (seed pairs, readout, reservoir size).
        protocol: Registered protocol (default: load_protocol()).
        model: 'pennylane_qrc' for the configured quantum reservoir, 'no_entangle' for its
            matched no-entangle ablation (D010), 'tuned_qrc' for design_STM(pair) of
            ``tuning_record`` (D011), or an ESN preset ('esn_linear', 'esn_nonlinear') with N
            matched to the QRC feature count.
        tuning_record: The STM tuning record (``tuning.load_record``); required by 'tuned_qrc'.

    Raises:
        ValueError: For an unknown model, a config the protocol refuses, or a tuned model
            without a usable record.
    """
    if model not in MODELS:
        raise ValueError(f'unknown G0.7 model {model!r}; choose from {list(MODELS)}')
    protocol = protocol or load_protocol()
    check_config(cfg, protocol)
    if model == 'tuned_qrc':
        if tuning_record is None:
            raise ValueError("model 'tuned_qrc' needs the STM tuning_record (run `tune stm`)")
        feature_map, details = tuned_feature_map(cfg, tuning_record)
    elif model in ('pennylane_qrc', 'no_entangle'):
        ablation = None if model == 'pennylane_qrc' else model
        feature_map = qrc_feature_map(cfg, ablation=ablation)
        details = {
            'reservoir': 'pennylane_qrc',
            'kind': 'quantum',
            'backend': cfg.reservoir.backend,
            'n_qubits': cfg.reservoir.n_qubits,
            'depth': cfg.reservoir.depth,
            'readout': cfg.reservoir.readout,
            'window': cfg.reservoir.window,
            'encoding_scale': cfg.reservoir.encoding_scale,
            'ablation': ablation,
        }
    else:
        from qrc_thresher.baselines import esn

        params = esn.ESN_PRESETS[model]
        n_units = esn._n_features(cfg.reservoir.n_qubits, cfg.reservoir.readout)
        feature_map = esn.esn_feature_map(n_units, params)
        details = {
            'reservoir': 'esn',
            'kind': 'classical',
            'n_units': n_units,
            'input_connectivity': esn.INPUT_CONNECTIVITY,
            'recurrent_connectivity': esn.RECURRENT_CONNECTIVITY,
            **params.__dict__,
        }
    details['experiment_name'] = cfg.experiment_name
    return evaluate(
        feature_map,
        protocol,
        seed_pairs_from_config(cfg),
        model_name=model,
        model_details=details,
    )


def evaluate(
    feature_map: FeatureMap,
    protocol: G07Protocol,
    seed_pairs: Sequence[Tuple[int, int]],
    model_name: str = '',
    model_details: Optional[dict] = None,
    measurement_model: Optional[str] = None,
) -> dict:
    """Evaluate G0.7 for one feature map.

    Args:
        feature_map: Maps an input sequence u of shape (T,) and a reservoir seed to
            features of shape (T, F).
        protocol: The registered protocol (see load_protocol).
        seed_pairs: Distinct (task_seed, reservoir_seed) pairs.
        model_name: Name recorded in the result and used in report file names.
        model_details: Optional description of the model, recorded as is.
        measurement_model: Measurement model the features were computed under.
            Defaults to the protocol's, and must equal it.

    Returns:
        A JSON-ready dict with the verdict, both clauses and every seed's evidence.

    Raises:
        ValueError: On duplicate seed pairs, a measurement model the protocol does not
            register, or a feature map that returns the wrong shape.
    """
    measurement_model = measurement_model or protocol.measurement.model
    if measurement_model != protocol.measurement.model:
        raise ValueError(
            f'G0.7 v1 registers measurement model {protocol.measurement.model!r}, '
            f'not {measurement_model!r}.'
        )
    pairs = [(int(t), int(r)) for t, r in seed_pairs]
    if len(set(pairs)) != len(pairs):
        raise ValueError(f'seed pairs must be distinct; got {pairs}')

    stm_seeds = [_evaluate_stm_seed(feature_map, protocol, t, r) for t, r in pairs]
    parity_seeds = [_evaluate_parity_seed(feature_map, protocol, t, r) for t, r in pairs]
    enough = len(pairs) >= protocol.seeds.min_seeds
    stm = _clause(stm_seeds, protocol.stm.significance_level, enough, 'sum_r2_k1_to_delay_max')
    parity = _clause(parity_seeds, protocol.parity.significance_level, enough, 'heldout_accuracy')

    if not enough:
        verdict = 'INSUFFICIENT_EVIDENCE'
        message = (
            f'{len(pairs)} seed pair(s); G0.7 v1 needs at least {protocol.seeds.min_seeds}.'
        )
    else:
        verdict = 'PASS' if stm['result'] == parity['result'] == 'PASS' else 'FAIL'
        message = (
            f"STM clause {stm['result']} ({stm['n_passed']} of {len(pairs)} seeds pass); "
            f"parity clause {parity['result']} ({parity['n_passed']} of {len(pairs)} seeds pass)."
        )

    return {
        'gate': GATE_NAME,
        'result': verdict,
        'message': message,
        'model': model_name,
        'model_details': dict(model_details or {}),
        'measurement_model': measurement_model,
        'measurement_label': measurement_label(measurement_model),
        'protocol_version': protocol.version,
        'protocol_option': protocol.option,
        'protocol_path': protocol.source_path,
        'protocol_sha256': protocol.sha256,
        'n_seeds': len(pairs),
        'min_seeds': protocol.seeds.min_seeds,
        'seed_pairs': [list(p) for p in pairs],
        'n_permutations': protocol.permutation_null.n_permutations,
        'clauses': {'stm': stm, 'parity': parity},
        'environment': _environment(),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    }


def write_report(result: dict, out_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Write a new gate JSON and forgetting-curve figure; never overwrite earlier ones.

    Args:
        result: Output of evaluate().
        out_dir: Directory for the report. Defaults to results/gates.

    Returns:
        {'json': path, 'figure': path}.
    """
    import matplotlib.pyplot as plt

    from qrc_thresher.viz.plots import plot_forgetting_curve

    out_dir = Path(out_dir) if out_dir is not None else Path('results') / 'gates'
    out_dir.mkdir(parents=True, exist_ok=True)
    model = re.sub(r'[^A-Za-z0-9_-]+', '_', str(result.get('model') or 'model'))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    base = f'{GATE_NAME}.{model}.{stamp}'
    stem, n = base, 1
    while (out_dir / f'{stem}.json').exists() or (
        out_dir / f'{stem}.forgetting_curve.png'
    ).exists():
        stem, n = f'{base}.{n}', n + 1
    json_path = out_dir / f'{stem}.json'
    figure_path = out_dir / f'{stem}.forgetting_curve.png'

    fig = plot_forgetting_curve(result)
    try:
        fig.savefig(figure_path, dpi=150)
    finally:
        plt.close(fig)

    payload = copy.deepcopy(result)
    payload['figure'] = figure_path.name
    with json_path.open('x', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    return {'json': json_path, 'figure': figure_path}


def _evaluate_stm_seed(
    feature_map: FeatureMap, protocol: G07Protocol, task_seed: int, reservoir_seed: int
) -> dict:
    spec = protocol.stm
    ds = generate_stm(
        length=spec.length,
        delay_max=spec.delay_max,
        train_frac=spec.train_frac,
        rng=np.random.default_rng(task_seed),
    )
    train, test = slice(spec.washout, ds.train_end), slice(ds.train_end, spec.length)
    entry = _seed_entry(task_seed, reservoir_seed, train, test, spec.length)
    empty = {
        'ridge_alpha': None,
        'r2': None,
        'r2_k0': None,
        'S': None,
        'p': None,
        'null_r2_q95': None,
        'null_S': None,
        'null_S_mean': None,
        'null_S_q95': None,
    }
    tol = protocol.degenerate.std_tolerance

    X = _features(feature_map, ds.u, reservoir_seed, spec.length)
    entry['n_features'] = int(X.shape[1])
    reason = _feature_degeneracy(X[train], tol)
    if reason:
        return _degenerate(entry, empty, reason)

    Y_train, Y_test = ds.targets[train], ds.targets[test]
    observed = _harness_readout(protocol, X[train], Y_train)
    pred = observed.predict(X[test])
    reason = _prediction_degeneracy(pred, tol, 'delay k={}')
    if reason:
        return _degenerate(entry, empty, reason)

    train_orders, test_orders = _permutations(
        protocol, spec.clause_index, task_seed, reservoir_seed, len(Y_train), len(Y_test)
    )
    Y_sets = np.stack([Y_train[o] for o in [np.arange(len(Y_train))] + train_orders], axis=-1)
    batched = fit_ridge_cv_batched(
        X[train], Y_sets, alphas=protocol.readout.ridge_alphas, cv_folds=protocol.readout.cv_folds
    )
    pred_sets = batched.predict(X[test])
    entry['batched_readout_check'] = _check_batched(batched, observed, pred_sets[:, :, 0], pred)

    r2 = _squared_pearson(pred, Y_test)
    null_targets = np.stack([Y_test[o] for o in test_orders], axis=-1)
    null_r2 = _squared_pearson(pred_sets[:, :, 1:], null_targets)
    if not (np.all(np.isfinite(r2)) and np.all(np.isfinite(null_r2))):
        return _degenerate(entry, empty, 'non-finite r2 in the observed or null fits')

    S = float(r2[1:].sum())
    null_S = null_r2[1:].sum(axis=0)
    p = _p_value(null_S, S)
    q = protocol.reporting.null_quantile
    entry.update(
        {
            'ridge_alpha': float(observed.alpha_),
            'r2': [float(v) for v in r2],
            'r2_k0': float(r2[0]),
            'S': S,
            'p': p,
            'null_r2_q95': [float(v) for v in np.quantile(null_r2, q, axis=1)],
            'null_S': [float(v) for v in null_S],
            'null_S_mean': float(null_S.mean()),
            'null_S_q95': float(np.quantile(null_S, q)),
            'passed': p <= protocol.stm.significance_level,
        }
    )
    return entry


def _evaluate_parity_seed(
    feature_map: FeatureMap, protocol: G07Protocol, task_seed: int, reservoir_seed: int
) -> dict:
    spec = protocol.parity
    ds = generate_parity(
        length=spec.length,
        window=spec.window,
        train_frac=spec.train_frac,
        rng=np.random.default_rng(task_seed),
    )
    train, test = slice(spec.washout, ds.train_end), slice(ds.train_end, spec.length)
    entry = _seed_entry(task_seed, reservoir_seed, train, test, spec.length)
    empty = {
        'ridge_alpha': None,
        'accuracy': None,
        'p': None,
        'null_accuracy': None,
        'null_accuracy_mean': None,
        'null_accuracy_q95': None,
    }
    tol = protocol.degenerate.std_tolerance

    X = _features(feature_map, ds.u.astype(np.float64), reservoir_seed, spec.length)
    entry['n_features'] = int(X.shape[1])
    reason = _feature_degeneracy(X[train], tol)
    if reason:
        return _degenerate(entry, empty, reason)

    y = ds.targets.astype(np.float64)
    y_train, y_test = y[train], y[test]
    observed = _harness_readout(protocol, X[train], y_train)
    pred = observed.predict(X[test])
    reason = _prediction_degeneracy(pred.reshape(-1, 1), tol, 'the parity target')
    if reason:
        return _degenerate(entry, empty, reason)

    train_orders, test_orders = _permutations(
        protocol, spec.clause_index, task_seed, reservoir_seed, len(y_train), len(y_test)
    )
    Y_sets = np.stack(
        [y_train[o][:, None] for o in [np.arange(len(y_train))] + train_orders], axis=-1
    )
    batched = fit_ridge_cv_batched(
        X[train], Y_sets, alphas=protocol.readout.ridge_alphas, cv_folds=protocol.readout.cv_folds
    )
    pred_sets = batched.predict(X[test])[:, 0, :]
    entry['batched_readout_check'] = _check_batched(batched, observed, pred_sets[:, 0], pred)

    threshold = spec.decision_threshold
    accuracy = float(_accuracy(pred, y_test, threshold))
    null_labels = np.stack([y_test[o] for o in test_orders], axis=-1)
    null_acc = _accuracy(pred_sets[:, 1:], null_labels, threshold)
    if not (np.isfinite(accuracy) and np.all(np.isfinite(null_acc))):
        return _degenerate(entry, empty, 'non-finite accuracy in the observed or null fits')

    p = _p_value(null_acc, accuracy)
    entry.update(
        {
            'ridge_alpha': float(observed.alpha_),
            'accuracy': accuracy,
            'p': p,
            'null_accuracy': [float(v) for v in null_acc],
            'null_accuracy_mean': float(null_acc.mean()),
            'null_accuracy_q95': float(np.quantile(null_acc, protocol.reporting.null_quantile)),
            'passed': p <= spec.significance_level,
        }
    )
    return entry


def _harness_readout(protocol: G07Protocol, X_train: np.ndarray, y_train: np.ndarray) -> RidgeCV:
    """The registered readout: RidgeCV over the registered alphas, contiguous folds.

    This is the harness's train_readout, spelled out here so that the frozen protocol does
    not change if that helper does.
    """
    model = RidgeCV(alphas=protocol.readout.ridge_alphas, cv=protocol.readout.cv_folds)
    return model.fit(X_train, y_train)


def _accuracy(pred: np.ndarray, labels: np.ndarray, threshold: float) -> np.ndarray:
    """Accuracy of thresholded predictions against {0, 1} labels, along axis 0."""
    return np.mean((pred >= threshold) == (labels >= 0.5), axis=0)


def _seed_entry(task_seed: int, reservoir_seed: int, train: slice, test: slice, length: int):
    return {
        'task_seed': task_seed,
        'reservoir_seed': reservoir_seed,
        'train_rows': [train.start, train.stop],
        'test_rows': [test.start, length],
        'degenerate': False,
        'degenerate_reason': None,
        'passed': False,
    }


def _degenerate(entry: dict, empty: dict, reason: str) -> dict:
    entry.update(empty)
    entry.update({'degenerate': True, 'degenerate_reason': reason, 'passed': False})
    return entry


def _features(feature_map: FeatureMap, u: np.ndarray, reservoir_seed: int, length: int):
    X = np.asarray(feature_map(u, reservoir_seed), dtype=np.float64)
    if X.ndim != 2 or X.shape[0] != length:
        raise ValueError(f'feature map must return shape ({length}, F); got {X.shape}')
    return X


def _feature_degeneracy(X_train: np.ndarray, tol: float) -> Optional[str]:
    if not np.all(np.isfinite(X_train)):
        return 'non-finite feature values in the training rows'
    if X_train.shape[1] == 0 or np.all(np.std(X_train, axis=0) <= tol):
        return f'constant features: every training feature column has std <= {tol:g}'
    return None


def _prediction_degeneracy(pred: np.ndarray, tol: float, what: str) -> Optional[str]:
    pred = pred.reshape(len(pred), -1)
    if not np.all(np.isfinite(pred)):
        return 'non-finite held-out predictions'
    flat = np.flatnonzero(np.std(pred, axis=0) <= tol)
    if flat.size:
        return f'constant held-out prediction for {what.format(int(flat[0]))} (std <= {tol:g})'
    return None


def _permutations(
    protocol: G07Protocol,
    clause_index: int,
    task_seed: int,
    reservoir_seed: int,
    n_train: int,
    n_test: int,
) -> Tuple[list, list]:
    """Registered permutation stream: for each permutation, the train order then the test."""
    rng = np.random.default_rng(
        [protocol.permutation_null.rng_seed, clause_index, task_seed, reservoir_seed]
    )
    train_orders, test_orders = [], []
    for _ in range(protocol.permutation_null.n_permutations):
        train_orders.append(rng.permutation(n_train))
        test_orders.append(rng.permutation(n_test))
    return train_orders, test_orders


def _squared_pearson(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Squared Pearson correlation along axis 0; NaN where either side is constant."""
    pc = pred - pred.mean(axis=0)
    tc = target - target.mean(axis=0)
    num = (pc * tc).sum(axis=0)
    den = np.sqrt((pc**2).sum(axis=0) * (tc**2).sum(axis=0))
    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.where(den > 0, num / den, np.nan)
    return r**2


def _p_value(null: np.ndarray, observed: float) -> float:
    """Registered permutation p-value: (1 + #{null >= observed}) / (n + 1)."""
    return float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1))


def _check_batched(batched, observed, batched_pred: np.ndarray, observed_pred: np.ndarray):
    """The batched readout's unpermuted set must reproduce the harness RidgeCV fit."""
    diff = float(np.max(np.abs(batched_pred.reshape(observed_pred.shape) - observed_pred)))
    alpha_match = float(batched.alpha_[0]) == float(observed.alpha_)
    if not alpha_match or diff > _BATCHED_READOUT_ATOL:
        raise RuntimeError(
            'batched readout disagrees with RidgeCV on the observed targets '
            f'(alpha {float(batched.alpha_[0])} vs {float(observed.alpha_)}, '
            f'max |prediction difference| {diff:.3g}); the permutation null would not '
            'refit the harness readout.'
        )
    return {'alpha_match': True, 'max_abs_prediction_diff': diff}


def _clause(seeds: List[dict], level: float, enough: bool, statistic: str) -> dict:
    n_passed = sum(1 for s in seeds if s['passed'])
    if not enough:
        result = 'INSUFFICIENT_EVIDENCE'
    else:
        result = 'PASS' if n_passed == len(seeds) else 'FAIL'
    return {
        'result': result,
        'statistic': statistic,
        'significance_level': level,
        'n_seeds': len(seeds),
        'n_passed': n_passed,
        'n_degenerate': sum(1 for s in seeds if s['degenerate']),
        'seeds': seeds,
    }


def _environment() -> dict:
    import sklearn

    try:
        from qrc_thresher.proof.run_manifest import _git_commit_hash

        commit = _git_commit_hash()
    except Exception:
        commit = 'unknown'
    return {
        'device': 'cpu',
        'precision': 'float64',
        'python': platform.python_version(),
        'numpy': np.__version__,
        'scikit-learn': sklearn.__version__,
        'git_commit_hash': commit,
    }
