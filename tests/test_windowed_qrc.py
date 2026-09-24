"""The windowed reservoir, design (a) (defect D1; docs/DECISIONS.md D003, D010).

With window w, qubit j re-uploads u_{t-(j mod w)} at every layer, using 0 where
t - (j mod w) < 0. These tests pin:

- w = 1, built through the config helper, is today's reservoir bit for bit (np.array_equal),
  for both readouts at three (n, L, seed) triples;
- a test-local PennyLane circuit, written from D010's text alone, matches the reservoir and its
  no-entangle ablation at w = 2 and 4 within 1e-12, and the same comparison rejects a reversed
  qubit order, a re-upload at layer 0 only and two wrong no-entangle branches;
- the exact memory cliff at k = w, which is what fixes D1: row t ignores u_{t-k} for every
  k >= w and every future input, and moves by more than 1e-6 when u_{t-k} changes, for each
  k < w;
- the D010 rules: the window draws no randomness, the circuit hash is compute_circuit_hash at
  w = 1 and includes w otherwise (followed, since D011 / PI ruling 8, by the encoding scale),
  and a window outside 1..n_qubits is refused;
- the per-call row cache is bit for bit the per-step loop;
- every config-driven build goes through the helper: a grep over src/, and the engine and the
  run command honour reservoir.window on the stm, parity and narma tasks.

The windowed module is imported inside each test, so that before it exists every test fails on
its own instead of the whole file failing at collection.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib
from pathlib import Path

import numpy as np
import pennylane as qml
import pytest
import yaml

from qrc_thresher.config import AlphaLiteConfig, load_config
from qrc_thresher.reservoirs.pennylane_qrc import (
    build_reservoir_params,
    compute_circuit_hash,
    extract_features,
)

REPO_ROOT = Path(__file__).parent.parent
SRC = REPO_ROOT / 'src' / 'qrc_thresher'
DEFAULT_CONFIG = REPO_ROOT / 'configs' / 'alpha_lite.yaml'
TRIPLES = [(2, 2, 2026), (4, 3, 137), (5, 4, 7)]  # (n_qubits, depth, reservoir_seed)
READOUTS = ['z_only', 'z_and_zz']
SEED_PAIRS = [(42, 137), (43, 138), (44, 139)]
TASKS = ['stm', 'parity', 'narma']
ROUTING_PAIRS = SEED_PAIRS[:2]  # the tiny routing configs run two seed pairs

# Replacement inputs for the memory-cliff test. None is +-1: RY(-pi) = -RY(pi) is a global
# phase, so a -1 <-> 1 swap changes nothing. A value is used only where it differs from the
# old input, and from the old input's negative, by at least 0.2.
GENERIC_VALUES = (0.3141, -0.5772, 0.6931, -0.1414)
CLIFF_ROW = 5  # the row t under test
CLIFF_LENGTH = 7  # rows 0..6; row 6 is the future input

# Continuous inputs for the reference circuit; at w = 4 rows 0..2 are zero-padded.
REF_INPUT = np.random.default_rng(3).uniform(-1.0, 1.0, size=5)

# Binary inputs in which every pair (u_{t-1}, u_t) occurs.
PATTERN_BITS = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0])

# Only these modules may build reservoir angles directly, besides G0.5 in commands/gate.py.
ALLOWED_BUILDERS = {
    'reservoirs/pennylane_qrc.py',
    'reservoirs/windowed_qrc.py',
    'proof/benchmark_health.py',
}


def _wq():
    """Import the windowed reservoir module (absent until CP3b)."""
    return importlib.import_module('qrc_thresher.reservoirs.windowed_qrc')


def _raw_config() -> dict:
    return yaml.safe_load(DEFAULT_CONFIG.read_text(encoding='utf-8'))


def _config(n_qubits: int = 4, depth: int = 3, readout: str = 'z_only', window=None):
    raw = _raw_config()
    raw['reservoir'].update({'n_qubits': n_qubits, 'depth': depth, 'readout': readout})
    if window is not None:
        raw['reservoir']['window'] = window
    return AlphaLiteConfig.model_validate(raw)


def _params(n_qubits: int, depth: int, readout: str, seed: int):
    return build_reservoir_params(
        n_qubits=n_qubits,
        depth=depth,
        readout=readout,
        backend='default.qubit',
        rng=np.random.default_rng(seed),
    )


def _todays_features(u, n_qubits: int, depth: int, readout: str, seed: int) -> np.ndarray:
    """Today's reservoir, pennylane_qrc.extract_features: the per-step reference loop."""
    params = _params(n_qubits, depth, readout, seed)
    return extract_features(np.asarray(u, dtype=np.float64), params)


def _reference_features(
    u,
    params,
    window: int,
    entangle: bool = True,
    *,
    reversed_order: bool = False,
    first_layer_only: bool = False,
    drop_rotations: bool = False,
) -> np.ndarray:
    """D010's schedule written out directly in PennyLane, without importing windowed_qrc.

    Per layer: RY(pi * x_j) on every qubit, with x_j = u_{t-(j mod w)} and 0 where that index
    is negative; RZ and RX on every qubit; then the CNOT ring j -> (j + 1) mod n, unless
    ``entangle`` is off. The keyword-only switches build wrong circuits for the rejection test.
    """
    n = params.n_qubits
    device = qml.device('default.qubit', wires=n)

    @qml.qnode(device)
    def circuit(x):
        for d in range(params.depth):
            if d == 0 or not first_layer_only:
                for j in range(n):
                    qml.RY(np.pi * x[j], wires=j)
            if not drop_rotations:
                for j in range(n):
                    qml.RZ(params.thetas[d, j], wires=j)
                    qml.RX(params.phis[d, j], wires=j)
            if entangle:
                for j in range(n):
                    qml.CNOT(wires=[j, (j + 1) % n])
        observables = [qml.expval(qml.PauliZ(j)) for j in range(n)]
        if params.readout == 'z_and_zz':
            observables += [
                qml.expval(qml.PauliZ(i) @ qml.PauliZ(j))
                for i in range(n)
                for j in range(i + 1, n)
            ]
        return observables

    rows = []
    for t in range(len(u)):
        lags = [((n - 1 - j) if reversed_order else j) % window for j in range(n)]
        x = [float(u[t - lag]) if t - lag >= 0 else 0.0 for lag in lags]
        rows.append(np.asarray(circuit(x), dtype=np.float64))
    return np.stack(rows)


def _generic_replacement(old: float) -> float:
    for value in GENERIC_VALUES:
        if abs(value - old) >= 0.2 and abs(value + old) >= 0.2:
            return value
    raise AssertionError(f'no generic replacement for {old}')


def _rows(path: Path) -> list:
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


class TestWindowOneIsTodaysReservoir:
    @pytest.mark.parametrize('readout', READOUTS)
    @pytest.mark.parametrize('n_qubits, depth, seed', TRIPLES)
    def test_through_the_helper_bit_for_bit(self, n_qubits, depth, seed, readout) -> None:
        wq = _wq()
        u = np.random.default_rng(0).uniform(-1.0, 1.0, size=8)
        expected = _todays_features(u, n_qubits, depth, readout, seed)
        reservoir = wq.reservoir_from_config(
            _config(n_qubits, depth, readout), reservoir_seed=seed
        )
        assert reservoir.window == 1
        features = reservoir.features(u)
        assert features.dtype == np.float64
        assert np.array_equal(features, expected)

    def test_window_defaults_to_1(self) -> None:
        assert load_config(DEFAULT_CONFIG).reservoir.window == 1
        raw = _raw_config()
        raw['reservoir'].pop('window', None)
        assert AlphaLiteConfig.model_validate(raw).reservoir.window == 1


class TestReferenceCircuit:
    @pytest.mark.parametrize('readout', READOUTS)
    @pytest.mark.parametrize('window', [2, 4])
    @pytest.mark.parametrize('ablation', [None, 'no_entangle'])
    def test_matches_the_schedule_written_out(self, ablation, window, readout) -> None:
        wq = _wq()
        reservoir = wq.reservoir_from_config(
            _config(readout=readout, window=window), reservoir_seed=137, ablation=ablation
        )
        expected = _reference_features(
            REF_INPUT, _params(4, 3, readout, 137), window, entangle=ablation is None
        )
        features = reservoir.features(REF_INPUT)
        assert features.shape == expected.shape
        assert np.allclose(features, expected, rtol=0.0, atol=1e-12)

    def test_the_comparison_rejects_plausible_wrong_circuits(self) -> None:
        params = _params(4, 3, 'z_only', 137)
        right = _reference_features(REF_INPUT, params, 4)
        right_no_entangle = _reference_features(REF_INPUT, params, 4, entangle=False)
        wrong = {
            'reversed qubit order': (
                _reference_features(REF_INPUT, params, 4, reversed_order=True),
                right,
            ),
            're-upload at layer 0 only': (
                _reference_features(REF_INPUT, params, 4, first_layer_only=True),
                right,
            ),
            'no_entangle keeps the CNOT ring': (right, right_no_entangle),
            'no_entangle also drops RZ and RX': (
                _reference_features(REF_INPUT, params, 4, entangle=False, drop_rotations=True),
                right_no_entangle,
            ),
        }
        for label, (candidate, target) in wrong.items():
            assert not np.allclose(candidate, target, rtol=0.0, atol=1e-12), label
            assert float(np.max(np.abs(candidate - target))) > 1e-6, label


class TestMemoryCliff:
    @pytest.mark.parametrize('window', [1, 2, 4])
    def test_row_t_depends_on_exactly_the_last_w_inputs(self, window) -> None:
        wq = _wq()
        reservoir = wq.reservoir_from_config(_config(window=window), reservoir_seed=137)
        u = np.random.default_rng(42).uniform(-1.0, 1.0, size=CLIFF_LENGTH)
        t = CLIFF_ROW
        row = reservoir.features(u)[t]
        for k in range(t + 1):
            changed = u.copy()
            changed[t - k] = _generic_replacement(u[t - k])
            new_row = reservoir.features(changed)[t]
            if k >= window:
                assert np.array_equal(new_row, row), f'w={window}: row {t} moved at k={k}'
            else:
                moved = float(np.max(np.abs(new_row - row)))
                assert moved > 1e-6, f'w={window}: row {t} moved by only {moved:g} at k={k}'
        future = u.copy()
        future[t + 1 :] = [_generic_replacement(x) for x in u[t + 1 :]]
        assert np.array_equal(reservoir.features(future)[t], row)

    def test_qubit_j_reuploads_u_t_minus_j_mod_w_padded_with_zero(self) -> None:
        wq = _wq()
        u = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        expected = {
            1: [[x, x, x, x] for x in u],
            2: [[0.1, 0.0, 0.1, 0.0], [0.2, 0.1, 0.2, 0.1], [0.3, 0.2, 0.3, 0.2],
                [0.4, 0.3, 0.4, 0.3], [0.5, 0.4, 0.5, 0.4]],
            3: [[0.1, 0.0, 0.0, 0.1], [0.2, 0.1, 0.0, 0.2], [0.3, 0.2, 0.1, 0.3],
                [0.4, 0.3, 0.2, 0.4], [0.5, 0.4, 0.3, 0.5]],
            4: [[0.1, 0.0, 0.0, 0.0], [0.2, 0.1, 0.0, 0.0], [0.3, 0.2, 0.1, 0.0],
                [0.4, 0.3, 0.2, 0.1], [0.5, 0.4, 0.3, 0.2]],
        }
        for window, rows in expected.items():
            got = wq.window_inputs(u, window=window, n_qubits=4)
            assert np.array_equal(got, np.array(rows)), window


class TestAnglesHashAndRefusal:
    def test_the_window_draws_no_randomness_and_the_hash_includes_it(self) -> None:
        wq = _wq()
        params = _params(4, 3, 'z_only', 137)
        base = compute_circuit_hash(params)
        hashes = []
        for window in range(1, 5):
            reservoir = wq.reservoir_from_config(_config(window=window), reservoir_seed=137)
            assert np.array_equal(reservoir.params.thetas, params.thetas)
            assert np.array_equal(reservoir.params.phis, params.phis)
            if window == 1:
                expected = base
            else:
                expected = hashlib.sha256(f'{base},window={window}'.encode()).hexdigest()
            # PI ruling 8 (CP4b, D011): the encoding scale (pi here) is in the preimage too.
            expected = hashlib.sha256(
                f'{expected},encoding_scale={float(reservoir.encoding_scale)!r}'.encode()
            ).hexdigest()
            assert reservoir.circuit_hash == expected, window
            hashes.append(reservoir.circuit_hash)
        assert len(set(hashes)) == 4

    @pytest.mark.parametrize('window', [0, 5])
    def test_a_window_outside_1_to_n_qubits_is_refused(self, window) -> None:
        with pytest.raises(ValueError, match='window'):
            _config(n_qubits=4, window=window)
        wq = _wq()
        with pytest.raises(ValueError, match='window'):
            wq.WindowedReservoir(params=_params(4, 3, 'z_only', 137), window=window)


class TestRowCache:
    @pytest.mark.parametrize(
        'window, ablation, readout',
        [
            (1, None, 'z_only'),
            (1, None, 'z_and_zz'),
            (2, None, 'z_only'),
            (2, None, 'z_and_zz'),
            (2, 'no_entangle', 'z_only'),
            (2, 'haar', 'z_only'),
            (2, 'phase_random', 'z_only'),
        ],
    )
    def test_is_bit_for_bit_the_per_step_loop(self, window, ablation, readout) -> None:
        wq = _wq()
        reservoir = wq.reservoir_from_config(
            _config(readout=readout, window=window), reservoir_seed=137, ablation=ablation
        )
        cached = reservoir.features(PATTERN_BITS)
        assert np.array_equal(cached, reservoir.features(PATTERN_BITS, cache=False))
        if window == 1 and ablation is None:
            assert np.array_equal(cached, _todays_features(PATTERN_BITS, 4, 3, readout, 137))


class TestRouting:
    def test_the_windowed_reservoir_is_a_builtin_plugin(self) -> None:
        from qrc_thresher.plugins.registry import create_registry_hub

        wq = _wq()
        hub = create_registry_hub(load_builtin=True, load_entry_points_flag=False)
        assert 'windowed' in hub.reservoirs.available()
        params = _params(2, 1, 'z_only', 137)
        u = np.array([0.2, -0.4, 0.7])
        expected = wq.WindowedReservoir(params=params, window=2).features(u)
        assert np.array_equal(hub.reservoirs.get('windowed')(u, params, window=2), expected)

    def test_only_g05_and_the_health_check_call_build_reservoir_params(self) -> None:
        offenders = []
        for path in sorted(SRC.rglob('*.py')):
            rel = path.relative_to(SRC).as_posix()
            if rel in ALLOWED_BUILDERS:
                continue
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for scope, line in _calls(tree, 'build_reservoir_params'):
                if rel == 'commands/gate.py' and scope is not None and 'g05' in scope.lower():
                    continue
                offenders.append(f'{rel}:{line} (in {scope})')
        assert offenders == [], (
            'config-driven builds must go through reservoir_from_config: ' + ', '.join(offenders)
        )

    @pytest.mark.parametrize('task', TASKS)
    def test_the_engine_honours_the_window(self, task, tmp_path, monkeypatch) -> None:
        from qrc_thresher.engine import ParallelRunner

        wq = _wq()
        cfg_path = _tiny_config_file(tmp_path, task, window=2)
        monkeypatch.chdir(tmp_path)
        seen = _spy_on_features(wq, monkeypatch)
        cfg = load_config(cfg_path)
        manifests = ParallelRunner(config=cfg, max_workers=1).run_seeds(
            task, config_path=cfg_path
        )
        assert [m.success for m in manifests] == [True, True], [
            m.failure_reason for m in manifests
        ]
        assert set(seen) == {(2, r, None) for _, r in ROUTING_PAIRS}
        expected = [wq.reservoir_from_config(cfg, r).circuit_hash for _, r in ROUTING_PAIRS]
        assert [m.circuit_hash for m in manifests] == expected

    @pytest.mark.parametrize('task', TASKS)
    def test_the_run_command_honours_the_window(self, task, tmp_path, monkeypatch) -> None:
        from qrc_thresher.commands.run import run_handler

        wq = _wq()
        cfg_path = _tiny_config_file(tmp_path, task, window=2)
        monkeypatch.chdir(tmp_path)
        seen = _spy_on_features(wq, monkeypatch)
        assert run_handler(task, str(cfg_path)) == 0  # every seed pair, no --seed (D013)
        assert set(seen) == {(2, r, None) for _, r in ROUTING_PAIRS}
        rows = _rows(Path('results') / 'runs.csv')
        cfg = load_config(cfg_path)
        expected = [wq.reservoir_from_config(cfg, r).circuit_hash for _, r in ROUTING_PAIRS]
        assert [r['circuit_hash'] for r in rows] == expected


def _calls(tree: ast.AST, name: str) -> list:
    """(enclosing top-level function or class, line) of every call to ``name``."""
    found = []

    def visit(node: ast.AST, scope) -> None:
        for child in ast.iter_child_nodes(node):
            inner = scope
            if scope is None and isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                inner = child.name
            if isinstance(child, ast.Call):
                func = child.func
                called = func.id if isinstance(func, ast.Name) else getattr(func, 'attr', None)
                if called == name:
                    found.append((inner, child.lineno))
            visit(child, inner)

    visit(tree, None)
    return found


def _tiny_config_file(tmp_path: Path, task: str, window: int) -> Path:
    """alpha_lite with a 2-qubit, depth-1 reservoir, a 100-step task and two seed pairs."""
    raw = _raw_config()
    raw['experiment_name'] = f'tiny_{task}_w{window}'
    raw['task'].update({'name': task, 'length': 100, 'delay_max': 5, 'parity_window': 2})
    raw['reservoir'].update({'n_qubits': 2, 'depth': 1, 'readout': 'z_only', 'window': window})
    raw['seeds']['n_seeds'] = len(ROUTING_PAIRS)
    path = tmp_path / 'configs' / f'tiny_{task}_w{window}.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw), encoding='utf-8')
    return path


def _spy_on_features(wq, monkeypatch) -> list:
    """Record (window, reservoir_seed, ablation) of every reservoir whose features are built."""
    seen: list = []
    original = wq.WindowedReservoir.features

    def spy(self, u, *args, **kwargs):
        seen.append((self.window, self.reservoir_seed, self.ablation))
        return original(self, u, *args, **kwargs)

    monkeypatch.setattr(wq.WindowedReservoir, 'features', spy)
    return seen
