"""Matched ablations of the windowed reservoir (defect D12; docs/DECISIONS.md D010).

phase_random, no_entangle and haar inherit the reservoir's per-pair reservoir_seed, readout (ZZ
included), window and re-upload schedule. Only the tested factor changes:

- no_entangle removes the CNOT ring;
- phase_random draws fresh RZ and RX angles at every step, from default_rng([reservoir_seed, 1]);
- haar replaces each layer's RZ, RX and CNOT ring, after the re-upload, with an independent Haar
  unitary drawn from default_rng([reservoir_seed, 2]).

Switching the tested factor back on reproduces the reservoir: bit for bit for no_entangle and
phase_random, and within D010's tolerance for haar when each layer is fed the unitary of its own
RZ, RX and CNOT ring. The ablation hashes follow D010's recipes. RKS (random_features) gets the
reservoir's F and the per-pair stream default_rng([reservoir_seed, 3]); its window input and
bandwidth wait for D13. An unknown ablation name is refused. The ablation command runs every
seed pair of the config and has no --seed option. The builtin plugins phase_random, no_entangle
and haar return the matched ablations; phase_random and haar refuse reservoir_seed=None. The
legacy functions in reservoirs/ablations.py are deprecated, with warnings that name them.

The windowed module is imported inside each test, so that before it exists every test fails on
its own instead of the whole file failing at collection.
"""

from __future__ import annotations

import csv
import dataclasses
import functools
import hashlib
import importlib
import re
from pathlib import Path

import numpy as np
import pennylane as qml
import pytest
import yaml

from qrc_thresher.config import AlphaLiteConfig, load_config
from qrc_thresher.reservoirs.pennylane_qrc import build_reservoir_params, compute_circuit_hash

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG = REPO_ROOT / 'configs' / 'alpha_lite.yaml'
ABLATIONS = ['phase_random', 'no_entangle', 'haar']
COMMAND_NAMES = ['phase_random', 'no_entangle', 'haar', 'random_features']
READOUTS = ['z_only', 'z_and_zz']
SEED_PAIRS = [(42, 137), (43, 138), (44, 139)]
N_QUBITS, DEPTH = 4, 3
U = np.random.default_rng(7).uniform(-1.0, 1.0, size=5)


def _wq():
    """Import the windowed reservoir module (absent until CP3b)."""
    return importlib.import_module('qrc_thresher.reservoirs.windowed_qrc')


def _raw_config() -> dict:
    return yaml.safe_load(DEFAULT_CONFIG.read_text(encoding='utf-8'))


def _config(readout: str = 'z_only', window: int = 1, n_qubits: int = N_QUBITS, depth=DEPTH):
    raw = _raw_config()
    raw['reservoir'].update(
        {'n_qubits': n_qubits, 'depth': depth, 'readout': readout, 'window': window}
    )
    return AlphaLiteConfig.model_validate(raw)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@functools.lru_cache(maxsize=None)
def _reservoir_features(readout: str, window: int) -> np.ndarray:
    return _wq().reservoir_from_config(_config(readout, window), reservoir_seed=137).features(U)


def _layer_unitary(params, d: int) -> np.ndarray:
    """The unitary of layer d's RZ, RX and CNOT ring, computed here with qml.matrix."""
    n = params.n_qubits

    def layer() -> None:
        for i in range(n):
            qml.RZ(params.thetas[d, i], wires=i)
            qml.RX(params.phis[d, i], wires=i)
        for i in range(n):
            qml.CNOT(wires=[i, (i + 1) % n])

    return np.asarray(qml.matrix(layer, wire_order=list(range(n)))())


class TestSwitchBack:
    @pytest.mark.parametrize('readout', READOUTS)
    @pytest.mark.parametrize('window', [1, 2])
    @pytest.mark.parametrize('ablation', ABLATIONS)
    def test_restoring_the_tested_factor_reproduces_the_reservoir(
        self, ablation, window, readout
    ) -> None:
        wq = _wq()
        cfg = _config(readout, window)
        reservoir = wq.reservoir_from_config(cfg, reservoir_seed=137)
        ablated = wq.reservoir_from_config(cfg, reservoir_seed=137, ablation=ablation)
        # Everything but the tested factor is inherited.
        assert ablated.ablation == ablation
        assert (ablated.window, ablated.reservoir_seed) == (window, 137)
        assert np.array_equal(ablated.params.thetas, reservoir.params.thetas)
        assert np.array_equal(ablated.params.phis, reservoir.params.phis)
        assert ablated.params.readout == readout
        expected = _reservoir_features(readout, window)
        features = ablated.features(U)
        assert features.shape == expected.shape  # ZZ included
        assert not np.array_equal(features, expected)  # the ablation changes something

        if ablation == 'no_entangle':
            assert ablated.entangle is False
            restored = dataclasses.replace(ablated, entangle=True)
            assert np.array_equal(restored.features(U), expected)
        elif ablation == 'phase_random':
            assert ablated.random_phases is True
            restored = dataclasses.replace(ablated, random_phases=False)
            assert np.array_equal(restored.features(U), expected)
        else:
            assert len(ablated.layer_unitaries) == DEPTH
            own = tuple(_layer_unitary(reservoir.params, d) for d in range(DEPTH))
            restored = dataclasses.replace(ablated, layer_unitaries=own)
            diff = float(np.max(np.abs(restored.features(U) - expected)))
            assert diff <= wq.HAAR_SWITCH_BACK_TOLERANCE, diff

    def test_the_haar_switch_back_tolerance_is_d010s(self) -> None:
        assert _wq().HAAR_SWITCH_BACK_TOLERANCE == 1e-10


class TestHashes:
    def test_the_ablation_hashes_follow_d010s_recipes(self) -> None:
        # Hashing only: no features are simulated here.
        wq = _wq()
        cfg = _config('z_only', 2)
        params = build_reservoir_params(
            n_qubits=N_QUBITS, depth=DEPTH, readout='z_only', backend='default.qubit',
            rng=np.random.default_rng(137),
        )
        base = _sha(f'{compute_circuit_hash(params)},window=2')
        reservoir_hash = wq.reservoir_from_config(cfg, reservoir_seed=137).circuit_hash
        assert reservoir_hash == base
        unitaries = wq.haar_layer_unitaries(n_qubits=N_QUBITS, depth=DEPTH, reservoir_seed=137)
        unitary_bytes = b''.join(
            np.ascontiguousarray(u, dtype=np.complex128).tobytes() for u in unitaries
        )
        expected = {
            'no_entangle': _sha(f'{base},entangle=False'),
            'phase_random': _sha(f'{base},random_phases=[137,1]'),
            'haar': _sha(f'{base},layer_unitaries={hashlib.sha256(unitary_bytes).hexdigest()}'),
        }
        for name, want in expected.items():
            got = wq.reservoir_from_config(cfg, reservoir_seed=137, ablation=name).circuit_hash
            assert got == want, name
        assert expected['haar'] != reservoir_hash

    @pytest.mark.parametrize('ablation', ABLATIONS)
    def test_restoring_the_factor_restores_the_reservoir_hash(self, ablation) -> None:
        wq = _wq()
        cfg = _config('z_only', 2)
        reservoir = wq.reservoir_from_config(cfg, reservoir_seed=137)
        ablated = wq.reservoir_from_config(cfg, reservoir_seed=137, ablation=ablation)
        assert ablated.circuit_hash != reservoir.circuit_hash
        factor = {
            'no_entangle': {'entangle': True},
            'phase_random': {'random_phases': False},
            'haar': {'layer_unitaries': None},
        }
        restored = dataclasses.replace(ablated, **factor[ablation])
        assert restored.circuit_hash == reservoir.circuit_hash

    def test_an_unknown_ablation_is_refused(self) -> None:
        wq = _wq()
        with pytest.raises(ValueError):
            wq.reservoir_from_config(_config('z_only', 2), 137, ablation='not_an_ablation')


class TestRandomStreams:
    def test_the_tags_are_d010s(self) -> None:
        assert _wq().ABLATION_TAGS == {'phase_random': 1, 'haar': 2, 'random_features': 3}

    def test_phase_random_draws_fresh_angles_per_step_from_its_own_stream(self) -> None:
        wq = _wq()
        T = 5
        thetas, phis = wq.phase_random_angles(
            T, depth=DEPTH, n_qubits=N_QUBITS, reservoir_seed=137
        )
        rng = np.random.default_rng([137, 1])
        for t in range(T):
            assert np.array_equal(thetas[t], rng.uniform(0.0, 2.0 * np.pi, (DEPTH, N_QUBITS)))
            assert np.array_equal(phis[t], rng.uniform(0.0, 2.0 * np.pi, (DEPTH, N_QUBITS)))
        # Today's default_rng(137 + 1) reused pair (43, 138)'s reservoir angles at step 0.
        next_pair = build_reservoir_params(
            n_qubits=N_QUBITS, depth=DEPTH, readout='z_only', backend='default.qubit',
            rng=np.random.default_rng(138),
        )
        assert not np.array_equal(thetas[0], next_pair.thetas)

    def test_phase_random_row_t_is_the_reservoir_with_step_t_angles(self) -> None:
        wq = _wq()
        cfg = _config('z_and_zz', 2)
        reservoir = wq.reservoir_from_config(cfg, reservoir_seed=137)
        ablated = wq.reservoir_from_config(cfg, reservoir_seed=137, ablation='phase_random')
        thetas, phis = wq.phase_random_angles(
            len(U), depth=DEPTH, n_qubits=N_QUBITS, reservoir_seed=137
        )
        features = ablated.features(U)
        for t in range(len(U)):
            params_t = dataclasses.replace(reservoir.params, thetas=thetas[t], phis=phis[t])
            step = dataclasses.replace(reservoir, params=params_t)
            assert np.array_equal(features[t], step.features(U)[t]), t

    def test_haar_draws_one_independent_unitary_per_layer_from_its_own_stream(self) -> None:
        from scipy.stats import unitary_group

        wq = _wq()
        unitaries = wq.haar_layer_unitaries(n_qubits=N_QUBITS, depth=DEPTH, reservoir_seed=137)
        rng = np.random.default_rng([137, 2])
        expected = [unitary_group.rvs(2**N_QUBITS, random_state=rng) for _ in range(DEPTH)]
        assert len(unitaries) == DEPTH
        for got, want in zip(unitaries, expected):
            assert np.array_equal(got, want)
            assert np.allclose(got @ got.conj().T, np.eye(2**N_QUBITS), atol=1e-12)
        assert not np.allclose(unitaries[0], unitaries[1])
        ablated = wq.reservoir_from_config(_config(), reservoir_seed=137, ablation='haar')
        assert all(np.array_equal(a, b) for a, b in zip(ablated.layer_unitaries, unitaries))
        other = wq.haar_layer_unitaries(n_qubits=N_QUBITS, depth=DEPTH, reservoir_seed=138)
        assert not np.allclose(other[0], unitaries[0])

    @pytest.mark.parametrize('readout, n_features', [('z_only', 4), ('z_and_zz', 10)])
    def test_rks_gets_the_reservoir_feature_count_and_a_per_pair_stream(
        self, readout, n_features
    ) -> None:
        from qrc_thresher.baselines.random_features import build_rks_params

        wq = _wq()
        cfg = _config(readout, 2)
        rks = wq.rks_from_config(cfg, reservoir_seed=137)
        expected = build_rks_params(
            n_features=n_features, sigma=1.0, rng=np.random.default_rng([137, 3])
        )
        assert (rks.n_features, rks.sigma) == (n_features, 1.0)  # bandwidth waits for D13
        assert np.array_equal(rks.W, expected.W) and np.array_equal(rks.b, expected.b)
        assert not np.array_equal(wq.rks_from_config(cfg, reservoir_seed=138).W, rks.W)
        with pytest.raises(ValueError):
            wq.reservoir_from_config(cfg, reservoir_seed=137, ablation='random_features')


class TestAblationCommand:
    @pytest.mark.parametrize('name', COMMAND_NAMES)
    def test_writes_one_row_per_seed_pair_with_distinct_draws(
        self, name, tmp_path, monkeypatch
    ) -> None:
        from click.testing import CliRunner

        from qrc_thresher.cli import cli

        wq = _wq()
        cfg_path = _tiny_parity_config(tmp_path)
        monkeypatch.chdir(tmp_path)
        seen = _spy_on_features(wq, monkeypatch)
        result = CliRunner().invoke(cli, ['ablation', name, '--config', str(cfg_path)])
        assert result.exit_code == 0, result.output
        rows = [
            r for r in _rows(Path('results') / 'runs.csv') if r['task_name'] == f'ablation:{name}'
        ]
        assert [(int(r['task_seed']), int(r['reservoir_seed'])) for r in rows] == SEED_PAIRS
        assert all(r['success'] == 'True' for r in rows)
        hashes = [r['circuit_hash'] for r in rows]
        assert len(set(hashes)) == 3
        assert all(re.fullmatch('[0-9a-f]{64}', h) for h in hashes), hashes
        cfg = load_config(cfg_path)
        if name == 'random_features':
            assert seen == []  # RKS never simulates the circuit
            expected = [wq.rks_circuit_hash(wq.rks_from_config(cfg, r)) for _, r in SEED_PAIRS]
        else:
            assert set(seen) == {(2, r, name) for _, r in SEED_PAIRS}
            expected = [
                wq.reservoir_from_config(cfg, r, ablation=name).circuit_hash for _, r in SEED_PAIRS
            ]
        assert hashes == expected

    @pytest.mark.parametrize('name', COMMAND_NAMES)
    def test_has_no_seed_option(self, name, tmp_path, monkeypatch) -> None:
        from click.testing import CliRunner

        from qrc_thresher.cli import cli

        cfg_path = _tiny_parity_config(tmp_path)
        monkeypatch.chdir(tmp_path)
        args = ['ablation', name, '--config', str(cfg_path), '--seed', '1']
        result = CliRunner().invoke(cli, args)
        assert result.exit_code == 2, f'exit code {result.exit_code}: {result.output[-300:]}'
        assert not (tmp_path / 'results' / 'runs.csv').exists()

    def test_random_features_is_not_an_enabled_baseline_until_d13(self) -> None:
        assert load_config(DEFAULT_CONFIG).baseline.enabled == ['esn']


class TestPlugins:
    def test_the_ablation_plugins_return_the_matched_ablations(self) -> None:
        from qrc_thresher.plugins.registry import create_registry_hub

        wq = _wq()
        cfg = _config('z_and_zz', 2, n_qubits=2, depth=1)
        params = wq.reservoir_from_config(cfg, reservoir_seed=137).params
        hub = create_registry_hub(load_builtin=True, load_entry_points_flag=False)
        for name in ABLATIONS:
            expected = wq.reservoir_from_config(cfg, reservoir_seed=137, ablation=name).features(U)
            got = hub.reservoirs.get(name)(U, params, window=2, reservoir_seed=137)
            assert np.array_equal(got, expected), name

    @pytest.mark.parametrize('name', ['phase_random', 'haar'])
    def test_the_seeded_plugins_refuse_a_missing_reservoir_seed(self, name) -> None:
        from qrc_thresher.plugins.registry import create_registry_hub

        params = build_reservoir_params(
            n_qubits=2, depth=1, readout='z_only', backend='default.qubit',
            rng=np.random.default_rng(137),
        )
        hub = create_registry_hub(load_builtin=True, load_entry_points_flag=False)
        with pytest.raises(ValueError, match='reservoir_seed'):
            hub.reservoirs.get(name)(U, params, window=2)

    @pytest.mark.parametrize('name', ABLATIONS)
    def test_the_legacy_ablation_functions_are_deprecated(self, name) -> None:
        from qrc_thresher.reservoirs import ablations

        params = build_reservoir_params(
            n_qubits=2, depth=1, readout='z_only', backend='default.qubit',
            rng=np.random.default_rng(137),
        )
        u = np.array([0.3, -0.2])
        legacy = {
            'phase_random': lambda: ablations.extract_features_phase_random(
                u, params, rng=np.random.default_rng(1)
            ),
            'no_entangle': lambda: ablations.extract_features_no_entangle(u, params),
            'haar': lambda: ablations.extract_features_haar(
                u, 2, 'default.qubit', rng=np.random.default_rng(2)
            ),
        }
        with pytest.warns(DeprecationWarning, match=f'extract_features_{name}'):
            legacy[name]()


def _tiny_parity_config(tmp_path: Path) -> Path:
    """alpha_lite with a 2-qubit, depth-1, w = 2 reservoir (ZZ readout) on 100-step parity."""
    raw = _raw_config()
    raw['experiment_name'] = 'tiny_ablation'
    raw['task'].update({'name': 'parity', 'length': 100, 'parity_window': 2})
    raw['reservoir'].update({'n_qubits': 2, 'depth': 1, 'readout': 'z_and_zz', 'window': 2})
    path = tmp_path / 'configs' / 'tiny_ablation.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw), encoding='utf-8')
    return path


def _rows(path: Path) -> list:
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def _spy_on_features(wq, monkeypatch) -> list:
    """Record (window, reservoir_seed, ablation) of every reservoir whose features are built."""
    seen: list = []
    original = wq.WindowedReservoir.features

    def spy(self, u, *args, **kwargs):
        seen.append((self.window, self.reservoir_seed, self.ablation))
        return original(self, u, *args, **kwargs)

    monkeypatch.setattr(wq.WindowedReservoir, 'features', spy)
    return seen
