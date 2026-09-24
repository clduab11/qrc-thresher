"""The encoding scale is a registered hyperparameter (docs/DECISIONS.md D011).

- reservoir.encoding_scale (default pi) and the WindowedReservoir field encoding_scale replace
  windowed_qrc's import of pennylane_qrc._ENCODING_SCALE; pennylane_qrc stays frozen at pi.
- w = 1 at pi is today's circuit bit for bit (the CP3 tests keep proving it); at any other scale
  the features change. The hash appends ",encoding_scale=<repr(float(scale))>" after
  ",window=<w>" and before any ablation suffix, for every scale including pi (PI ruling 8, CP4b:
  the preimage includes the scale unconditionally; no literal circuit hash is pinned anywhere).
- The scale multiplies every re-upload: a test-local PennyLane circuit with RY(scale * x) at every
  layer matches the reservoir within 1e-12, and the Qiskit side, given the same scale, within the
  G0.5 tolerance.
- G0.7's model details record the scale.
"""

from __future__ import annotations

import hashlib
import importlib
import math
from pathlib import Path

import numpy as np
import pennylane as qml
import pytest
import yaml
from pydantic import ValidationError

from qrc_thresher.config import AlphaLiteConfig, load_config
from qrc_thresher.reservoirs.pennylane_qrc import (
    _ENCODING_SCALE,
    build_reservoir_params,
    compute_circuit_hash,
    extract_features,
)

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG = REPO_ROOT / 'configs' / 'alpha_lite.yaml'
U = np.random.default_rng(11).uniform(-1.0, 1.0, size=6)
HALF_PI = math.pi / 2


def _wq():
    return importlib.import_module('qrc_thresher.reservoirs.windowed_qrc')


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _params(n: int, depth: int, readout: str = 'z_only', seed: int = 137):
    return build_reservoir_params(n_qubits=n, depth=depth, readout=readout,
                                  backend='default.qubit', rng=np.random.default_rng(seed))


def _config(**reservoir) -> AlphaLiteConfig:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding='utf-8'))
    raw['reservoir'].update(reservoir)
    return AlphaLiteConfig.model_validate(raw)


def _reference(u, params, window: int, scale: float, entangle: bool = True) -> np.ndarray:
    """D010's schedule with RY(scale * x_j) at every layer, written here without windowed_qrc."""
    n, depth = params.n_qubits, params.depth
    dev = qml.device('default.qubit', wires=n)

    @qml.qnode(dev)
    def circuit(x):
        for d in range(depth):
            for j in range(n):
                qml.RY(scale * x[j], wires=j)
            for j in range(n):
                qml.RZ(params.thetas[d, j], wires=j)
                qml.RX(params.phis[d, j], wires=j)
            if entangle:
                for j in range(n):
                    qml.CNOT(wires=[j, (j + 1) % n])
        obs = [qml.expval(qml.PauliZ(j)) for j in range(n)]
        if params.readout == 'z_and_zz':
            obs += [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j))
                    for i in range(n) for j in range(i + 1, n)]
        return obs

    rows = []
    for t in range(len(u)):
        x = [float(u[t - (j % window)]) if t - (j % window) >= 0 else 0.0 for j in range(n)]
        rows.append(np.array(circuit(x), dtype=np.float64))
    return np.stack(rows)


class TestConfigAndField:
    def test_the_config_key_defaults_to_pi_and_refuses_a_non_positive_scale(self) -> None:
        cfg = load_config(DEFAULT_CONFIG)
        assert cfg.reservoir.encoding_scale == math.pi
        raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding='utf-8'))
        assert 'encoding_scale' not in raw['reservoir']  # optional: no existing config needs it
        assert _config(encoding_scale=HALF_PI).reservoir.encoding_scale == HALF_PI
        with pytest.raises(ValidationError):
            _config(encoding_scale=0.0)

    def test_the_reservoir_field_defaults_to_pi_and_comes_from_the_config(self) -> None:
        wq = _wq()
        params = _params(2, 1)
        assert wq.WindowedReservoir(params=params).encoding_scale == math.pi
        assert wq.reservoir_from_config(_config(), 137).encoding_scale == math.pi
        built = wq.reservoir_from_config(_config(encoding_scale=HALF_PI), 137)
        assert built.encoding_scale == HALF_PI
        assert _ENCODING_SCALE == math.pi  # pennylane_qrc stays frozen

    def test_windowed_qrc_no_longer_imports_the_frozen_constant(self) -> None:
        wq = _wq()
        source = Path(wq.__file__).read_text(encoding='utf-8')
        assert '_ENCODING_SCALE' not in source


class TestFeatures:
    def test_w1_at_pi_is_todays_circuit_and_pi_over_2_is_not(self) -> None:
        wq = _wq()
        params = _params(2, 2)
        today = extract_features(U, params)
        at_pi = wq.WindowedReservoir(params=params, window=1, encoding_scale=math.pi).features(U)
        assert np.array_equal(at_pi, today)
        at_half = wq.WindowedReservoir(params=params, window=1, encoding_scale=HALF_PI).features(U)
        assert at_half.shape == today.shape
        assert np.max(np.abs(at_half - today)) > 1e-3

    @pytest.mark.parametrize('n, depth, window', [(2, 2, 1), (3, 2, 2), (4, 1, 4)])
    @pytest.mark.parametrize('readout', ['z_only', 'z_and_zz'])
    @pytest.mark.parametrize('scale', [math.pi / 4, HALF_PI, 3 * math.pi / 4])
    def test_the_scale_multiplies_every_reupload(self, n, depth, window, readout, scale) -> None:
        wq = _wq()
        params = _params(n, depth, readout)
        reservoir = wq.WindowedReservoir(params=params, window=window, encoding_scale=scale)
        expected = _reference(U, params, window, scale)
        np.testing.assert_allclose(reservoir.features(U), expected, rtol=0, atol=1e-12)
        ablated = wq.WindowedReservoir(params=params, window=window, encoding_scale=scale,
                                       ablation='no_entangle', entangle=False)
        np.testing.assert_allclose(ablated.features(U), _reference(U, params, window, scale, False),
                                   rtol=0, atol=1e-12)

    def test_the_qiskit_side_takes_the_scale_and_defaults_to_pi(self) -> None:
        from qrc_thresher.reservoirs.qiskit_crosscheck import CROSSCHECK_TOLERANCE, qiskit_features

        wq = _wq()
        params = _params(2, 2, 'z_and_zz', seed=2026)
        u = U[:4]
        for scale in (math.pi / 4, HALF_PI):
            pl = wq.WindowedReservoir(params=params, window=2, encoding_scale=scale).features(u)
            qk = qiskit_features(u, params.thetas, params.phis, 2, 2, 2, 'z_and_zz',
                                 encoding_scale=scale)
            assert np.max(np.abs(pl - qk)) <= CROSSCHECK_TOLERANCE
        positional = qiskit_features(u, params.thetas, params.phis, 2, 2, 2, 'z_and_zz')
        explicit = qiskit_features(u, params.thetas, params.phis, 2, 2, 2, 'z_and_zz',
                                   encoding_scale=math.pi)
        assert np.array_equal(positional, explicit)


class TestHash:
    """PI ruling 8 (CP4b): the scale is in the preimage unconditionally, after the window."""

    def test_pi_is_in_the_preimage_too(self) -> None:
        wq = _wq()
        params = _params(4, 3)
        base = compute_circuit_hash(params)
        pi_suffix = ',encoding_scale=3.141592653589793'
        w1 = wq.WindowedReservoir(params=params, window=1, encoding_scale=math.pi)
        assert w1.circuit_hash == _sha(f'{base}{pi_suffix}') != base
        w2 = wq.WindowedReservoir(params=params, window=2, encoding_scale=math.pi).circuit_hash
        assert w2 == _sha(f'{_sha(f"{base},window=2")}{pi_suffix}')
        assert wq.WindowedReservoir(params=params, window=2).circuit_hash == w2  # default pi

    def test_another_scale_appends_its_repr_after_the_window(self) -> None:
        wq = _wq()
        params = _params(4, 3)
        base = compute_circuit_hash(params)
        suffix = f',encoding_scale={float(HALF_PI)!r}'
        assert suffix == ',encoding_scale=1.5707963267948966'
        w1 = wq.WindowedReservoir(params=params, window=1, encoding_scale=HALF_PI).circuit_hash
        assert w1 == _sha(f'{base}{suffix}')
        w2 = wq.WindowedReservoir(params=params, window=2, encoding_scale=HALF_PI).circuit_hash
        assert w2 == _sha(f'{_sha(f"{base},window=2")}{suffix}')
        at_pi = wq.WindowedReservoir(params=params, window=1).circuit_hash
        assert len({base, w1, w2, at_pi}) == 4

    def test_the_ablation_suffixes_come_after_the_scale(self) -> None:
        wq = _wq()
        cfg = _config(window=2, encoding_scale=HALF_PI)
        design = wq.reservoir_from_config(cfg, 137).circuit_hash
        no_entangle = wq.reservoir_from_config(cfg, 137, ablation='no_entangle')
        assert no_entangle.encoding_scale == HALF_PI
        assert no_entangle.circuit_hash == _sha(f'{design},entangle=False')
        phase_random = wq.reservoir_from_config(cfg, 137, ablation='phase_random')
        assert phase_random.circuit_hash == _sha(f'{design},random_phases=[137,1]')
        haar = wq.reservoir_from_config(cfg, 137, ablation='haar')
        digest = wq._unitaries_digest(haar.layer_unitaries)
        assert haar.circuit_hash == _sha(f'{design},layer_unitaries={digest}')


class TestG07Details:
    def test_the_model_details_record_the_scale(self, monkeypatch) -> None:
        g07 = importlib.import_module('qrc_thresher.gates.g07')
        captured = {}

        def capture(feature_map, protocol, seed_pairs, **kwargs):
            captured.update(kwargs)
            return {}

        monkeypatch.setattr(g07, 'evaluate', capture)
        g07.evaluate_config(_config(window=2, encoding_scale=HALF_PI), model='pennylane_qrc')
        assert captured['model_details']['encoding_scale'] == HALF_PI
        assert captured['model_details']['window'] == 2
        g07.evaluate_config(load_config(DEFAULT_CONFIG), model='no_entangle')
        assert captured['model_details']['encoding_scale'] == math.pi
