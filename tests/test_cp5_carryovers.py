"""The CP3 carry-overs (CP5 items D.1–D.4; docs/DECISIONS.md D018, "Test dependencies", for the
source-health list).

- D.1 ``window_inputs``: all three copies (windowed_qrc, qiskit_crosscheck, random_features)
  equal the zero-padded definition u_{t-lag} for T from 1 to 10 at w = n = 5; T = 3 is the case
  the slice `out[lag:, j] = u[: T - lag]` gets wrong when T < lag <= 2T - 2.
- D.2 the phase_random path builds one device per features() call, and its features stay bit
  identical to a reference computed the way HEAD does, with a fresh device and QNode per row.
- D.3 one n_features helper; the formula appears in code exactly once in src/ outside the
  independent Qiskit cross-check.
- D.4 source_health no longer requires pytest or reservoirpy (test dependencies).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent
SRC = REPO_ROOT / 'src' / 'qrc_thresher'
FORMULA = re.compile(r'\*\s*\(\s*\w+\s*-\s*1\s*\)\s*//\s*2')


def _zero_padded(u: np.ndarray, lags) -> np.ndarray:
    T = len(u)
    out = np.zeros((T, len(lags)), dtype=np.float64)
    for j, lag in enumerate(lags):
        for t in range(T):
            if t - lag >= 0:
                out[t, j] = u[t - lag]
    return out


class TestWindowInputs:
    @pytest.mark.parametrize('T', range(1, 11))
    def test_the_windowed_reservoir_copy(self, T) -> None:
        from qrc_thresher.reservoirs.windowed_qrc import window_inputs

        u = np.arange(1, T + 1, dtype=np.float64)
        expected = _zero_padded(u, [j % 5 for j in range(5)])
        assert np.array_equal(window_inputs(u, 5, 5), expected), T

    @pytest.mark.parametrize('T', range(1, 11))
    def test_the_qiskit_crosscheck_copy(self, T) -> None:
        from qrc_thresher.reservoirs.qiskit_crosscheck import window_inputs

        u = np.arange(1, T + 1, dtype=np.float64)
        expected = _zero_padded(u, [j % 5 for j in range(5)])
        assert np.array_equal(window_inputs(u, 5, 5), expected), T

    @pytest.mark.parametrize('T', range(1, 11))
    def test_the_random_features_copy(self, T) -> None:
        from qrc_thresher.baselines.random_features import window_inputs

        u = np.arange(1, T + 1, dtype=np.float64)
        expected = _zero_padded(u, list(range(5)))
        assert np.array_equal(window_inputs(u, 5), expected), T


class TestPhaseRandomDevice:
    def _reservoir(self):
        from qrc_thresher.reservoirs import windowed_qrc
        from qrc_thresher.reservoirs.pennylane_qrc import build_reservoir_params

        params = build_reservoir_params(n_qubits=2, depth=2, readout='z_only',
                                        backend='default.qubit',
                                        rng=np.random.default_rng(137))
        return windowed_qrc.WindowedReservoir(params=params, window=2, reservoir_seed=137,
                                              ablation='phase_random', random_phases=True)

    def _reference(self, reservoir, u) -> np.ndarray:
        """HEAD's computation: fresh device and QNode per row through ``_circuit``."""
        from qrc_thresher.reservoirs import windowed_qrc

        u = np.asarray(u, dtype=np.float64)
        inputs = windowed_qrc.window_inputs(u, reservoir.window, reservoir.params.n_qubits)
        thetas, phis = windowed_qrc.phase_random_angles(
            len(u), reservoir.params.depth, reservoir.params.n_qubits, reservoir.reservoir_seed
        )
        rows = []
        for t in range(len(u)):
            circuit = reservoir._circuit(thetas[t], phis[t])
            rows.append(np.array(circuit([float(v) for v in inputs[t]]), dtype=np.float64))
        return np.stack(rows, axis=0)

    def test_one_device_per_features_call(self, monkeypatch) -> None:
        from qrc_thresher.reservoirs import windowed_qrc

        reservoir = self._reservoir()
        u = np.random.default_rng(1).uniform(-1.0, 1.0, size=6)
        calls = []
        original = windowed_qrc.qml.device

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        monkeypatch.setattr(windowed_qrc.qml, 'device', spy)
        reference = self._reference(reservoir, u)  # HEAD's way: one device per row
        assert len(calls) == len(u), f'{len(calls)} devices for {len(u)} reference rows'
        calls.clear()
        features = reservoir.features(u)
        assert len(calls) == 1, f'{len(calls)} devices for {len(u)} steps'
        assert features.shape == (6, 2)
        assert np.array_equal(features, reference)  # exact equality


class TestNFeatures:
    def test_the_helper(self) -> None:
        from qrc_thresher.features import n_features

        assert n_features(4, 'z_only') == 4
        assert n_features(4, 'z_and_zz') == 10
        assert n_features(2, 'z_and_zz') == 3
        with pytest.raises(ValueError, match='readout'):
            n_features(4, 'z_and_x')

    def test_the_old_names_agree_with_the_helper(self) -> None:
        from qrc_thresher.features import n_features

        from qrc_thresher.baselines import esn
        from qrc_thresher.config import load_config
        from qrc_thresher.reservoirs import windowed_qrc

        cfg = load_config(REPO_ROOT / 'configs' / 'alpha_lite.yaml')
        assert windowed_qrc.n_features_from_config(cfg) == n_features(4, 'z_only') == 4
        assert esn._n_features(4, 'z_and_zz') == n_features(4, 'z_and_zz') == 10
        reservoir = windowed_qrc.reservoir_from_config(cfg, 137)
        assert reservoir.n_features == n_features(cfg.reservoir.n_qubits, cfg.reservoir.readout)

    def test_the_formula_appears_in_code_exactly_once(self) -> None:
        hits = []
        for path in sorted(SRC.rglob('*.py')):
            if path.name == 'qiskit_crosscheck.py':
                continue
            for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
                if FORMULA.search(line):
                    hits.append(f'{path.relative_to(REPO_ROOT).as_posix()}:{number}')
        assert len(hits) == 1, hits


class TestSourceHealth:
    def test_test_dependencies_are_not_required_packages(self) -> None:
        from qrc_thresher.proof import source_health

        required = set(source_health._REQUIRED_PACKAGES)
        assert not required & {'pytest', 'reservoirpy', 'ruff'}
        assert {'pennylane', 'qiskit', 'numpy', 'scipy', 'pandas', 'sklearn', 'click'} <= required
