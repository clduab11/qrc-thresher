"""Known-answer tests for gate G0.7, the memory sanity gate (pre-registered v1).

The protocol is pre-registered in configs/gates/G0.7.v1.yaml (docs/DECISIONS.md D005, D008).
These tests pin what the gate must conclude about feature maps whose memory is known:

- a delay line (u_t ... u_{t-3}) remembers three steps, so it passes the STM clause;
- adding the product u_t * u_{t-1} makes window-2 parity linearly decodable, so the parity
  clause passes; the delay line alone cannot solve parity;
- constant features carry no information and must be flagged degenerate, not scored;
- today's default QRC (4 qubits, depth 3, Z readout) encodes only u_t, so it must fail.

The batched permutation readout must reproduce sklearn's RidgeCV: the same alpha, and
predictions within 1e-10.

The gate module is imported inside each test, so that before the gate exists every test
fails on its own instead of the whole file failing at collection.
"""

from __future__ import annotations

import functools
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from qrc_thresher.config import load_config
from qrc_thresher.reservoirs.pennylane_qrc import build_reservoir_params, extract_features
from qrc_thresher.tasks.stm import generate_stm
from qrc_thresher.tasks.temporal_parity import generate_parity

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG = REPO_ROOT / 'configs' / 'alpha_lite.yaml'
SEED_PAIRS = [(42, 137), (43, 138), (44, 139)]
N_PERMUTATIONS = 200
P_FLOOR = 1.0 / (N_PERMUTATIONS + 1)
WASHOUT = 50


def _g07():
    """Import the gate module (absent until G0.7 is implemented)."""
    return importlib.import_module('qrc_thresher.gates.g07')


def _fit_ridge_cv_batched():
    """Import the batched readout (absent until G0.7 is implemented)."""
    return importlib.import_module('qrc_thresher.readout').fit_ridge_cv_batched


def _lagged(u: np.ndarray, k: int) -> np.ndarray:
    """Return u_{t-k}, zero-padded for t < k (those rows fall inside the washout)."""
    out = np.zeros(len(u), dtype=np.float64)
    out[k:] = u[: len(u) - k]
    return out


def delay_line(u: np.ndarray, reservoir_seed: int) -> np.ndarray:
    """Features u_t, u_{t-1}, u_{t-2}, u_{t-3}: exact memory of three steps."""
    u = np.asarray(u, dtype=np.float64)
    return np.column_stack([_lagged(u, k) for k in range(4)])


def delay_line_with_product(u: np.ndarray, reservoir_seed: int) -> np.ndarray:
    """The delay line plus u_t * u_{t-1}. For bits, XOR(a, b) = a + b - 2ab."""
    u = np.asarray(u, dtype=np.float64)
    return np.column_stack([delay_line(u, reservoir_seed), u * _lagged(u, 1)])


def constant_features(u: np.ndarray, reservoir_seed: int) -> np.ndarray:
    """Four constant columns: no information about the input at all."""
    return np.ones((len(u), 4), dtype=np.float64)


_QRC_FEATURES: dict = {}


def default_qrc(u: np.ndarray, reservoir_seed: int) -> np.ndarray:
    """Today's default QRC (configs/alpha_lite.yaml), built the way the harness builds it."""
    key = (reservoir_seed, np.asarray(u).tobytes())
    if key not in _QRC_FEATURES:
        cfg = load_config(DEFAULT_CONFIG)
        params = build_reservoir_params(
            n_qubits=cfg.reservoir.n_qubits,
            depth=cfg.reservoir.depth,
            readout=cfg.reservoir.readout,
            backend=cfg.reservoir.backend,
            rng=np.random.default_rng(reservoir_seed),
        )
        _QRC_FEATURES[key] = extract_features(np.asarray(u, dtype=np.float64), params)
    return _QRC_FEATURES[key]


FEATURE_MAPS = {
    'delay_line': delay_line,
    'delay_line_with_product': delay_line_with_product,
    'constant': constant_features,
    'default_qrc': default_qrc,
}


@functools.lru_cache(maxsize=None)
def _evaluate(name: str) -> dict:
    """Evaluate G0.7 once per feature map; tests only read the result."""
    g07 = _g07()
    return g07.evaluate(FEATURE_MAPS[name], g07.load_protocol(), SEED_PAIRS, model_name=name)


def _reject_non_json(constant: str):
    raise ValueError(f'gate JSON must not contain {constant}')


class TestProtocol:
    def test_loads_the_registered_v1_protocol(self) -> None:
        protocol = _g07().load_protocol()
        assert protocol.version == 1
        assert protocol.stm.length == 500
        assert protocol.stm.train_frac == 0.7
        assert protocol.stm.delay_max == 20
        assert protocol.stm.washout == WASHOUT
        assert protocol.stm.significance_level == 0.05
        assert protocol.parity.window == 2
        assert protocol.parity.washout == WASHOUT
        assert protocol.parity.significance_level == 0.05
        assert protocol.permutation_null.n_permutations == N_PERMUTATIONS
        assert protocol.seeds.min_seeds == 3

    def test_seed_pairs_come_from_the_config(self, default_config) -> None:
        assert _g07().seed_pairs_from_config(default_config) == SEED_PAIRS

    def test_refuses_a_config_whose_readout_differs(self, default_config) -> None:
        g07 = _g07()
        protocol = g07.load_protocol()
        g07.check_config(default_config, protocol)  # the default config matches
        other = default_config.model_copy(deep=True)
        other.training.ridge_alphas = [1.0]
        with pytest.raises(ValueError, match='ridge_alphas'):
            g07.check_config(other, protocol)
        other = default_config.model_copy(deep=True)
        other.training.cv_folds = 3
        with pytest.raises(ValueError, match='cv_folds'):
            g07.check_config(other, protocol)


class TestBatchedReadout:
    """The batched permutation null must refit exactly what sklearn's RidgeCV fits."""

    @staticmethod
    def _row_orders(n: int) -> list:
        rng = np.random.default_rng(0)
        return [np.arange(n)] + [rng.permutation(n) for _ in range(5)]

    def test_matches_ridgecv_on_default_qrc_stm_features(self, default_config) -> None:
        from sklearn.linear_model import RidgeCV

        fit_ridge_cv_batched = _fit_ridge_cv_batched()

        ds = generate_stm(length=500, delay_max=20, train_frac=0.7, rng=np.random.default_rng(42))
        X = default_qrc(ds.u, 137)
        train, test = slice(WASHOUT, ds.train_end), slice(ds.train_end, None)
        Y = ds.targets[train]
        orders = self._row_orders(len(Y))
        alphas = default_config.training.ridge_alphas
        fit = fit_ridge_cv_batched(
            X[train], np.stack([Y[o] for o in orders], axis=-1), alphas=alphas, cv_folds=5
        )
        pred = fit.predict(X[test])
        assert pred.shape == (len(X[test]), Y.shape[1], len(orders))
        for j, order in enumerate(orders):
            ref = RidgeCV(alphas=alphas, cv=5).fit(X[train], Y[order])
            assert fit.alpha_[j] == ref.alpha_
            np.testing.assert_allclose(pred[:, :, j], ref.predict(X[test]), rtol=0, atol=1e-10)

    def test_matches_ridgecv_on_a_single_parity_target(self, default_config) -> None:
        from sklearn.linear_model import RidgeCV

        fit_ridge_cv_batched = _fit_ridge_cv_batched()

        ds = generate_parity(length=500, window=2, train_frac=0.7, rng=np.random.default_rng(42))
        X = delay_line_with_product(ds.u, 137)
        train, test = slice(WASHOUT, ds.train_end), slice(ds.train_end, None)
        y = ds.targets[train].astype(np.float64)
        orders = self._row_orders(len(y))
        alphas = default_config.training.ridge_alphas
        fit = fit_ridge_cv_batched(
            X[train], np.stack([y[o][:, None] for o in orders], axis=-1), alphas=alphas, cv_folds=5
        )
        pred = fit.predict(X[test])
        assert pred.shape == (len(X[test]), 1, len(orders))
        for j, order in enumerate(orders):
            ref = RidgeCV(alphas=alphas, cv=5).fit(X[train], y[order])
            assert fit.alpha_[j] == ref.alpha_
            np.testing.assert_allclose(pred[:, 0, j], ref.predict(X[test]), rtol=0, atol=1e-10)


class TestDelayLine:
    def test_passes_the_stm_clause(self) -> None:
        stm = _evaluate('delay_line')['clauses']['stm']
        assert stm['result'] == 'PASS'
        assert len(stm['seeds']) == len(SEED_PAIRS)
        for seed in stm['seeds']:
            assert seed['degenerate'] is False
            r2 = np.asarray(seed['r2'])
            assert r2.shape == (21,)
            assert seed['r2_k0'] == r2[0]  # k = 0 is reported ...
            assert r2[0] > 0.99
            assert seed['S'] == pytest.approx(r2[1:].sum())  # ... but never counted
            assert np.all(r2[1:4] > 0.99)
            assert r2[4:].sum() < 0.5
            assert 2.9 < seed['S'] < 3.5
            assert seed['p'] == pytest.approx(P_FLOOR)
            assert seed['passed'] is True

    def test_cannot_solve_parity_without_the_product(self) -> None:
        parity = _evaluate('delay_line')['clauses']['parity']
        assert parity['result'] == 'FAIL'
        assert all(seed['degenerate'] is False for seed in parity['seeds'])


class TestDelayLineWithProduct:
    def test_passes_the_parity_clause(self) -> None:
        parity = _evaluate('delay_line_with_product')['clauses']['parity']
        assert parity['result'] == 'PASS'
        for seed in parity['seeds']:
            assert seed['degenerate'] is False
            assert seed['accuracy'] >= 0.99
            assert seed['p'] == pytest.approx(P_FLOOR)
            assert seed['passed'] is True

    def test_passes_the_gate(self) -> None:
        result = _evaluate('delay_line_with_product')
        assert result['clauses']['stm']['result'] == 'PASS'
        assert result['result'] == 'PASS'


class TestConstantFeatures:
    def test_fail_both_clauses_flagged_degenerate_not_scored(self) -> None:
        result = _evaluate('constant')
        assert result['result'] == 'FAIL'
        for name in ('stm', 'parity'):
            clause = result['clauses'][name]
            assert clause['result'] == 'FAIL'
            for seed in clause['seeds']:
                assert seed['degenerate'] is True
                assert seed['degenerate_reason']
                assert seed['p'] is None
                assert seed['passed'] is False
        for seed in result['clauses']['stm']['seeds']:
            assert seed['S'] is None
            assert seed['r2'] is None
        for seed in result['clauses']['parity']['seeds']:
            assert seed['accuracy'] is None


class TestDefaultQRC:
    def test_default_config_is_todays_qrc(self, default_config) -> None:
        res = default_config.reservoir
        assert (res.n_qubits, res.depth, res.readout) == (4, 3, 'z_only')
        assert res.backend == 'default.qubit'
        seeds = default_config.seeds
        assert (seeds.task_seed, seeds.reservoir_seed, seeds.n_seeds) == (42, 137, 3)

    def test_fails_the_gate(self) -> None:
        result = _evaluate('default_qrc')
        assert result['result'] == 'FAIL'
        assert result['clauses']['stm']['result'] == 'FAIL'
        assert result['clauses']['parity']['result'] == 'FAIL'
        # It fails on the evidence, not on a degeneracy flag.
        for clause in result['clauses'].values():
            assert all(seed['degenerate'] is False for seed in clause['seeds'])
        assert any(seed['p'] > 0.05 for seed in result['clauses']['stm']['seeds'])


class TestSeedRule:
    def test_fewer_than_three_seeds_is_insufficient_evidence(self) -> None:
        g07 = _g07()
        result = g07.evaluate(
            delay_line_with_product,
            g07.load_protocol(),
            SEED_PAIRS[:2],
            model_name='delay_line_with_product',
        )
        assert result['result'] == 'INSUFFICIENT_EVIDENCE'


class TestReporting:
    def test_result_carries_the_measurement_label(self) -> None:
        result = _evaluate('delay_line')
        assert result['measurement_model'] == 'exact'
        assert result['measurement_label'] == 'exact (oracle upper bound)'

    def test_null_95th_percentile_is_reported_per_delay(self) -> None:
        for seed in _evaluate('delay_line')['clauses']['stm']['seeds']:
            q95 = np.asarray(seed['null_r2_q95'])
            assert q95.shape == (21,)
            assert np.all((q95 >= 0.0) & (q95 <= 1.0))

    def test_every_evaluation_writes_a_new_json_and_figure(self, tmp_path: Path) -> None:
        g07 = _g07()
        result = _evaluate('delay_line')
        first = g07.write_report(result, tmp_path)
        second = g07.write_report(result, tmp_path)
        assert first['json'] != second['json']
        for paths in (first, second):
            assert paths['json'].name.startswith('G0.7.')
            assert paths['json'].suffix == '.json'
            assert paths['figure'].suffix == '.png'
            assert paths['figure'].stat().st_size > 0
            text = paths['json'].read_text(encoding='utf-8')
            data = json.loads(text, parse_constant=_reject_non_json)
            assert data['gate'] == 'G0.7'
            assert data['result'] == result['result']
            assert data['measurement_label'] == 'exact (oracle upper bound)'
            assert data['figure'] == paths['figure'].name
            assert len(data['protocol_sha256']) == 64

    def test_forgetting_curve_shows_each_seed_and_its_null(self) -> None:
        import matplotlib.text

        from qrc_thresher.viz.plots import plot_forgetting_curve

        fig = plot_forgetting_curve(_evaluate('delay_line'))
        texts = [t.get_text() for t in fig.findobj(matplotlib.text.Text)]
        assert any('exact (oracle)' in t for t in texts)
        ax = fig.axes[0]
        assert len(ax.get_lines()) >= len(SEED_PAIRS)  # one r2_k curve per seed
        assert len(ax.collections) >= len(SEED_PAIRS)  # one shaded null band per seed
