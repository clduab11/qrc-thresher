"""The ESN baseline is a working reservoir (defects D9 and D5; docs/DECISIONS.md D009).

- Every unit sees the input, the recurrent matrix is dense, and a draw whose spectral radius
  cannot be scaled is refused.
- One reservoir draw per reservoir_seed; hyperparameters rescale that same draw, and the
  deployed ESN is the validated one (identical weight hashes). Since D011 the draw carries an
  input bias (tests/test_tuning.py).
- Selection uses contiguous validation blocks inside the post-washout training rows only, on
  the memory sum over k >= 1 (D011).
- Known answer: a 4-unit linear ESN reaches total memory capacity >= 0.75 N = 3.0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qrc_thresher.metrics.scoring import memory_capacity
from qrc_thresher.tasks.stm import generate_stm

REPO_ROOT = Path(__file__).parent.parent
SEED_PAIRS = [(42, 137), (43, 138), (44, 139)]
WASHOUT = 50


def _esn():
    import importlib

    return importlib.import_module('qrc_thresher.baselines.esn')


class TestWiring:
    @pytest.mark.parametrize('n_units', [4, 6, 8, 10])
    def test_every_unit_sees_the_input_and_the_recurrence_is_dense(self, n_units) -> None:
        esn = _esn()
        for seed in range(20):
            draw = esn.draw_reservoir(n_units, seed)
            assert draw.Win_unit.shape == (n_units, 1)
            assert np.all(draw.Win_unit != 0.0)
            assert np.all(draw.W_unit != 0.0)

    @pytest.mark.parametrize('n_units', [4, 6, 8, 10])
    def test_scaled_weights_are_finite_with_the_requested_spectral_radius(self, n_units) -> None:
        esn = _esn()
        params = esn.ESNParams(spectral_radius=0.9, input_scaling=0.1, leak_rate=1.0)
        for seed in range(20):
            model = esn.build_esn(esn.draw_reservoir(n_units, seed), params)
            assert np.all(np.isfinite(model.W)) and np.all(np.isfinite(model.Win))
            assert np.max(np.abs(model.W)) <= 1e3
            rho = np.max(np.abs(np.linalg.eigvals(model.W)))
            assert rho == pytest.approx(0.9, abs=1e-12)

    def test_refuses_a_recurrent_matrix_that_cannot_be_scaled(self) -> None:
        esn = _esn()
        nilpotent = np.triu(np.ones((4, 4)), k=1)  # spectral radius exactly 0
        with pytest.raises(ValueError, match='spectral radius'):
            esn.scale_recurrent(nilpotent, 0.9)
        with pytest.raises(ValueError, match='spectral radius'):
            esn.scale_recurrent(np.zeros((4, 4)), 0.9)
        bad = np.eye(4)
        bad[0, 1] = np.nan
        with pytest.raises(ValueError, match='non-finite'):
            esn.scale_recurrent(bad, 0.9)

    def test_states_match_reservoirpy_given_the_same_weights(self) -> None:
        from reservoirpy.nodes import Reservoir

        esn = _esn()
        params = esn.ESNParams(spectral_radius=0.9, input_scaling=0.5, leak_rate=0.3)
        model = esn.build_esn(esn.draw_reservoir(4, 137), params)
        u = np.random.default_rng(42).uniform(-1.0, 1.0, size=200)
        # The input bias b = input_scaling * b_unit enters inside the tanh (D011).
        reference = Reservoir(
            W=model.W, Win=model.Win, bias=model.b, lr=params.leak_rate
        ).run(u.reshape(-1, 1))
        np.testing.assert_allclose(model.states(u), reference, rtol=0, atol=1e-12)


class TestOneDrawPerSeed:
    def test_same_seed_same_draw_and_hyperparameters_rescale_it(self) -> None:
        esn = _esn()
        a, b = esn.draw_reservoir(4, 137), esn.draw_reservoir(4, 137)
        np.testing.assert_array_equal(a.W_unit, b.W_unit)
        np.testing.assert_array_equal(a.Win_unit, b.Win_unit)
        p1 = esn.ESNParams(spectral_radius=0.8, input_scaling=0.1, leak_rate=0.3)
        p2 = esn.ESNParams(spectral_radius=1.0, input_scaling=1.0, leak_rate=1.0)
        m1, m2 = esn.build_esn(a, p1), esn.build_esn(a, p2)
        np.testing.assert_allclose(m1.W / 0.8, m2.W / 1.0, rtol=1e-12)
        np.testing.assert_allclose(m1.Win / 0.1, m2.Win / 1.0, rtol=1e-12)
        assert esn.draw_reservoir(4, 138).W_unit.tobytes() != a.W_unit.tobytes()

    def test_weight_hash_identifies_draw_and_hyperparameters(self) -> None:
        esn = _esn()
        draw = esn.draw_reservoir(4, 137)
        p = esn.ESNParams(spectral_radius=0.9, input_scaling=0.1, leak_rate=1.0)
        assert esn.build_esn(draw, p).weight_hash() == esn.build_esn(draw, p).weight_hash()
        other = esn.ESNParams(spectral_radius=0.9, input_scaling=0.1, leak_rate=0.5)
        assert esn.build_esn(draw, p).weight_hash() != esn.build_esn(draw, other).weight_hash()


class TestTuning:
    GRID = {'spectral_radius': [0.8, 1.0], 'input_scaling': [0.1, 1.0], 'leak_rate': [0.3, 1.0]}

    def _data(self, length: int = 300):
        return generate_stm(length=length, delay_max=5, train_frac=0.7,
                            rng=np.random.default_rng(42))

    def test_deployed_esn_is_the_validated_one(self, default_config) -> None:
        esn = _esn()
        ds = self._data()
        draw = esn.draw_reservoir(4, 137)
        alphas, folds = default_config.training.ridge_alphas, default_config.training.cv_folds
        search = esn.tune_esn(ds.u, ds.targets, ds.train_end, draw, self.GRID, WASHOUT,
                              alphas, folds, task='stm')
        assert search.n_configs == 8
        assert search.n_validation_evals == 8 * folds
        pred, deployed, _ = esn.fit_predict_esn(ds.u, ds.targets, ds.train_end, draw,
                                                search.best, WASHOUT, alphas, folds)
        assert deployed.weight_hash() == search.best_hash
        assert pred.shape == (len(ds.u) - ds.train_end, ds.targets.shape[1])

    def test_selection_never_sees_the_test_rows(self, default_config) -> None:
        esn = _esn()
        ds = self._data()
        draw = esn.draw_reservoir(4, 137)
        alphas, folds = default_config.training.ridge_alphas, default_config.training.cv_folds
        first = esn.tune_esn(ds.u, ds.targets, ds.train_end, draw, self.GRID, WASHOUT,
                             alphas, folds, task='stm')
        scrambled = ds.targets.copy()
        noise_shape = scrambled[ds.train_end:].shape
        scrambled[ds.train_end:] = np.random.default_rng(0).normal(size=noise_shape)
        second = esn.tune_esn(ds.u, scrambled, ds.train_end, draw, self.GRID, WASHOUT,
                              alphas, folds, task='stm')
        assert first.best == second.best
        assert [c['score'] for c in first.configs] == [c['score'] for c in second.configs]

    def test_a_config_with_constant_states_is_flagged_not_scored(self, default_config) -> None:
        esn = _esn()
        ds = self._data()
        draw = esn.draw_reservoir(4, 137)
        grid = {'spectral_radius': [0.9], 'input_scaling': [0.0, 0.1], 'leak_rate': [1.0]}
        search = esn.tune_esn(ds.u, ds.targets, ds.train_end, draw, grid, WASHOUT,
                              default_config.training.ridge_alphas,
                              default_config.training.cv_folds, task='stm')
        dead = search.configs[0]
        assert dead['degenerate'] is True and dead['score'] is None and dead['reason']
        assert search.best.input_scaling == 0.1

    def test_validation_scores_are_the_harness_readout(self, default_config) -> None:
        from sklearn.model_selection import KFold

        from qrc_thresher.readout import fit_ridge_cv

        esn = _esn()
        ds = self._data()
        draw = esn.draw_reservoir(4, 137)
        alphas, folds = default_config.training.ridge_alphas, default_config.training.cv_folds
        grid = {'spectral_radius': [0.9], 'input_scaling': [0.1, 1.0], 'leak_rate': [1.0]}
        search = esn.tune_esn(ds.u, ds.targets, ds.train_end, draw, grid, WASHOUT,
                              alphas, folds, task='stm')
        rows = np.arange(WASHOUT, ds.train_end)
        for record in search.configs:
            X = esn.build_esn(draw, esn.ESNParams(**record['params'])).states(ds.u)
            scores = []
            for fit_idx, val_idx in KFold(n_splits=folds).split(rows):
                fit_rows, val_rows = rows[fit_idx], rows[val_idx]
                model = fit_ridge_cv(X[fit_rows], ds.targets[fit_rows], alphas, folds)
                pred = model.predict(X[val_rows])
                # The selection metric is the memory sum over k >= 1 (D011); k = 0 is excluded.
                scores.append(memory_capacity(pred[:, 1:], ds.targets[val_rows][:, 1:]))
            assert record['score'] == pytest.approx(float(np.mean(scores)), abs=1e-10)
        assert search.score_name == 'stm_memory'

    def test_an_unrelated_error_is_raised_not_flagged(self, default_config, monkeypatch) -> None:
        esn = _esn()
        ds = self._data()
        draw = esn.draw_reservoir(4, 137)

        def broken_states(self, u):
            raise ValueError('a bug, not a degenerate configuration')

        monkeypatch.setattr(esn.ESN, 'states', broken_states)
        with pytest.raises(ValueError, match='a bug'):
            esn.tune_esn(ds.u, ds.targets, ds.train_end, draw, self.GRID, WASHOUT,
                         default_config.training.ridge_alphas,
                         default_config.training.cv_folds, task='stm')


class TestKnownAnswer:
    def test_linear_esn_total_capacity_is_at_least_three_quarters_of_n(
        self, default_config
    ) -> None:
        esn = _esn()
        params = esn.ESN_PRESETS['esn_linear']
        assert (params.spectral_radius, params.input_scaling, params.leak_rate) == (0.9, 0.1, 1.0)
        n_units = 4
        for task_seed, reservoir_seed in SEED_PAIRS:
            ds = generate_stm(length=500, delay_max=20, train_frac=0.7,
                              rng=np.random.default_rng(task_seed))
            pred, _, _ = esn.fit_predict_esn(
                ds.u, ds.targets, ds.train_end, esn.draw_reservoir(n_units, reservoir_seed),
                params, WASHOUT, default_config.training.ridge_alphas,
                default_config.training.cv_folds,
            )
            mc = memory_capacity(pred, ds.targets[ds.train_end:])
            assert mc >= 0.75 * n_units, (task_seed, reservoir_seed, mc)
            assert mc < n_units + 1.0, (task_seed, reservoir_seed, mc)  # ceiling N, with slack


class TestHealthSmoke:
    def test_esn_smoke_check_asserts_learning(self) -> None:
        from qrc_thresher.proof.benchmark_health import check_esn_smoke

        assert check_esn_smoke()['status'] == 'PASS'
        dead = _esn().ESNParams(spectral_radius=0.9, input_scaling=0.0, leak_rate=1.0)
        assert check_esn_smoke(params=dead)['status'] != 'PASS'
