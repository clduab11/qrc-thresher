"""Tests for task generators: STM, parity, NARMA-10."""

from __future__ import annotations

import numpy as np

from qrc_thresher.tasks.narma10 import generate_narma10
from qrc_thresher.tasks.stm import generate_stm
from qrc_thresher.tasks.temporal_parity import generate_parity


class TestSTM:
    """Tests for STM task generator."""

    def test_deterministic_same_seed(self) -> None:
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        ds1 = generate_stm(length=100, delay_max=5, train_frac=0.7, rng=rng1)
        ds2 = generate_stm(length=100, delay_max=5, train_frac=0.7, rng=rng2)
        assert np.array_equal(ds1.u, ds2.u)
        assert np.array_equal(ds1.targets, ds2.targets)

    def test_different_seed_differs(self) -> None:
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(99)
        ds1 = generate_stm(length=100, delay_max=5, train_frac=0.7, rng=rng1)
        ds2 = generate_stm(length=100, delay_max=5, train_frac=0.7, rng=rng2)
        assert not np.array_equal(ds1.u, ds2.u)

    def test_u_shape(self) -> None:
        rng = np.random.default_rng(0)
        ds = generate_stm(length=200, delay_max=10, train_frac=0.8, rng=rng)
        assert ds.u.shape == (200,)

    def test_targets_shape(self) -> None:
        rng = np.random.default_rng(0)
        K = 7
        ds = generate_stm(length=100, delay_max=K, train_frac=0.7, rng=rng)
        assert ds.targets.shape == (100, K + 1)

    def test_u_range(self) -> None:
        rng = np.random.default_rng(0)
        ds = generate_stm(length=500, delay_max=5, train_frac=0.7, rng=rng)
        assert ds.u.min() >= -1.0
        assert ds.u.max() <= 1.0

    def test_train_end(self) -> None:
        rng = np.random.default_rng(0)
        ds = generate_stm(length=100, delay_max=5, train_frac=0.7, rng=rng)
        assert ds.train_end == 70

    def test_delay_k0_equals_u(self) -> None:
        rng = np.random.default_rng(0)
        ds = generate_stm(length=100, delay_max=5, train_frac=0.7, rng=rng)
        assert np.array_equal(ds.targets[:, 0], ds.u)

    def test_delay_k1_shifted(self) -> None:
        rng = np.random.default_rng(0)
        ds = generate_stm(length=100, delay_max=5, train_frac=0.7, rng=rng)
        # y_t^(1) = u_{t-1} for t >= 1
        assert np.array_equal(ds.targets[1:, 1], ds.u[:-1])


class TestParity:
    """Tests for temporal parity task generator."""

    def test_deterministic_same_seed(self) -> None:
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        p1 = generate_parity(length=100, window=3, train_frac=0.7, rng=rng1)
        p2 = generate_parity(length=100, window=3, train_frac=0.7, rng=rng2)
        assert np.array_equal(p1.u, p2.u)
        assert np.array_equal(p1.targets, p2.targets)

    def test_u_binary(self) -> None:
        rng = np.random.default_rng(0)
        p = generate_parity(length=200, window=3, train_frac=0.7, rng=rng)
        assert set(np.unique(p.u)).issubset({0, 1})

    def test_targets_binary(self) -> None:
        rng = np.random.default_rng(0)
        p = generate_parity(length=200, window=3, train_frac=0.7, rng=rng)
        assert set(np.unique(p.targets[p.window - 1 :])).issubset({0, 1})

    def test_targets_shape(self) -> None:
        rng = np.random.default_rng(0)
        p = generate_parity(length=100, window=3, train_frac=0.7, rng=rng)
        assert p.targets.shape == (100,)

    def test_train_end(self) -> None:
        rng = np.random.default_rng(0)
        p = generate_parity(length=100, window=3, train_frac=0.7, rng=rng)
        assert p.train_end == 70

    def test_xor_correctness(self) -> None:
        rng = np.random.default_rng(5)
        window = 3
        p = generate_parity(length=50, window=window, train_frac=0.7, rng=rng)
        for t in range(window - 1, 50):
            expected = int(np.bitwise_xor.reduce(p.u[t - window + 1 : t + 1]))
            assert p.targets[t] == expected


class TestNARMA10:
    """Tests for NARMA-10 task generator."""

    def test_u_range(self) -> None:
        rng = np.random.default_rng(0)
        ds = generate_narma10(length=200, train_frac=0.7, rng=rng)
        assert ds.u.min() >= 0.0
        assert ds.u.max() <= 0.5

    def test_shapes(self) -> None:
        rng = np.random.default_rng(0)
        ds = generate_narma10(length=200, train_frac=0.7, rng=rng)
        assert ds.u.shape == (200,)
        assert ds.targets.shape == (200,)

    def test_train_end(self) -> None:
        rng = np.random.default_rng(0)
        ds = generate_narma10(length=200, train_frac=0.8, rng=rng)
        assert ds.train_end == 160

    def test_deterministic(self) -> None:
        rng1 = np.random.default_rng(7)
        rng2 = np.random.default_rng(7)
        ds1 = generate_narma10(length=100, train_frac=0.7, rng=rng1)
        ds2 = generate_narma10(length=100, train_frac=0.7, rng=rng2)
        assert np.array_equal(ds1.u, ds2.u)
        assert np.allclose(ds1.targets, ds2.targets)

    def test_recurrence_relation(self) -> None:
        from qrc_thresher.tasks.narma10 import verify_narma10_recurrence
        rng = np.random.default_rng(123)
        ds = generate_narma10(length=200, train_frac=0.7, rng=rng)
        y_with_initial = np.zeros(201)
        y_with_initial[1:] = ds.targets
        assert verify_narma10_recurrence(ds.u, y_with_initial)
