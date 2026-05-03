"""Tests for metric functions: MC, NRMSE, accuracy, bootstrap CI."""

from __future__ import annotations

import numpy as np
import pytest

from qrc_thresher.metrics.scoring import (
    _safe_corrcoef,
    classification_accuracy,
    memory_capacity,
    nrmse,
)
from qrc_thresher.metrics.stats import (
    bca_ci,
    bootstrap_ci,
    holm_bonferroni,
    paired_test,
    power_analysis,
)


class TestMemoryCapacity:
    """Tests for memory_capacity function."""

    def test_perfect_prediction(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, size=(100, 5))
        mc = memory_capacity(y, y)
        assert abs(mc - 5.0) < 1e-10

    def test_zero_prediction(self) -> None:
        rng = np.random.default_rng(0)
        y_true = rng.normal(0, 1, size=(100, 5))
        y_pred = np.zeros_like(y_true)
        mc = memory_capacity(y_pred, y_true)
        assert mc >= 0.0

    def test_shape_1d(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, size=50)
        mc = memory_capacity(y, y)
        assert abs(mc - 1.0) < 1e-10

    def test_nan_raises(self) -> None:
        y = np.array([[1.0, np.nan]])
        with pytest.raises(ValueError, match='non-finite'):
            memory_capacity(y, y)

    def test_inf_raises(self) -> None:
        y = np.array([[np.inf, 1.0]])
        with pytest.raises(ValueError, match='non-finite'):
            memory_capacity(y, y)

    def test_mc_nonnegative(self) -> None:
        rng = np.random.default_rng(42)
        y_pred = rng.normal(0, 1, size=(50, 10))
        y_true = rng.normal(0, 1, size=(50, 10))
        mc = memory_capacity(y_pred, y_true)
        assert mc >= 0.0


class TestNRMSE:
    """Tests for nrmse function."""

    def test_perfect_prediction_zero(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, size=100)
        assert nrmse(y, y) == 0.0

    def test_zero_variance_raises(self) -> None:
        y = np.ones(50)
        with pytest.raises(ValueError, match='zero variance'):
            nrmse(y, y)

    def test_nonnegative(self) -> None:
        rng = np.random.default_rng(0)
        y_pred = rng.normal(0, 1, size=100)
        y_true = rng.normal(0, 1, size=100)
        assert nrmse(y_pred, y_true) >= 0.0

    def test_nan_raises(self) -> None:
        y = np.array([1.0, np.nan, 2.0])
        with pytest.raises(ValueError, match='non-finite'):
            nrmse(y, y)


class TestClassificationAccuracy:
    """Tests for classification_accuracy function."""

    def test_all_correct(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8])
        assert classification_accuracy(y_pred, y_true) == 1.0

    def test_all_wrong(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.9, 0.1, 0.8, 0.2])
        assert classification_accuracy(y_pred, y_true) == 0.0

    def test_range(self) -> None:
        rng = np.random.default_rng(0)
        y_pred = rng.uniform(0, 1, size=100)
        y_true = rng.integers(0, 2, size=100)
        acc = classification_accuracy(y_pred, y_true)
        assert 0.0 <= acc <= 1.0

    def test_nan_raises(self) -> None:
        y_pred = np.array([np.nan, 0.5])
        y_true = np.array([0, 1])
        with pytest.raises(ValueError, match='non-finite'):
            classification_accuracy(y_pred, y_true)


class TestSafeCorrcoef:
    """Tests for _safe_corrcoef helper."""

    def test_constant_a(self) -> None:
        a = np.ones(50)
        b = np.arange(50, dtype=float)
        assert _safe_corrcoef(a, b) == 0.0

    def test_constant_b(self) -> None:
        a = np.arange(50, dtype=float)
        b = np.ones(50)
        assert _safe_corrcoef(a, b) == 0.0

    def test_perfect_correlation(self) -> None:
        x = np.arange(50, dtype=float)
        assert abs(_safe_corrcoef(x, x) - 1.0) < 1e-10


class TestPairedTest:
    """Tests for paired_test function."""

    def test_identical_arrays(self) -> None:
        scores = np.random.default_rng(0).normal(0, 1, size=10)
        result = paired_test(scores, scores)
        assert result.mean_diff == 0.0
        assert result.cohens_d == 0.0

    def test_length_mismatch_raises(self) -> None:
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match='equal length'):
            paired_test(a, b)

    def test_single_element_raises(self) -> None:
        a = np.array([1.0])
        b = np.array([2.0])
        with pytest.raises(ValueError, match='at least 2'):
            paired_test(a, b)

    def test_cohens_d_sign(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(1.0, 0.1, size=20)
        b = rng.normal(0.0, 0.1, size=20)
        result = paired_test(a, b)
        assert result.cohens_d > 0


class TestBootstrapCI:
    """Tests for bootstrap_ci function."""

    def test_ci_contains_mean(self) -> None:
        rng = np.random.default_rng(42)
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lower, upper = bootstrap_ci(scores, rng=rng, n_resamples=500)
        mean = np.mean(scores)
        assert lower <= mean <= upper

    def test_ci_ordering(self) -> None:
        rng = np.random.default_rng(0)
        scores = rng.normal(0, 1, size=50)
        lower, upper = bootstrap_ci(scores, rng=rng)
        assert lower < upper

    def test_nan_raises(self) -> None:
        rng = np.random.default_rng(0)
        scores = np.array([1.0, np.nan, 3.0])
        with pytest.raises(ValueError, match='non-finite'):
            bootstrap_ci(scores, rng=rng)


class TestBcaCI:
    """Tests for bca_ci function."""

    def test_ci_contains_mean(self) -> None:
        rng = np.random.default_rng(42)
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lower, upper = bca_ci(scores, rng=rng, n_resamples=400)
        mean = float(np.mean(scores))
        assert lower <= mean <= upper

    def test_non_finite_raises(self) -> None:
        rng = np.random.default_rng(0)
        scores = np.array([1.0, np.nan, 3.0])
        with pytest.raises(ValueError, match='non-finite'):
            bca_ci(scores, rng=rng)

    def test_requires_minimum_samples(self) -> None:
        rng = np.random.default_rng(0)
        scores = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match='at least 3'):
            bca_ci(scores, rng=rng)


class TestPowerAnalysis:
    """Tests for power_analysis sample-size estimator."""

    def test_positive_result(self) -> None:
        n = power_analysis(effect_size=0.5, alpha=0.05, power=0.8)
        assert n > 0

    def test_larger_effect_requires_fewer_samples(self) -> None:
        n_small = power_analysis(effect_size=0.3, alpha=0.05, power=0.8)
        n_large = power_analysis(effect_size=0.8, alpha=0.05, power=0.8)
        assert n_large < n_small

    def test_invalid_effect_size_raises(self) -> None:
        with pytest.raises(ValueError, match='effect_size'):
            power_analysis(effect_size=0.0)


class TestHolmBonferroni:
    """Tests for holm_bonferroni function."""

    def test_basic_functionality(self) -> None:
        p_values = [0.01, 0.05, 0.10]
        adjusted = holm_bonferroni(p_values)
        assert all(0.0 <= p <= 1.0 for p in adjusted)

    def test_smallest_p_least_corrected(self) -> None:
        p_values = [0.001, 0.01, 0.05, 0.10]
        adjusted = holm_bonferroni(p_values)
        sorted_idx = np.argsort(p_values)
        sorted_adj = np.array(adjusted)[sorted_idx]
        assert sorted_adj[0] <= sorted_adj[1] <= sorted_adj[2] <= sorted_adj[3]

    def test_non_significant_rejected(self) -> None:
        p_values = [0.51, 0.51]
        adjusted = holm_bonferroni(p_values)
        assert adjusted[0] == 1.0

    def test_single_p_value(self) -> None:
        adjusted = holm_bonferroni([0.05])
        assert adjusted[0] == 0.05

    def test_p05_in_family_of_5(self) -> None:
        p_values = [0.01, 0.05, 0.10, 0.20, 0.50]
        adjusted = holm_bonferroni(p_values)
        original_05_idx = p_values.index(0.05)
        assert adjusted[original_05_idx] == 0.20

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match='at least one p-value'):
            holm_bonferroni([])
