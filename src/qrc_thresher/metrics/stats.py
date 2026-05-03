"""Statistical analysis: paired t-test, Wilcoxon, bootstrap CIs, effect size."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.random import Generator
from scipy import stats

logger = logging.getLogger(__name__)

_BOOTSTRAP_N_RESAMPLES = 1000
_CI_LEVEL = 0.95


@dataclass(frozen=True, slots=True)
class PairedTestResult:
    """Result of a paired statistical test."""

    mean_diff: float
    std_diff: float
    t_statistic: float
    p_value: float
    cohens_d: float


def paired_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> PairedTestResult:
    """Perform paired t-test and compute Cohen's d.

    Args:
        scores_a: Scores from condition A (e.g., QRC), shape (n,).
        scores_b: Scores from condition B (e.g., ESN), shape (n,).

    Returns:
        PairedTestResult with t-stat, p-value, Cohen's d.

    Raises:
        ValueError: If arrays have different lengths or length < 2.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f'Arrays must have equal length, got {len(scores_a)} and {len(scores_b)}'
        )
    if len(scores_a) < 2:
        raise ValueError('Need at least 2 observations for paired test')

    diff = scores_a - scores_b
    t_stat, p_val = stats.ttest_rel(scores_a, scores_b)
    mean_d = float(np.mean(diff))
    std_d = float(np.std(diff, ddof=1))
    cohens_d = mean_d / std_d if std_d > 0 else 0.0

    result = PairedTestResult(
        mean_diff=mean_d,
        std_diff=std_d,
        t_statistic=float(t_stat),
        p_value=float(p_val),
        cohens_d=cohens_d,
    )
    logger.debug(
        'Paired t-test: mean_diff=%.4f, p=%.4f, d=%.4f',
        mean_d,
        p_val,
        cohens_d,
    )
    return result


def wilcoxon_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> Tuple[float, float]:
    """Perform Wilcoxon signed-rank test.

    Args:
        scores_a: Scores from condition A, shape (n,).
        scores_b: Scores from condition B, shape (n,).

    Returns:
        Tuple of (statistic, p_value).
    """
    stat, p_val = stats.wilcoxon(scores_a, scores_b)
    logger.debug('Wilcoxon: stat=%.4f, p=%.4f', stat, p_val)
    return float(stat), float(p_val)


def bootstrap_ci(
    scores: np.ndarray,
    rng: Generator,
    n_resamples: int = _BOOTSTRAP_N_RESAMPLES,
    ci_level: float = _CI_LEVEL,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for mean.

    Args:
        scores: Observed scores, shape (n,).
        rng: Seeded generator for bootstrap resampling.
        n_resamples: Number of bootstrap resamples.
        ci_level: Confidence level (default 0.95).

    Returns:
        Tuple of (lower_bound, upper_bound) of the CI.

    Raises:
        ValueError: If scores contain non-finite values.
    """
    if not np.isfinite(scores).all():
        raise ValueError('Bootstrap CI: scores contain non-finite values')

    boot_means = np.zeros(n_resamples, dtype=np.float64)
    n = len(scores)
    for i in range(n_resamples):
        sample = rng.choice(scores, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = 1.0 - ci_level
    lower = float(np.percentile(boot_means, 100.0 * alpha / 2))
    upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2)))
    logger.debug('Bootstrap CI (%.0f%%): [%.4f, %.4f]', 100 * ci_level, lower, upper)
    return lower, upper
