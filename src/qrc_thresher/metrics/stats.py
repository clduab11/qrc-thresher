"""Statistical analysis: paired tests, corrections, and uncertainty metrics.

The comparative gates use ``metrics.paired`` (docs/DECISIONS.md D013), which builds on the
functions here: ``bca_ci`` for the bootstrap interval, ``holm_bonferroni`` for the family
adjustment (unchanged since D001) and ``power_analysis`` / ``achieved_power`` for the run count.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.random import Generator
from scipy import stats

logger = logging.getLogger(__name__)

_BOOTSTRAP_N_RESAMPLES = 1000
_CI_LEVEL = 0.95
SIDED = (1, 2)


@dataclass(frozen=True, slots=True)
class PairedTestResult:
    """Result of a paired t-test.

    ``d_z`` is mean(diff) / sd(diff) with sd over n - 1, and None when sd = 0 (D013; it was
    named ``cohens_d`` and written as 0.0 before, defect D20).
    """

    mean_diff: float
    std_diff: float
    t_statistic: float
    p_value: float
    d_z: Optional[float]


def paired_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> PairedTestResult:
    """Perform a two-sided paired t-test and compute d_z.

    Args:
        scores_a: Scores from condition A (e.g., QRC), shape (n,).
        scores_b: Scores from condition B (e.g., ESN), shape (n,).

    Returns:
        PairedTestResult with t-stat, p-value, d_z (None at zero variance).

    Raises:
        ValueError: If arrays have different lengths or length < 2.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f'Arrays must have equal length, got {len(scores_a)} and {len(scores_b)}'
        )
    if len(scores_a) < 2:
        raise ValueError('Need at least 2 observations for paired test')

    diff = np.asarray(scores_a, dtype=np.float64) - np.asarray(scores_b, dtype=np.float64)
    t_stat, p_val = stats.ttest_rel(scores_a, scores_b)
    mean_d = float(np.mean(diff))
    std_d = float(np.std(diff, ddof=1))
    d_z = d_z_effect(diff)

    result = PairedTestResult(
        mean_diff=mean_d,
        std_diff=std_d,
        t_statistic=float(t_stat),
        p_value=float(p_val),
        d_z=d_z,
    )
    logger.debug('Paired t-test: mean_diff=%.4f, p=%.4f, d_z=%s', mean_d, p_val, d_z)
    return result


def d_z_effect(diff: np.ndarray) -> Optional[float]:
    """d_z = mean(diff) / sd(diff) (sd over n - 1); None when sd = 0 (D013)."""
    diff = np.asarray(diff, dtype=np.float64)
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    if sd == 0.0:
        return None
    return float(np.mean(diff) / sd)


def wilcoxon_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> Tuple[float, float]:
    """Perform a two-sided Wilcoxon signed-rank test.

    The comparative gates use the one-sided form in ``metrics.paired`` (D013).

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
    """Compute a percentile bootstrap confidence interval for the mean.

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


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni correction to a list of p-values.

    Args:
        p_values: List of uncorrected p-values from multiple comparisons.

    Returns:
        List of Holm-Bonferroni adjusted p-values, in same order as input.

    Raises:
        ValueError: If any p-value is outside [0, 1] or list is empty.
    """
    if not p_values:
        raise ValueError('holm_bonferroni requires at least one p-value')
    for p in p_values:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f'p-values must be in [0, 1], got {p}')

    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    adjusted = np.minimum(1.0, np.maximum.accumulate((n - np.arange(n)) * sorted_p))
    corrected = np.zeros_like(adjusted)
    corrected[sorted_indices] = adjusted
    return corrected.tolist()


def bca_ci(
    scores: np.ndarray,
    rng: Generator,
    n_resamples: int = _BOOTSTRAP_N_RESAMPLES,
    ci_level: float = _CI_LEVEL,
) -> tuple[float, float]:
    """Compute bias-corrected and accelerated bootstrap CI for the mean.

    This implementation uses jackknife acceleration and percentile bootstrap
    with BCa transformation.
    """
    if scores.ndim != 1:
        raise ValueError(f'bca_ci expects 1D scores, got shape {scores.shape}')
    if len(scores) < 3:
        raise ValueError('bca_ci requires at least 3 observations')
    if not np.isfinite(scores).all():
        raise ValueError('bca_ci: scores contain non-finite values')

    n = len(scores)
    theta_hat = float(np.mean(scores))

    # Bootstrap distribution
    boot = np.zeros(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        sample = rng.choice(scores, size=n, replace=True)
        boot[i] = float(np.mean(sample))

    # Bias-correction
    prop_less = np.mean(boot < theta_hat)
    prop_less = np.clip(prop_less, 1e-10, 1 - 1e-10)
    z0 = stats.norm.ppf(prop_less)

    # Jackknife acceleration
    jack = np.zeros(n, dtype=np.float64)
    for i in range(n):
        jack_sample = np.delete(scores, i)
        jack[i] = float(np.mean(jack_sample))
    jack_mean = np.mean(jack)
    num = np.sum((jack_mean - jack) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5)
    accel = float(num / den) if den != 0 else 0.0

    alpha = 1.0 - ci_level
    z_low = stats.norm.ppf(alpha / 2)
    z_high = stats.norm.ppf(1 - alpha / 2)

    adj_low = stats.norm.cdf(z0 + (z0 + z_low) / (1 - accel * (z0 + z_low)))
    adj_high = stats.norm.cdf(z0 + (z0 + z_high) / (1 - accel * (z0 + z_high)))

    lower = float(np.quantile(boot, np.clip(adj_low, 0.0, 1.0)))
    upper = float(np.quantile(boot, np.clip(adj_high, 0.0, 1.0)))
    return lower, upper


def _check_power_args(effect_size: float, alpha: float, sided: int) -> None:
    if effect_size <= 0:
        raise ValueError(f'effect_size must be > 0, got {effect_size}')
    if not 0 < alpha < 1:
        raise ValueError(f'alpha must be in (0,1), got {alpha}')
    if sided not in SIDED:
        raise ValueError(f'sided must be 1 or 2, got {sided!r}')


def achieved_power(n_pairs: int, effect_size: float, alpha: float = 0.05, *, sided: int) -> float:
    """Power of the paired t-test at ``n_pairs`` for a true d_z of ``effect_size`` (D013).

    The paired t statistic has a noncentral t distribution with n - 1 degrees of freedom and
    noncentrality d_z * sqrt(n). ``sided`` is 1 (the registered one-sided test) or 2.

    Raises:
        ValueError: If n_pairs < 2, effect_size <= 0, alpha is outside (0, 1) or sided is
            not 1 or 2.
    """
    _check_power_args(effect_size, alpha, sided)
    if n_pairs < 2:
        raise ValueError(f'n_pairs must be >= 2, got {n_pairs}')
    df = n_pairs - 1
    ncp = effect_size * np.sqrt(n_pairs)
    if sided == 1:
        t_crit = stats.t.ppf(1.0 - alpha, df)
        return float(1.0 - stats.nct.cdf(t_crit, df, ncp))
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df)
    return float(1.0 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp))


def power_analysis(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.8,
    *,
    sided: int,
) -> int:
    """Smallest number of pairs whose paired t-test reaches ``power`` for d_z = effect_size.

    Uses the noncentral t distribution (D013; the normal approximation and the implicit
    two-sidedness of the earlier version were defect D20). ``sided`` is required: 1 for the
    registered one-sided family tests, 2 for two-sided use (G6). 12 pairs detect d_z >= 0.77 at
    80% power one-sided; 5 pairs detect d_z >= 1.36.

    Raises:
        ValueError: If effect_size <= 0, alpha or power are outside (0, 1), or sided is not
            1 or 2.
    """
    _check_power_args(effect_size, alpha, sided)
    if not 0 < power < 1:
        raise ValueError(f'power must be in (0,1), got {power}')
    n = 2
    while achieved_power(n, effect_size, alpha, sided=sided) < power:
        n += 1
        if n > 1_000_000:  # pragma: no cover
            raise RuntimeError('power_analysis did not converge')
    return n
