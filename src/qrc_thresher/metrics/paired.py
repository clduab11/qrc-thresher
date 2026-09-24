"""Paired comparisons for the comparative gates (docs/DECISIONS.md D013; defects D3, D11).

Two arms of manifest rows are joined on (config_hash, sweep_id, task_seed, reservoir_seed) with
the registered design labels; nothing is truncated or pooled, and row order never matters. A pair
present in one arm only, a duplicate pair that is not an exact rerun, a budget or design mismatch,
a metric other than the registered one, a non-finite value, an all-zero difference vector, too
few pairs or more than one config_hash / sweep_id gives INSUFFICIENT_EVIDENCE naming the cause.

Per comparison: the mean difference (arm A minus arm B), the paired t-test one-sided in the
registered direction (and two-sided), the Wilcoxon signed-rank test with the same alternative
(zero_method wilcox, exact when there are no ties among the non-zero |differences|, approx
otherwise, effective n reported), a 95% BCa bootstrap CI of the mean difference (B = 2000, rng
seed 20260923) and d_z (None at zero variance). Every result serialises with allow_nan=False.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from qrc_thresher.metrics.stats import bca_ci, d_z_effect

DIRECTIONS = ('greater', 'less')
BOOTSTRAP_N_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260923
CI_LEVEL = 0.95
MIN_PAIRS = 12
MATCH_RULES = ('budget', 'hash', None)
BUDGET_COLUMNS = ('n_configs', 'n_validation_evals')

Pair = Tuple[int, int]
ExpectedHash = Callable[[str, Pair], str]  # (design_hash, (task_seed, reservoir_seed)) -> hash


def _finite_or_none(value) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def paired_statistics(
    values_a: Sequence[float],
    values_b: Sequence[float],
    direction: str,
    *,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    rng_seed: int = BOOTSTRAP_SEED,
) -> dict:
    """The D013 statistics for two paired value vectors (arm A minus arm B).

    Args:
        values_a: Arm A values in pair order, shape (n,).
        values_b: Arm B values in the same pair order.
        direction: 'greater' (H1: mean(a - b) > 0) or 'less' (H1: mean(a - b) < 0).

    Returns:
        A JSON-ready dict: n_pairs, mean_a, mean_b, mean_diff, t_statistic (None when not
        finite), p_one_sided, p_two_sided, wilcoxon_statistic, p_wilcoxon, wilcoxon_method,
        wilcoxon_n_effective, ci_low, ci_high, ci_level, n_resamples, rng_seed, d_z, direction.

    Raises:
        ValueError: For an unknown direction, mismatched lengths, fewer than 3 pairs, or
            non-finite values.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f'direction must be one of {DIRECTIONS}; got {direction!r}')
    a = np.asarray(values_a, dtype=np.float64).ravel()
    b = np.asarray(values_b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f'paired arms must have equal length; got {a.shape} and {b.shape}')
    if len(a) < 3:
        raise ValueError(f'paired_statistics needs at least 3 pairs; got {len(a)}')
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError('paired_statistics: non-finite values')
    diff = a - b

    t_one = scipy_stats.ttest_rel(a, b, alternative=direction)
    t_two = scipy_stats.ttest_rel(a, b, alternative='two-sided')

    nonzero = diff[diff != 0.0]
    ties = len(np.unique(np.abs(nonzero))) < len(nonzero)
    method = 'approx' if ties else 'exact'
    if len(nonzero) == 0:
        w_stat, w_p = None, 1.0
    else:
        w = scipy_stats.wilcoxon(a, b, alternative=direction, zero_method='wilcox', method=method)
        w_stat, w_p = float(w.statistic), float(w.pvalue)

    ci_low, ci_high = bca_ci(
        diff, rng=np.random.default_rng(rng_seed), n_resamples=n_resamples, ci_level=CI_LEVEL
    )
    return {
        'n_pairs': int(len(a)),
        'mean_a': float(a.mean()),
        'mean_b': float(b.mean()),
        'mean_diff': float(diff.mean()),
        't_statistic': _finite_or_none(t_one.statistic),
        'p_one_sided': float(t_one.pvalue),
        'p_two_sided': float(t_two.pvalue),
        'wilcoxon_statistic': w_stat,
        'p_wilcoxon': float(w_p),
        'wilcoxon_method': method,
        'wilcoxon_n_effective': int(len(nonzero)),
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'ci_level': CI_LEVEL,
        'n_resamples': int(n_resamples),
        'rng_seed': int(rng_seed),
        'd_z': d_z_effect(diff),
        'direction': direction,
    }


@dataclass
class PairedComparison:
    """Outcome of ``compare_arms``: the pairing verdict and, when OK, every statistic."""

    status: str
    reason: Optional[str]
    metric: str
    direction: str
    designs: List[str]
    match: Optional[str]
    task_name_a: Optional[str]
    task_name_b: Optional[str]
    config_hash: Optional[str]
    sweep_id: Optional[str]
    n_pairs: int = 0
    pairs: List[List[int]] = field(default_factory=list)
    values_a: List[float] = field(default_factory=list)
    values_b: List[float] = field(default_factory=list)
    mean_a: Optional[float] = None
    mean_b: Optional[float] = None
    mean_diff: Optional[float] = None
    t_statistic: Optional[float] = None
    p_one_sided: Optional[float] = None
    p_two_sided: Optional[float] = None
    wilcoxon_statistic: Optional[float] = None
    p_wilcoxon: Optional[float] = None
    wilcoxon_method: Optional[str] = None
    wilcoxon_n_effective: Optional[int] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    ci_level: float = CI_LEVEL
    n_resamples: int = BOOTSTRAP_N_RESAMPLES
    rng_seed: int = BOOTSTRAP_SEED
    d_z: Optional[float] = None
    budgets_a: List[List[int]] = field(default_factory=list)
    budgets_b: List[List[int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-ready dict; serialises with allow_nan=False."""
        d = asdict(self)
        json.dumps(d, allow_nan=False)
        return d


def _insufficient(reason: str, **kwargs) -> PairedComparison:
    return PairedComparison(status='INSUFFICIENT_EVIDENCE', reason=reason, **kwargs)


def _pairs_text(pairs: Sequence[Pair]) -> str:
    return ', '.join(f'({t}, {r})' for t, r in sorted(pairs))


def _single(frame: pd.DataFrame, column: str) -> Optional[str]:
    values = sorted({str(v) for v in frame[column].tolist()}) if len(frame) else []
    return values[0] if len(values) == 1 else None


def _collapse(
    arm: pd.DataFrame, label: str
) -> Tuple[Optional[Dict[Pair, pd.Series]], Optional[str]]:
    """One row per pair; exact reruns (same circuit_hash and value) collapse, others are refused."""
    rows: Dict[Pair, pd.Series] = {}
    refused: List[Pair] = []
    for _, row in arm.iterrows():
        pair = (int(row['task_seed']), int(row['reservoir_seed']))
        if pair in rows:
            first = rows[pair]
            same = str(first['circuit_hash']) == str(row['circuit_hash']) and _same_value(
                first['primary_metric_value'], row['primary_metric_value']
            )
            if not same:
                refused.append(pair)
            continue
        rows[pair] = row
    if refused:
        return None, (
            f'arm {label} has more than one candidate row for pair(s) {_pairs_text(refused)} '
            'that are not exact reruns (same circuit_hash and value); refused, not chosen'
        )
    return rows, None


def _same_value(x, y) -> bool:
    try:
        fx, fy = float(x), float(y)
    except (TypeError, ValueError):
        return str(x) == str(y)
    if math.isnan(fx) and math.isnan(fy):
        return True
    return fx == fy


def _budget(row: pd.Series) -> List[int]:
    return [int(float(row[c])) for c in BUDGET_COLUMNS]


def compare_arms(
    arm_a: pd.DataFrame,
    arm_b: pd.DataFrame,
    *,
    metric: str,
    direction: str,
    designs: Tuple[str, str] = ('tuned', 'tuned'),
    match: Optional[str] = 'budget',
    expected_hash: Optional[ExpectedHash] = None,
    min_pairs: int = MIN_PAIRS,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    rng_seed: int = BOOTSTRAP_SEED,
) -> PairedComparison:
    """Pair two arms of manifest rows and compute the D013 statistics on arm A minus arm B.

    Args:
        arm_a, arm_b: Manifest rows (runs.csv columns) of the two arms, any order.
        metric: The registered primary_metric_name both arms must carry.
        direction: 'greater' or 'less' (H1 about mean(a - b)).
        designs: The design label each arm's rows must carry, e.g. ('tuned', 'inherited').
        match: 'budget' requires equal (n_configs, n_validation_evals) pair by pair; 'hash'
            requires row b's circuit_hash == expected_hash(row a's circuit_hash, (task_seed,
            reservoir_seed)); None requires neither (a default QRC against a tuned baseline).
        expected_hash: Required when match == 'hash'.
        min_pairs: The registered minimum number of pairs.

    Returns:
        A PairedComparison; status 'OK' or 'INSUFFICIENT_EVIDENCE' with a reason naming the
        pairs or columns concerned.

    Raises:
        ValueError: For an unknown direction or match rule, or match == 'hash' without
            expected_hash.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f'direction must be one of {DIRECTIONS}; got {direction!r}')
    if match not in MATCH_RULES:
        raise ValueError(f'match must be one of {MATCH_RULES}; got {match!r}')
    if match == 'hash' and expected_hash is None:
        raise ValueError(
            "match='hash' needs expected_hash(design_hash, (task_seed, reservoir_seed))"
        )
    designs = tuple(str(d) for d in designs)
    common = dict(
        metric=metric,
        direction=direction,
        designs=list(designs),
        match=match,
        task_name_a=_single(arm_a, 'task_name') if len(arm_a) else None,
        task_name_b=_single(arm_b, 'task_name') if len(arm_b) else None,
        config_hash=None,
        sweep_id=None,
    )
    both = pd.concat([arm_a, arm_b], ignore_index=True) if len(arm_a) or len(arm_b) else None
    if both is None or len(arm_a) == 0 or len(arm_b) == 0:
        return _insufficient(
            f'an arm has no rows (arm A: {len(arm_a)}, arm B: {len(arm_b)})', **common
        )
    for column in ('config_hash', 'sweep_id'):
        value = _single(both, column)
        if value is None:
            values = sorted({str(v) for v in both[column].tolist()})
            return _insufficient(
                f'rows carry more than one {column}: {values}; every arm must come from one '
                'config file and one tuning sweep', **common
            )
        common[column] = value

    for label, arm, design in (('A', arm_a, designs[0]), ('B', arm_b, designs[1])):
        wrong_design = arm[arm['design'].astype(str) != design]
        if len(wrong_design):
            pairs = [(int(r['task_seed']), int(r['reservoir_seed']))
                     for _, r in wrong_design.iterrows()]
            return _insufficient(
                f'arm {label} rows for pair(s) {_pairs_text(pairs)} carry design '
                f'{sorted(set(wrong_design["design"].astype(str)))} instead of the registered '
                f'{design!r}', **common
            )
        wrong_metric = arm[arm['primary_metric_name'].astype(str) != metric]
        if len(wrong_metric):
            pairs = [(int(r['task_seed']), int(r['reservoir_seed']))
                     for _, r in wrong_metric.iterrows()]
            return _insufficient(
                f'arm {label} rows for pair(s) {_pairs_text(pairs)} carry metric '
                f'{sorted(set(wrong_metric["primary_metric_name"].astype(str)))} instead of the '
                f'registered {metric!r}', **common
            )

    rows_a, reason = _collapse(arm_a, 'A')
    if reason:
        return _insufficient(reason, **common)
    rows_b, reason = _collapse(arm_b, 'B')
    if reason:
        return _insufficient(reason, **common)

    only_a = sorted(set(rows_a) - set(rows_b))
    only_b = sorted(set(rows_b) - set(rows_a))
    if only_a or only_b:
        parts = []
        if only_a:
            parts.append(f'pair(s) {_pairs_text(only_a)} present in arm A only')
        if only_b:
            parts.append(f'pair(s) {_pairs_text(only_b)} present in arm B only')
        return _insufficient('; '.join(parts) + '; nothing is truncated', **common)

    pairs = sorted(rows_a)
    if len(pairs) < min_pairs:
        return _insufficient(
            f'{len(pairs)} pair(s) present; the registered minimum is {min_pairs}', **common
        )

    values_a, values_b, budgets_a, budgets_b = [], [], [], []
    for pair in pairs:
        ra, rb = rows_a[pair], rows_b[pair]
        if match == 'budget':
            if _budget(ra) != _budget(rb):
                return _insufficient(
                    f'budget mismatch for pair {pair}: '
                    f'arm A {dict(zip(BUDGET_COLUMNS, _budget(ra)))} '
                    f'vs arm B {dict(zip(BUDGET_COLUMNS, _budget(rb)))} '
                    f'({", ".join(BUDGET_COLUMNS)} must be equal between two tuned arms)',
                    **common,
                )
        elif match == 'hash':
            want = expected_hash(str(ra['circuit_hash']), pair)
            if str(rb['circuit_hash']) != want:
                return _insufficient(
                    f'design mismatch for pair {pair}: arm B circuit_hash {rb["circuit_hash"]} is '
                    f'not the design hash plus the ablation suffix ({want})', **common
                )
        try:
            va, vb = float(ra['primary_metric_value']), float(rb['primary_metric_value'])
        except (TypeError, ValueError):
            va, vb = math.nan, math.nan
        if not (math.isfinite(va) and math.isfinite(vb)):
            return _insufficient(
                f'non-finite value for pair {pair}: arm A {ra["primary_metric_value"]!r}, '
                f'arm B {rb["primary_metric_value"]!r}', **common
            )
        values_a.append(va)
        values_b.append(vb)
        budgets_a.append(_budget(ra))
        budgets_b.append(_budget(rb))

    diff = np.array(values_a) - np.array(values_b)
    if np.all(diff == 0.0):
        return _insufficient(
            f'the difference vector is all zero over pair(s) {_pairs_text(pairs)} (identical arms)',
            **common,
        )

    statistics = paired_statistics(
        values_a, values_b, direction, n_resamples=n_resamples, rng_seed=rng_seed
    )
    return PairedComparison(
        status='OK',
        reason=None,
        pairs=[list(p) for p in pairs],
        values_a=values_a,
        values_b=values_b,
        budgets_a=budgets_a,
        budgets_b=budgets_b,
        **common,
        **{k: v for k, v in statistics.items() if k not in ('direction',)},
    )


__all__ = [
    'BOOTSTRAP_N_RESAMPLES',
    'BOOTSTRAP_SEED',
    'BUDGET_COLUMNS',
    'CI_LEVEL',
    'DIRECTIONS',
    'MATCH_RULES',
    'MIN_PAIRS',
    'PairedComparison',
    'compare_arms',
    'paired_statistics',
]
