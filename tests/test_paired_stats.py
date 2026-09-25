"""Paired statistics for the comparative gates (defects D3, D20; docs/DECISIONS.md D013).

- ``metrics.paired.paired_statistics`` gives, for two paired value vectors: the paired t-test
  one-sided in the registered direction (and two-sided), the Wilcoxon signed-rank test with the
  same alternative (zero_method wilcox, exact without ties, the effective n), the 95% BCa CI of
  the mean difference (B = 2000, rng seed 20260923) and d_z (null at zero variance). Known answers
  come from scipy on a hand-made example.
- ``metrics.paired.compare_arms`` joins two arms of manifest rows on
  (config_hash, sweep_id, task_seed, reservoir_seed) with the registered design labels; every
  statistic and the verdict are invariant under row shuffling; a missing pair, a non-exact
  duplicate, a budget mismatch, a design mismatch, identical arms, a non-finite value, too few
  pairs or two config hashes give INSUFFICIENT_EVIDENCE naming the cause; an exact rerun collapses.
- ``paired_test`` names d_z ``d_z`` (null at zero variance); ``power_analysis`` takes a required
  ``sided`` and uses the noncentral t: 12 pairs detect d_z >= 0.77 one-sided at 80% power.

The paired module is imported inside each test, so that before it exists every test fails on its
own instead of the whole file failing at collection.
"""

from __future__ import annotations

import importlib
import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats
from synthetic_rows import (
    CONFIG_HASH,
    PAIRS,
    TUNED_BUDGET,
    ablation_hash,
    classical_arm,
    design_hash,
    inherited_arm,
    qrc_arm,
)

from qrc_thresher.metrics.stats import bca_ci, paired_test, power_analysis

# Hand-made example: 12 paired values, no ties among the non-zero |differences|.
DIFF = np.array([0.31, -0.12, 0.44, 0.05, 0.27, 0.62, -0.03, 0.18, 0.39, 0.09, 0.51, 0.22])
A = np.array([0.81, 0.77, 0.90, 0.66, 0.72, 0.95, 0.70, 0.83, 0.88, 0.75, 0.92, 0.79])
B = A - DIFF
TIED = B.copy()
TIED[0] = A[0] - DIFF[4]  # two differences of 0.27: a tie among the non-zero |d|
BOOTSTRAP_B, BOOTSTRAP_SEED = 2000, 20260923


def _paired():
    return importlib.import_module('qrc_thresher.metrics.paired')


def _dict(comparison) -> dict:
    d = comparison.to_dict()
    json.dumps(d, allow_nan=False)  # every report is written with allow_nan=False
    return d


class TestStatisticsKnownAnswers:
    def test_the_registered_constants(self) -> None:
        paired = _paired()
        assert paired.BOOTSTRAP_N_RESAMPLES == BOOTSTRAP_B
        assert paired.BOOTSTRAP_SEED == BOOTSTRAP_SEED
        assert set(paired.DIRECTIONS) == {'greater', 'less'}

    @pytest.mark.parametrize('direction', ['greater', 'less'])
    def test_the_paired_t_and_wilcoxon_match_scipy(self, direction) -> None:
        s = _paired().paired_statistics(A, B, direction=direction)
        t = scipy_stats.ttest_rel(A, B, alternative=direction)
        assert s['n_pairs'] == 12
        assert s['mean_diff'] == pytest.approx(float(DIFF.mean()))
        assert s['t_statistic'] == pytest.approx(float(t.statistic))
        assert s['p_one_sided'] == pytest.approx(float(t.pvalue))
        two = scipy_stats.ttest_rel(A, B, alternative='two-sided')
        assert s['p_two_sided'] == pytest.approx(float(two.pvalue))
        w = scipy_stats.wilcoxon(A, B, alternative=direction, zero_method='wilcox', method='exact')
        assert s['wilcoxon_statistic'] == pytest.approx(float(w.statistic))
        assert s['p_wilcoxon'] == pytest.approx(float(w.pvalue))
        assert s['wilcoxon_method'] == 'exact'
        assert s['wilcoxon_n_effective'] == 12
        assert s['d_z'] == pytest.approx(float(DIFF.mean() / DIFF.std(ddof=1)))
        assert s['direction'] == direction

    def test_one_sided_p_is_the_registered_direction(self) -> None:
        paired = _paired()
        greater = paired.paired_statistics(A, B, direction='greater')['p_one_sided']
        less = paired.paired_statistics(A, B, direction='less')['p_one_sided']
        assert greater < 0.01 < 0.99 < less
        assert greater + less == pytest.approx(1.0)

    def test_ties_switch_the_wilcoxon_to_approx_and_zeros_leave_the_effective_n(self) -> None:
        paired = _paired()
        tied = paired.paired_statistics(A, TIED, direction='greater')
        w = scipy_stats.wilcoxon(A, TIED, alternative='greater', zero_method='wilcox',
                                 method='approx')
        assert tied['wilcoxon_method'] == 'approx'
        assert tied['p_wilcoxon'] == pytest.approx(float(w.pvalue))
        with_zero = B.copy()
        with_zero[1] = A[1]  # one zero difference, dropped by zero_method='wilcox'
        s = paired.paired_statistics(A, with_zero, direction='greater')
        assert s['wilcoxon_n_effective'] == 11
        assert s['n_pairs'] == 12

    def test_the_bca_ci_is_the_registered_bootstrap(self) -> None:
        s = _paired().paired_statistics(A, B, direction='greater')
        lo, hi = bca_ci(DIFF, rng=np.random.default_rng(BOOTSTRAP_SEED), n_resamples=BOOTSTRAP_B)
        assert (s['ci_low'], s['ci_high']) == (pytest.approx(lo), pytest.approx(hi))
        assert lo < DIFF.mean() < hi

    def test_bca_ci_reproduces_a_known_interval(self) -> None:
        # scipy's BCa on the same sample; both are Monte Carlo, so agree to 0.02 (0.1 sd).
        lo, hi = bca_ci(DIFF, rng=np.random.default_rng(BOOTSTRAP_SEED), n_resamples=20000)
        ref = scipy_stats.bootstrap(
            (DIFF,), np.mean, n_resamples=20000, method='BCa',
            random_state=np.random.default_rng(1), confidence_level=0.95,
        ).confidence_interval
        assert lo == pytest.approx(float(ref.low), abs=0.02)
        assert hi == pytest.approx(float(ref.high), abs=0.02)

    def test_d_z_is_null_at_zero_variance_and_nothing_is_nan(self) -> None:
        s = _paired().paired_statistics(A, A - 0.1, direction='greater')
        assert s['d_z'] is None
        assert s['t_statistic'] is None  # scipy's t is not finite; the p-value is kept
        assert 0.0 <= s['p_one_sided'] <= 1.0
        json.dumps(s, allow_nan=False)

    def test_an_unknown_direction_is_refused(self) -> None:
        with pytest.raises(ValueError, match='direction'):
            _paired().paired_statistics(A, B, direction='two-sided')


class TestPairedTestAndPower:
    def test_paired_test_names_d_z_and_writes_null_at_zero_variance(self) -> None:
        result = paired_test(A, B)
        assert result.d_z == pytest.approx(float(DIFF.mean() / DIFF.std(ddof=1)))
        assert not hasattr(result, 'cohens_d')
        assert paired_test(A, A - 0.1).d_z is None

    def test_power_analysis_requires_a_sidedness(self) -> None:
        with pytest.raises(TypeError):
            power_analysis(effect_size=0.5, alpha=0.05, power=0.8)
        with pytest.raises(ValueError, match='sided'):
            power_analysis(effect_size=0.5, alpha=0.05, power=0.8, sided=3)

    def test_power_analysis_known_values_use_the_noncentral_t(self) -> None:
        assert power_analysis(effect_size=0.77, alpha=0.05, power=0.8, sided=1) == 12
        assert power_analysis(effect_size=1.36, alpha=0.05, power=0.8, sided=1) == 5
        assert power_analysis(effect_size=0.5, alpha=0.05, power=0.8, sided=2) == 34
        assert power_analysis(effect_size=0.5, alpha=0.05, power=0.8, sided=1) == 27

    def test_achieved_power_at_the_registered_run_count(self) -> None:
        from qrc_thresher.metrics.stats import achieved_power

        assert achieved_power(12, 0.77, alpha=0.05, sided=1) == pytest.approx(0.803, abs=0.005)
        assert achieved_power(5, 1.36, alpha=0.05, sided=1) == pytest.approx(0.800, abs=0.005)
        assert achieved_power(11, 0.77, alpha=0.05, sided=1) < 0.8
        assert achieved_power(12, 0.77, alpha=0.05, sided=2) < 0.7


def _arms(n: int = 12, task: str = 'stm'):
    rng = np.random.default_rng(0)
    pairs = PAIRS[:n]
    qrc = rng.uniform(2.0, 3.0, size=n)
    esn = qrc - rng.uniform(0.1, 0.6, size=n)
    return (
        qrc_arm(task, task, qrc, metric='stm_memory', pairs=pairs),
        classical_arm('esn', esn, metric='stm_memory', pairs=pairs),
    )


def _compare(a, b, **kwargs):
    defaults = dict(metric='stm_memory', direction='greater', designs=('tuned', 'tuned'),
                    match='budget', min_pairs=12)
    defaults.update(kwargs)
    return _paired().compare_arms(a, b, **defaults)


class TestCompareArms:
    def test_statistics_and_verdict_are_invariant_under_row_shuffling(self) -> None:
        a, b = _arms()
        base = _dict(_compare(a, b))
        assert base['status'] == 'OK'
        assert base['n_pairs'] == 12
        assert base['pairs'] == [list(p) for p in PAIRS]
        for seed in range(3):
            shuffled = _compare(a.sample(frac=1, random_state=seed),
                                b.sample(frac=1, random_state=seed + 10))
            assert _dict(shuffled) == base

    def test_the_values_are_the_rows_in_pair_order_and_match_the_statistics(self) -> None:
        a, b = _arms()
        c = _compare(a, b)
        va = [float(a.set_index(['task_seed', 'reservoir_seed']).loc[p, 'primary_metric_value'])
              for p in PAIRS]
        vb = [float(b.set_index(['task_seed', 'reservoir_seed']).loc[p, 'primary_metric_value'])
              for p in PAIRS]
        assert c.values_a == pytest.approx(va) and c.values_b == pytest.approx(vb)
        s = _paired().paired_statistics(np.array(va), np.array(vb), direction='greater')
        assert c.p_one_sided == pytest.approx(s['p_one_sided'])
        assert (c.ci_low, c.ci_high) == (pytest.approx(s['ci_low']), pytest.approx(s['ci_high']))
        assert c.d_z == pytest.approx(s['d_z'])
        assert c.mean_a == pytest.approx(np.mean(va)) and c.mean_b == pytest.approx(np.mean(vb))

    def test_a_missing_pair_is_insufficient_and_named(self) -> None:
        a, b = _arms()
        c = _compare(a, b[b['task_seed'] != 47])
        assert c.status == 'INSUFFICIENT_EVIDENCE'
        assert '(47, 142)' in c.reason
        c = _compare(a[a['task_seed'] != 50], b)
        assert c.status == 'INSUFFICIENT_EVIDENCE' and '(50, 145)' in c.reason

    def test_fewer_than_min_pairs_is_insufficient(self) -> None:
        a, b = _arms(11)
        c = _compare(a, b)
        assert c.status == 'INSUFFICIENT_EVIDENCE'
        assert '11' in c.reason and '12' in c.reason
        assert _compare(a, b, min_pairs=11).status == 'OK'

    def test_an_exact_rerun_collapses_but_a_different_rerun_is_refused(self) -> None:
        a, b = _arms()
        exact = pd.concat([a, a.iloc[[3]]], ignore_index=True)
        c = _compare(exact, b)
        assert c.status == 'OK' and c.n_pairs == 12
        assert _dict(c) == _dict(_compare(a, b))
        changed = a.iloc[[3]].copy()
        changed['primary_metric_value'] = changed['primary_metric_value'] + 1e-6
        c = _compare(pd.concat([a, changed], ignore_index=True), b)
        assert c.status == 'INSUFFICIENT_EVIDENCE' and '(45, 140)' in c.reason
        other_hash = a.iloc[[3]].copy()
        other_hash['circuit_hash'] = 'f' * 64
        c = _compare(pd.concat([a, other_hash], ignore_index=True), b)
        assert c.status == 'INSUFFICIENT_EVIDENCE' and '(45, 140)' in c.reason

    def test_a_budget_mismatch_between_tuned_arms_is_insufficient(self) -> None:
        a, b = _arms()
        b = b.copy()
        b.loc[b['task_seed'] == 44, 'n_configs'] = TUNED_BUDGET[0] - 1
        c = _compare(a, b)
        assert c.status == 'INSUFFICIENT_EVIDENCE' and '(44, 139)' in c.reason
        assert 'n_configs' in c.reason

    def test_a_design_label_other_than_the_registered_one_is_insufficient(self) -> None:
        a, b = _arms()
        b = b.copy()
        b.loc[b['task_seed'] == 46, 'design'] = 'default'
        c = _compare(a, b)
        assert c.status == 'INSUFFICIENT_EVIDENCE' and '(46, 141)' in c.reason

    def test_an_inherited_arm_must_match_the_design_hash_plus_suffix(self) -> None:
        rng = np.random.default_rng(1)
        full = rng.uniform(0.7, 1.0, size=12)
        a = qrc_arm('parity', 'stm', full, metric='accuracy')
        b = inherited_arm('no_entangle', 'parity', 'stm', full - 0.2, metric='accuracy')
        expected = lambda design, pair: ablation_hash(design, 'no_entangle', pair)  # noqa: E731
        c = _compare(a, b, metric='accuracy', designs=('tuned', 'inherited'), match='hash',
                     expected_hash=expected)
        assert c.status == 'OK' and c.n_pairs == 12
        wrong = b.copy()
        wrong.loc[wrong['task_seed'] == 49, 'circuit_hash'] = design_hash('stm', (49, 144))
        c = _compare(a, wrong, metric='accuracy', designs=('tuned', 'inherited'), match='hash',
                     expected_hash=expected)
        assert c.status == 'INSUFFICIENT_EVIDENCE' and '(49, 144)' in c.reason
        with pytest.raises(ValueError, match='expected_hash'):
            _compare(a, b, metric='accuracy', designs=('tuned', 'inherited'), match='hash')

    def test_identical_arms_and_non_finite_values_are_insufficient(self) -> None:
        a, _ = _arms()
        same = a.copy()
        same['task_name'] = 'esn'
        same['circuit_hash'] = 'e' * 64
        c = _compare(a, same)
        assert c.status == 'INSUFFICIENT_EVIDENCE' and 'zero' in c.reason.lower()
        a2, b2 = _arms()
        b2 = b2.copy()
        b2.loc[b2['task_seed'] == 42, 'primary_metric_value'] = np.nan
        c = _compare(a2, b2)
        assert c.status == 'INSUFFICIENT_EVIDENCE' and '(42, 137)' in c.reason
        json.dumps(c.to_dict(), allow_nan=False)

    def test_two_config_hashes_or_sweep_ids_are_insufficient(self) -> None:
        a, b = _arms()
        b = b.copy()
        b.loc[b['task_seed'] == 43, 'config_hash'] = 'a' * 64
        c = _compare(a, b)
        assert c.status == 'INSUFFICIENT_EVIDENCE' and 'config_hash' in c.reason
        a, b = _arms()
        b = b.copy()
        b['sweep_id'] = 'another'
        c = _compare(a, b)
        assert c.status == 'INSUFFICIENT_EVIDENCE' and 'sweep_id' in c.reason
        assert CONFIG_HASH in _compare(*_arms()).to_dict()['config_hash']

    def test_a_metric_other_than_the_registered_one_is_insufficient(self) -> None:
        a, b = _arms()
        b = b.copy()
        b.loc[b['task_seed'] == 42, 'primary_metric_name'] = 'mc'
        c = _compare(a, b)
        assert c.status == 'INSUFFICIENT_EVIDENCE' and 'mc' in c.reason

    def test_d_z_is_null_and_the_json_has_no_nan_at_constant_differences(self) -> None:
        a, b = _arms()
        b = b.copy()
        b['primary_metric_value'] = a['primary_metric_value'].to_numpy() - 0.25
        c = _compare(a, b)
        d = _dict(c)
        assert c.status == 'OK' and d['d_z'] is None and d['mean_diff'] == pytest.approx(0.25)

    def test_the_less_direction_tests_a_smaller_mean(self) -> None:
        rng = np.random.default_rng(2)
        qrc = rng.uniform(0.3, 0.5, size=12)
        a = qrc_arm('narma', 'narma', qrc, metric='nrmse')
        b = classical_arm('esn_narma', qrc + rng.uniform(0.05, 0.2, size=12), metric='nrmse')
        c = _compare(a, b, metric='nrmse', direction='less')
        assert c.status == 'OK' and c.p_one_sided < 0.01 and c.mean_diff < 0
        assert c.p_one_sided == pytest.approx(float(
            scipy_stats.ttest_rel(c.values_a, c.values_b, alternative='less').pvalue
        ))


# --- CP5 (docs/DECISIONS.md D019, items A.1 and A.2): every comparison names its rows ------------

NEW_FIELDS = ('run_ids_a', 'run_ids_b', 'circuit_hashes_a', 'circuit_hashes_b', 'run_ids')


def _by_pair(arm: pd.DataFrame, column: str) -> dict:
    return {(int(r['task_seed']), int(r['reservoir_seed'])): r[column] for _, r in arm.iterrows()}


class TestComparisonsNameTheirRows:
    def test_per_pair_run_ids_and_hashes_come_in_pair_order(self) -> None:
        a, b = _arms()
        c = _compare(a, b)
        d = _dict(c)
        assert c.status == 'OK'
        ids_a, ids_b = _by_pair(a, 'run_id'), _by_pair(b, 'run_id')
        hashes_a, hashes_b = _by_pair(a, 'circuit_hash'), _by_pair(b, 'circuit_hash')
        pairs = [tuple(p) for p in d['pairs']]
        assert pairs == PAIRS
        assert d['run_ids_a'] == [[ids_a[p]] for p in pairs]
        assert d['run_ids_b'] == [[ids_b[p]] for p in pairs]
        assert d['circuit_hashes_a'] == [hashes_a[p] for p in pairs]
        assert d['circuit_hashes_b'] == [hashes_b[p] for p in pairs]
        union = sorted(set(ids_a.values()) | set(ids_b.values()))
        assert d['run_ids'] == union and len(d['run_ids']) == 24

    def test_an_exact_rerun_with_a_distinct_run_id_lists_both_ids_and_keeps_the_smallest(self):
        a, b = _arms()
        base = _dict(_compare(a, b))
        rerun = a.iloc[[3]].copy()  # pair (45, 140), same circuit_hash and value
        rerun['run_id'] = 'zzz-rerun'
        first, second = pd.concat([a, rerun], ignore_index=True), pd.concat([rerun, a],
                                                                            ignore_index=True)
        c1, c2 = _dict(_compare(first, b)), _dict(_compare(second, b))
        assert c1 == c2  # whichever row comes first
        original = str(a.iloc[3]['run_id'])
        assert c1['run_ids_a'][3] == sorted([original, 'zzz-rerun'])
        assert 'zzz-rerun' in c1['run_ids'] and original in c1['run_ids']
        # The statistics are those of the collapsed arm, and the kept row is the smallest id.
        for key in ('mean_a', 'mean_b', 'mean_diff', 'p_one_sided', 'ci_low', 'ci_high', 'd_z',
                    'wilcoxon_statistic', 'values_a', 'values_b', 'n_pairs'):
            assert c1[key] == base[key], key
        assert c1['circuit_hashes_a'] == base['circuit_hashes_a']
        assert base['run_ids_a'][3] == [original] and min(original, 'zzz-rerun') == original

    def test_collapse_exact_reruns_is_public_and_order_independent(self) -> None:
        paired = _paired()
        a, _ = _arms()
        rerun = a.iloc[[3]].copy()
        rerun['run_id'] = '000-first'  # sorts before the content-derived id
        rows, run_ids, reason = paired.collapse_exact_reruns(pd.concat([a, rerun]), 'A')
        assert reason is None and set(rows) == set(PAIRS)
        assert str(rows[(45, 140)]['run_id']) == '000-first'  # the smallest str(run_id) is kept
        assert run_ids[(45, 140)] == sorted(['000-first', str(a.iloc[3]['run_id'])])
        assert all(len(run_ids[p]) == 1 for p in PAIRS if p != (45, 140))
        rows2, run_ids2, _ = paired.collapse_exact_reruns(pd.concat([rerun, a]), 'A')
        assert str(rows2[(45, 140)]['run_id']) == '000-first' and run_ids2 == run_ids
        differing = a.iloc[[3]].copy()
        differing['primary_metric_value'] = differing['primary_metric_value'] + 1e-6
        rows3, run_ids3, reason3 = paired.collapse_exact_reruns(pd.concat([a, differing]), 'A')
        assert rows3 is None and '(45, 140)' in reason3 and 'exact rerun' in reason3

    def test_an_insufficient_comparison_has_the_five_new_fields_empty(self) -> None:
        a, b = _arms()
        c = _dict(_compare(a, b[b['task_seed'] != 47]))
        assert c['status'] == 'INSUFFICIENT_EVIDENCE'
        for key in NEW_FIELDS:
            assert c[key] == [], key
