"""The comparative family: registration, evaluation as a unit, and the re-registered gates
G1-G4 (defects D3, D4, D10, D11; docs/DECISIONS.md D013, D014).

- configs/gates/COMPARATIVE.v1.yaml loads into a frozen spec whose content hash is pinned, and it
  holds the registered members, metrics, directions, floors, alpha, min_pairs, B and seed.
- configs/comparative.yaml holds 12 seed pairs, the washout, parity_window 3, the three tuning
  grids and both baselines, and no gates block.
- evaluate_family computes all five raw p-values in one pass with m = 5 fixed; an INSUFFICIENT
  member contributes p = 1 and is shown as INSUFFICIENT, never FAIL; a three-member synthetic
  family reproduces a Holm known answer through the evaluator; verdicts are invariant under row
  order; unpaired rows give INSUFFICIENT; the floors; the "baseline better" report; the
  default-design report; the family JSON carries the provenance keys and no NaN.

The family module is imported inside each test, so that before it exists every test fails on its
own instead of the whole file failing at collection.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import stats as scipy_stats
from synthetic_rows import (
    CONFIG_HASH,
    PAIRS,
    RECORD_SHA,
    SWEEP_ID,
    ablation_hash,
    classical_arm,
    default_hash,
    default_inherited_arm,
    default_qrc_arm,
    design_hash,
    inherited_arm,
    qrc_arm,
)

from qrc_thresher.config import load_config

REPO_ROOT = Path(__file__).parent.parent
PROTOCOL = REPO_ROOT / 'configs' / 'gates' / 'COMPARATIVE.v1.yaml'
COMPARATIVE = REPO_ROOT / 'configs' / 'comparative.yaml'
# SHA-256 of the canonical JSON of the parsed YAML (sorted keys). Pinned when v1 was registered.
PROTOCOL_SHA256 = '9a051a19edb3c08c79785076b0312671c8703a450cceada149bb52002d829e40'
MEMBERS = ['G1', 'G2', 'G2.5', 'G3', 'G4']
ALPHA = 0.05
PROVENANCE_KEYS = {'config_hash', 'sweep_id', 'git_commit', 'measurement_model',
                   'measurement_label', 'protocol_sha256', 'timestamp_utc'}


def _family():
    return importlib.import_module('qrc_thresher.gates.comparative')


def _registered() -> dict:
    return yaml.safe_load(PROTOCOL.read_text(encoding='utf-8'))


class TestRegistration:
    def test_the_protocol_is_frozen_by_hash(self) -> None:
        canonical = json.dumps(_registered(), sort_keys=True, ensure_ascii=True)
        assert hashlib.sha256(canonical.encode()).hexdigest() == PROTOCOL_SHA256

    def test_the_spec_loads_and_carries_its_hash(self) -> None:
        protocol = _family().load_protocol()
        assert protocol.sha256 == PROTOCOL_SHA256
        assert protocol.source_path.endswith('configs/gates/COMPARATIVE.v1.yaml')
        assert (protocol.family, protocol.version) == ('COMPARATIVE', 1)
        assert list(protocol.members) == MEMBERS
        assert _family().FAMILY_SIZE == 5 and protocol.statistics.family_size == 5

    def test_the_registered_values(self) -> None:
        reg = _registered()
        assert reg['experiment_config'] == 'configs/comparative.yaml'
        assert reg['pairing']['min_pairs'] == 12 and reg['pairing']['washout'] == 50
        assert reg['pairing']['keys'] == [
            'config_hash', 'sweep_id', 'design', 'task_seed', 'reservoir_seed'
        ]
        stat = reg['statistics']
        members = reg['members']
        assert (stat['decision'], stat['alpha'], stat['correction'], stat['family_size']) == (
            'paired_t_one_sided', 0.05, 'holm', 5
        )
        assert stat['insufficient_member_p'] == 1.0
        assert stat['holm_family'] == [f'{m}.paired_t_one_sided' for m in MEMBERS]  # ruling 9
        defaults = reg['defaults']
        assert defaults['compared_qrc'] == {'design': 'default', 'depth': 3,
                                            'encoding_scale': math.pi, 'window': 2}
        assert defaults['reported_qrc'] == {'design': 'default_w1', 'depth': 3,
                                            'encoding_scale': math.pi, 'window': 1}
        assert defaults['esn'] == {'design': 'default', 'preset': 'esn_nonlinear'}
        assert defaults['one_candidate_row_per_pair'] is True
        assert defaults['comparators'] == {m: members[m]['baseline'] for m in MEMBERS}
        assert stat['bootstrap'] == {'method': 'bca', 'n_resamples': 2000,
                                     'rng_seed': 20260923, 'ci_level': 0.95}
        assert list(members) == MEMBERS
        assert {m: members[m]['metric'] for m in MEMBERS} == {
            'G1': 'accuracy', 'G2': 'accuracy', 'G2.5': 'stm_memory', 'G3': 'stm_memory',
            'G4': 'nrmse',
        }
        assert {m: members[m]['direction'] for m in MEMBERS} == {
            'G1': 'greater', 'G2': 'greater', 'G2.5': 'greater', 'G3': 'greater', 'G4': 'less',
        }
        assert {m: members[m]['design'] for m in MEMBERS} == {
            'G1': 'stm', 'G2': 'parity', 'G2.5': 'stm', 'G3': 'stm', 'G4': 'narma',
        }
        assert {m: members[m]['baseline'] for m in MEMBERS} == {
            'G1': 'ablation:no_entangle', 'G2': 'rks_parity', 'G2.5': 'ablation:haar',
            'G3': 'esn', 'G4': 'esn_narma',
        }
        assert members['G2']['floor'] == {'metric': 'accuracy', 'rule': 'mean_greater_than',
                                          'value': 0.70}
        assert members['G4']['floor'] == {'metric': 'nrmse', 'rule': 'mean_less_than',
                                          'value': 0.60}
        assert members['G1']['floor'] is None and members['G3']['floor'] is None
        assert members['G1']['readout'] == members['G2.5']['readout'] == 'z_only'
        assert members['G1']['clause_a'] == 'g07_v1_pass_on_tuned_design'

    def test_comparative_config_holds_the_registered_sweep(self) -> None:
        raw = yaml.safe_load(COMPARATIVE.read_text(encoding='utf-8'))
        cfg = load_config(COMPARATIVE)
        assert (cfg.seeds.task_seed, cfg.seeds.reservoir_seed, cfg.seeds.n_seeds) == (42, 137, 12)
        assert cfg.task.parity_window == 3 and cfg.training.washout == 50
        assert cfg.baseline.enabled == ['esn', 'random_features']
        assert cfg.baseline.esn_grid is None  # the ESN grid is tuning.esn (D011)
        assert 'gates' not in raw and 'esn_washout' not in raw['baseline']
        assert (cfg.reservoir.readout, cfg.reservoir.n_qubits, cfg.reservoir.depth) == (
            'z_only', 4, 3
        )
        assert cfg.reservoir.encoding_scale == math.pi
        grids = raw['tuning']
        assert grids['qrc'] == {
            'depth': [1, 2, 3, 4, 5], 'window': [1, 2, 4],
            'encoding_scale': [math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi],
        }
        assert grids['esn'] == {'spectral_radius': [0.8, 0.9, 0.95, 0.99, 1.0],
                                'input_scaling': [0.1, 0.5, 1.0],
                                'leak_rate': [0.1, 0.3, 0.5, 1.0]}
        assert grids['rks']['sigma'] == [float(s) for s in np.logspace(-1, 1, 20)]
        assert grids['rks']['window'] == [1, 2, 4]
        for grid in grids.values():
            assert math.prod(len(v) for v in grid.values()) == 60
        tuning = cfg.tuning
        assert tuning.qrc.encoding_scale[-1] == math.pi and len(tuning.rks.sigma) == 20
        alpha = load_config(REPO_ROOT / 'configs' / 'alpha_lite.yaml')
        assert (alpha.training.ridge_alphas, alpha.training.cv_folds) == (
            cfg.training.ridge_alphas, cfg.training.cv_folds
        )
        assert alpha.measurement.model == cfg.measurement.model == 'exact'

    def test_no_config_has_a_gates_block(self) -> None:
        from qrc_thresher import config as config_module

        for path in sorted((REPO_ROOT / 'configs').glob('*.yaml')):
            assert 'gates' not in yaml.safe_load(path.read_text(encoding='utf-8')), path.name
        assert 'gates' not in config_module.AlphaLiteConfig.model_fields
        assert not hasattr(config_module, 'GateThresholds')


# --- synthetic sweeps -------------------------------------------------------------------------

def _designs():
    fam = _family()
    return fam.Designs(
        tuned={task: {p: design_hash(task, p) for p in PAIRS} for task in ('stm', 'parity',
                                                                          'narma')},
        default={'default': {p: default_hash(2, p) for p in PAIRS},
                 'default_w1': {p: default_hash(1, p) for p in PAIRS}},
        ablation_hash=ablation_hash,
    )


def _sweep(rng_seed: int = 0, *, g2_mean: float = 0.85, g4_mean: float = 0.45,
           g3_sign: float = 1.0, with_defaults: bool = True) -> pd.DataFrame:
    """Every arm of the family with clear margins in the registered directions."""
    rng = np.random.default_rng(rng_seed)
    n = len(PAIRS)
    stm = rng.uniform(2.0, 3.0, size=n)
    par_stm = rng.uniform(0.80, 0.95, size=n)
    par = np.clip(g2_mean + rng.uniform(-0.05, 0.05, size=n), 0, 1)
    narma = g4_mean + rng.uniform(-0.03, 0.03, size=n)
    frames = [
        qrc_arm('stm', 'stm', stm, metric='stm_memory',
                secondary={p: {'mc_total': v + 0.7, 'mc_k0': 0.7} for p, v in zip(PAIRS, stm)}),
        qrc_arm('parity', 'stm', par_stm, metric='accuracy'),
        qrc_arm('parity', 'parity', par, metric='accuracy'),
        qrc_arm('narma', 'narma', narma, metric='nrmse'),
        inherited_arm('no_entangle', 'parity', 'stm', par_stm - rng.uniform(0.15, 0.3, size=n),
                      metric='accuracy'),
        inherited_arm('no_entangle', 'stm', 'stm', stm + rng.uniform(0.1, 0.5, size=n),
                      metric='stm_memory'),
        inherited_arm('haar', 'stm', 'stm', stm - rng.uniform(0.3, 0.8, size=n),
                      metric='stm_memory'),
        classical_arm('esn', stm - g3_sign * rng.uniform(0.3, 0.8, size=n), metric='stm_memory',
                      secondary={p: {'mc_total': 0.0, 'mc_k0': 0.9} for p in PAIRS}),
        classical_arm('rks_parity', par - rng.uniform(0.15, 0.3, size=n), metric='accuracy'),
        classical_arm('esn_narma', narma + rng.uniform(0.05, 0.2, size=n), metric='nrmse'),
    ]
    if with_defaults:
        for w in (1, 2):
            d_stm = rng.uniform(0.1, 0.4, size=n)
            d_par = rng.uniform(0.5, 0.7, size=n)
            frames += [
                default_qrc_arm('stm', w, d_stm, metric='stm_memory'),
                default_qrc_arm('parity', w, d_par, metric='accuracy'),
                default_qrc_arm('narma', w, narma + 0.2, metric='nrmse'),
            ]
            if w == 2:  # the compared default (design 'default'); w = 1 is reported only
                frames += [
                    default_inherited_arm('no_entangle', 'parity', w, d_par - 0.05,
                                          metric='accuracy'),
                    default_inherited_arm('haar', 'stm', w, d_stm - 0.05, metric='stm_memory'),
                ]
        frames += [
            classical_arm('esn', stm - 1.0, metric='stm_memory', design='default', budget=(1, 0)),
            classical_arm('esn_narma', narma + 0.3, metric='nrmse', design='default',
                          budget=(1, 0)),
        ]
    return pd.concat(frames, ignore_index=True)


# The newest results/gates/G0.7.tuned_qrc.*.json, as the gate command discovers it (ruling 4):
# its verdict, path, SHA-256 and the model details that name the tuning config and sweep.
G07_PASS = {
    'result': 'PASS',
    'json': 'results/gates/G0.7.tuned_qrc.20260923T120000000000Z.json',
    'sha256': hashlib.sha256(b'g07 tuned json').hexdigest(),
    'model_details': {'tuning_config_hash': CONFIG_HASH, 'sweep_id': SWEEP_ID},
}


def _evaluate(runs, **kwargs):
    fam = _family()
    defaults = dict(g07_tuned=G07_PASS, readout='z_only', config_hash=CONFIG_HASH,
                    sweep_id=SWEEP_ID, tuning_record_sha=RECORD_SHA)
    defaults.update(kwargs)
    return fam.evaluate_family(runs, fam.load_protocol(), _designs(), **defaults)


def _without_timestamps(result: dict) -> dict:
    out = json.loads(json.dumps(result))
    out.pop('timestamp_utc', None)
    return out


def _holm_by_hand(raw: dict) -> dict:
    names = sorted(raw, key=lambda m: raw[m])
    m, running, adjusted = len(names), 0.0, {}
    for i, name in enumerate(names):
        running = max(running, (m - i) * raw[name])
        adjusted[name] = min(1.0, running)
    return adjusted


class TestFamilyEvaluation:
    def test_all_five_members_in_one_pass_with_m_five(self) -> None:
        result = _evaluate(_sweep())
        assert result['family'] == 'COMPARATIVE' and result['version'] == 1
        assert result['family_size'] == 5 and result['alpha'] == ALPHA
        assert list(result['members']) == MEMBERS
        raw = {m: result['members'][m]['raw_p'] for m in MEMBERS}
        assert all(0.0 <= p <= 1.0 for p in raw.values())
        adjusted = {m: result['members'][m]['adjusted_p'] for m in MEMBERS}
        assert adjusted == pytest.approx(_holm_by_hand(raw))
        for m in MEMBERS:
            member = result['members'][m]
            assert member['result'] == 'PASS', (m, member)
            assert member['n_pairs'] == 12
            assert member['comparison']['status'] == 'OK'
            assert member['baseline_better'] is False
            assert member['adjusted_p'] <= ALPHA
        assert result['protocol_sha256'] == PROTOCOL_SHA256

    def test_verdicts_and_statistics_are_invariant_under_row_order(self) -> None:
        runs = _sweep()
        base = _without_timestamps(_evaluate(runs))
        for seed in (1, 2):
            shuffled = _without_timestamps(_evaluate(runs.sample(frac=1, random_state=seed)))
            assert shuffled == base

    def test_an_insufficient_member_contributes_p_one_and_is_not_a_fail(self) -> None:
        runs = _sweep()
        result = _evaluate(runs[runs['task_name'] != 'esn_narma'])
        g4 = result['members']['G4']
        assert g4['result'] == 'INSUFFICIENT_EVIDENCE'
        assert g4['raw_p'] == 1.0 and g4['adjusted_p'] == 1.0
        assert 'esn_narma' in g4['message']
        raw = {m: result['members'][m]['raw_p'] for m in MEMBERS}
        adjusted = {m: result['members'][m]['adjusted_p'] for m in MEMBERS}
        assert adjusted == pytest.approx(_holm_by_hand(raw))  # m = 5 including the 1.0
        assert all(result['members'][m]['result'] == 'PASS' for m in MEMBERS if m != 'G4')

    def test_a_three_member_family_reproduces_a_holm_known_answer(self) -> None:
        runs = _sweep()
        runs = runs[~runs['task_name'].isin(['esn_narma', 'ablation:no_entangle'])]
        result = _evaluate(runs)
        members = result['members']
        assert members['G1']['result'] == members['G4']['result'] == 'INSUFFICIENT_EVIDENCE'
        computed = {m: members[m]['raw_p'] for m in ('G2', 'G2.5', 'G3')}
        for m in computed:
            c = members[m]['comparison']
            expected = scipy_stats.ttest_rel(
                c['values_a'], c['values_b'], alternative=c['direction']
            ).pvalue
            assert computed[m] == pytest.approx(float(expected))
        p1, p2, p3 = sorted(computed.values())
        order = sorted(computed, key=computed.get)
        assert members[order[0]]['adjusted_p'] == pytest.approx(min(1.0, 5 * p1))
        assert members[order[1]]['adjusted_p'] == pytest.approx(min(1.0, max(5 * p1, 4 * p2)))
        assert members[order[2]]['adjusted_p'] == pytest.approx(
            min(1.0, max(5 * p1, 4 * p2, 3 * p3))
        )
        assert members['G1']['adjusted_p'] == members['G4']['adjusted_p'] == 1.0

    def test_unpaired_rows_make_the_member_insufficient_and_name_the_pair(self) -> None:
        runs = _sweep()
        drop = (runs['task_name'] == 'ablation:haar') & (runs['task_seed'] == 47)
        result = _evaluate(runs[~drop])
        g25 = result['members']['G2.5']
        assert g25['result'] == 'INSUFFICIENT_EVIDENCE' and '(47, 142)' in g25['message']
        assert result['members']['G3']['result'] == 'PASS'

    def test_the_floors(self) -> None:
        low = _evaluate(_sweep(g2_mean=0.65))
        g2 = low['members']['G2']
        assert g2['adjusted_p'] <= ALPHA and g2['result'] == 'FAIL'
        assert g2['floor']['passed'] is False and g2['floor']['value'] == 0.70
        assert g2['floor']['observed'] == pytest.approx(g2['comparison']['mean_a'])
        high = _evaluate(_sweep(g4_mean=0.70))
        g4 = high['members']['G4']
        assert g4['adjusted_p'] <= ALPHA and g4['result'] == 'FAIL'
        assert g4['floor']['passed'] is False and g4['floor']['value'] == 0.60
        ok = _evaluate(_sweep())
        assert ok['members']['G2']['floor']['passed'] and ok['members']['G4']['floor']['passed']
        assert ok['members']['G3']['floor'] is None

    def test_baseline_better_is_reported_with_a_two_sided_p(self) -> None:
        result = _evaluate(_sweep(g3_sign=-1.0))
        g3 = result['members']['G3']
        assert g3['result'] == 'FAIL' and g3['baseline_better'] is True
        assert g3['comparison']['mean_diff'] < 0
        assert g3['p_two_sided'] < ALPHA
        assert g3['p_two_sided'] == pytest.approx(g3['comparison']['p_two_sided'])
        assert g3['raw_p'] > 0.5  # the one-sided p in the registered direction

    def test_g1_needs_the_tuned_g07_verdict(self) -> None:
        runs = _sweep()
        passed = _evaluate(runs)['members']['G1']
        assert passed['result'] == 'PASS' and passed['g07']['result'] == 'PASS'
        assert passed['g07']['json'] == G07_PASS['json']
        assert passed['g07']['sha256'] == G07_PASS['sha256']  # ruling 4: path and hash recorded
        assert (passed['g07']['config_hash'], passed['g07']['sweep_id']) == (CONFIG_HASH, SWEEP_ID)
        failed = _evaluate(runs, g07_tuned={**G07_PASS, 'result': 'FAIL'})['members']['G1']
        assert failed['result'] == 'FAIL' and failed['adjusted_p'] <= ALPHA
        missing = _evaluate(runs, g07_tuned=None)['members']['G1']
        assert missing['result'] == 'INSUFFICIENT_EVIDENCE' and 'G0.7' in missing['message']
        assert missing['raw_p'] == 1.0

    def test_g1_refuses_a_g07_file_from_another_config_or_sweep(self) -> None:
        # Ruling 4: "newest" is only the discovery rule; the file's hashes must match the rows.
        runs = _sweep()
        other_hash = hashlib.sha256(b'another config').hexdigest()
        foreign = {**G07_PASS, 'model_details': {'tuning_config_hash': other_hash,
                                                  'sweep_id': SWEEP_ID}}
        g1 = _evaluate(runs, g07_tuned=foreign)['members']['G1']
        assert g1['result'] == 'INSUFFICIENT_EVIDENCE' and g1['raw_p'] == 1.0
        assert other_hash in g1['message'] and CONFIG_HASH in g1['message']
        stale = {**G07_PASS, 'model_details': {'tuning_config_hash': CONFIG_HASH,
                                                'sweep_id': 'older'}}
        g1 = _evaluate(runs, g07_tuned=stale)['members']['G1']
        assert g1['result'] == 'INSUFFICIENT_EVIDENCE'
        assert 'older' in g1['message'] and SWEEP_ID in g1['message']

    def test_g1_reports_the_stm_memory_margin_beside_the_gated_parity_margin(self) -> None:
        g1 = _evaluate(_sweep())['members']['G1']
        assert g1['comparison']['metric'] == 'accuracy'
        beside = g1['reported']['stm_memory_margin_over_no_entangle']
        assert beside['status'] == 'OK' and beside['metric'] == 'stm_memory'
        assert beside['mean_diff'] < 0  # the synthetic no-entangle rows have more linear memory
        assert 'adjusted_p' not in beside  # reported, never gated

    def test_g3_reports_mc_k0_separately(self) -> None:
        g3 = _evaluate(_sweep())['members']['G3']
        assert g3['reported']['mc_k0'] == {'qrc': pytest.approx(0.7), 'esn': pytest.approx(0.9)}
        assert g3['comparison']['metric'] == 'stm_memory'

    def test_entanglement_members_need_the_z_only_readout(self) -> None:
        result = _evaluate(_sweep(), readout='z_and_zz')
        for m in ('G1', 'G2.5'):
            member = result['members'][m]
            assert member['result'] == 'INSUFFICIENT_EVIDENCE' and 'z_only' in member['message']
            assert member['raw_p'] == 1.0
        assert result['members']['G2']['result'] == 'PASS'

    def test_the_default_design_report(self) -> None:
        # Ruling 1: one compared default (design 'default', w = 2) per member and task, the
        # w = 1 row ('default_w1') reported as a mean; ruling 5: the comparators.
        result = _evaluate(_sweep())
        for m in MEMBERS:
            default = result['members'][m]['default']
            assert default['status'] == 'OK', (m, default)
            assert default['designs'][0] == 'default'
            assert 'adjusted_p' not in default  # never in the Holm family
            reported = result['members'][m]['default_w1']
            assert reported['design'] == 'default_w1' and reported['n_rows'] == 12
            assert math.isfinite(reported['mean']) and math.isfinite(reported['std'])
        g3 = result['members']['G3']['default']
        assert (g3['designs'], g3['task_name_b']) == (['default', 'default'], 'esn')
        g4 = result['members']['G4']['default']
        assert (g4['designs'], g4['task_name_b']) == (['default', 'default'], 'esn_narma')
        g2 = result['members']['G2']['default']
        assert (g2['designs'], g2['task_name_b']) == (['default', 'tuned'], 'rks_parity')
        g1 = result['members']['G1']['default']
        assert (g1['designs'], g1['task_name_b']) == (['default', 'inherited'],
                                                       'ablation:no_entangle')
        g25 = result['members']['G2.5']['default']
        assert (g25['designs'], g25['task_name_b']) == (['default', 'inherited'], 'ablation:haar')
        bare = _evaluate(_sweep(with_defaults=False))
        for m in MEMBERS:
            assert bare['members'][m]['result'] == 'PASS'
            assert bare['members'][m]['default']['status'] == 'INSUFFICIENT_EVIDENCE'
            assert bare['members'][m]['default_w1']['n_rows'] == 0

    def test_the_default_table_never_flips_a_verdict(self) -> None:
        # Ruling 5: a member that passes on the default and fails on the tuned design is FAIL.
        runs = _sweep(g3_sign=-1.0)  # tuned QRC loses to the tuned ESN ...
        esn_default = (runs['task_name'] == 'esn') & (runs['design'] == 'default')
        runs = runs.copy()
        runs.loc[esn_default, 'primary_metric_value'] = 0.0  # ... but the default QRC wins
        g3 = _evaluate(runs)['members']['G3']
        assert g3['default']['status'] == 'OK' and g3['default']['mean_diff'] > 0
        assert g3['default']['p_one_sided'] < ALPHA
        assert g3['result'] == 'FAIL' and g3['baseline_better'] is True

    def test_two_candidate_rows_for_one_pair_make_the_member_insufficient(self) -> None:
        # Ruling 1: the evaluator never chooses between two candidate QRC rows for a pair.
        runs = _sweep()
        extra = runs[(runs['task_name'] == 'stm') & (runs['design'] == 'tuned')
                     & (runs['task_seed'] == 45)].copy()
        extra['primary_metric_value'] = extra['primary_metric_value'] + 0.01
        result = _evaluate(pd.concat([runs, extra], ignore_index=True))
        for m in ('G2.5', 'G3'):
            member = result['members'][m]
            assert member['result'] == 'INSUFFICIENT_EVIDENCE' and '(45, 140)' in member['message']
        assert result['members']['G2']['result'] == 'PASS'
        dup_default = runs[(runs['task_name'] == 'parity') & (runs['design'] == 'default')
                           & (runs['task_seed'] == 43)].copy()
        dup_default['primary_metric_value'] = 0.99
        result = _evaluate(pd.concat([runs, dup_default], ignore_index=True))
        g2 = result['members']['G2']
        assert g2['result'] == 'PASS'  # the tuned verdict is untouched ...
        assert g2['default']['status'] == 'INSUFFICIENT_EVIDENCE'  # ... the default table is not
        assert '(43, 138)' in g2['default']['reason']

    def test_rows_of_another_config_or_sweep_are_ignored(self) -> None:
        runs = _sweep()
        foreign = runs.copy()
        foreign['sweep_id'] = 'older'
        result = _evaluate(pd.concat([runs, foreign], ignore_index=True))
        assert all(result['members'][m]['result'] == 'PASS' for m in MEMBERS)
        assert result['members']['G3']['n_pairs'] == 12


class TestFamilyReport:
    def test_the_json_carries_the_provenance_and_no_nan(self, tmp_path) -> None:
        fam = _family()
        result = _evaluate(_sweep())
        paths = fam.write_family_report(result, tmp_path)
        assert set(paths) == {'family', *MEMBERS}
        family = json.loads(paths['family'].read_text(encoding='utf-8'))
        assert PROVENANCE_KEYS <= set(family)
        assert family['config_hash'] == CONFIG_HASH and family['sweep_id'] == SWEEP_ID
        assert family['measurement_label'] == 'exact (oracle upper bound)'
        assert family['protocol_sha256'] == PROTOCOL_SHA256
        assert family['holm_family'] == [f'{m}.paired_t_one_sided' for m in MEMBERS]  # ruling 9
        assert family['tuning_record_sha'] == RECORD_SHA
        family_sha = hashlib.sha256(paths['family'].read_bytes()).hexdigest()
        for m in MEMBERS:
            gate = json.loads(paths[m].read_text(encoding='utf-8'))
            assert PROVENANCE_KEYS <= set(gate)
            assert gate['gate'] == m and gate['n_pairs'] == 12
            assert {'raw_p', 'adjusted_p', 'comparison', 'result'} <= set(gate)
            # Ruling 7: the family file is the record, the per-gate file a view of it.
            assert Path(gate['family_json']).name == paths['family'].name
            assert gate['family_sha256'] == family_sha
            assert {'p_one_sided', 'p_two_sided', 'p_wilcoxon', 'ci_low', 'ci_high', 'd_z',
                    'mean_diff', 'n_pairs'} <= set(gate['comparison'])
        text = paths['family'].read_text(encoding='utf-8')
        assert 'NaN' not in text and 'Infinity' not in text

    def test_reports_are_timestamped_and_never_overwritten(self, tmp_path) -> None:
        fam = _family()
        result = _evaluate(_sweep())
        first = fam.write_family_report(result, tmp_path)
        second = fam.write_family_report(result, tmp_path)
        assert first['family'] != second['family'] and first['G3'] != second['G3']
        assert first['family'].name.startswith('COMPARATIVE.v1.')
        assert first['G3'].name.startswith('G3.') and first['G2.5'].name.startswith('G2.5.')
        assert len(list(tmp_path.glob('*.json'))) == 12
        assert not (tmp_path / 'G3.json').exists()


class TestGateCommand:
    @pytest.mark.parametrize('name', ['family', 'G3'])
    def test_the_family_needs_the_tuning_record(self, name, tmp_path, monkeypatch) -> None:
        from click.testing import CliRunner

        from qrc_thresher.cli import cli

        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(cli, ['gate', name, '--config', str(COMPARATIVE)])
        assert result.exit_code == 2, result.output
        assert 'INSUFFICIENT_EVIDENCE' in result.output
        assert 'tuning' in result.output.lower()
        assert not (tmp_path / 'results' / 'gates' / f'{name}.json').exists()
