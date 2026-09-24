"""Matched tuning budgets (defects D5, D13 and the CP2/CP3 carry-overs; docs/DECISIONS.md D011).

On a tiny tuning block (2 configurations per model, 2 seed pairs, T = 100):
- the ESN, QRC and RKS tuners log equal budgets, select on validation blocks only, build one
  feature matrix per configuration, and write one record per task with the D011 schema;
- `run` and the engine deploy design_TASK(pair): the row's circuit_hash is the record's, with
  design=tuned, the record's sweep_id and its budget; `ablation` inherits design_TASK(pair) with
  the D010 suffix hash and design=inherited; `baseline` deploys the record's ESN and RKS;
- the ESN validation score excludes k = 0; the QRC tuner varies depth, window and scale;
- the ESN bias is drawn after W_in, leaves W and W_in unchanged for the same seed, enters the
  update inside the tanh and is covered by weight_hash;
- RKS has std(W) = sigma / sqrt(d), its features use the zero-padded window, and at d = 1 the
  stream positions of the D010 draw are unchanged.

The tuning module is imported inside each test, so that before it exists every test fails on its
own instead of the whole file failing at collection.
"""

from __future__ import annotations

import copy
import csv
import dataclasses
import hashlib
import importlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from qrc_thresher.config import AlphaLiteConfig, load_config
from qrc_thresher.metrics.scoring import memory_capacity
from qrc_thresher.tasks.stm import generate_stm

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG = REPO_ROOT / 'configs' / 'alpha_lite.yaml'
PAIRS = [(42, 137), (43, 138)]
TASKS = ['stm', 'parity', 'narma']
MODELS = ['qrc', 'esn', 'rks']
WASHOUT = 20
TINY_TUNING = {
    'qrc': {'depth': [1, 2], 'window': [2], 'encoding_scale': [math.pi / 2]},
    'esn': {'spectral_radius': [0.9], 'input_scaling': [0.1, 1.0], 'leak_rate': [1.0]},
    'rks': {'sigma': [0.5, 2.0], 'window': [2]},
}
N_CONFIGS, N_EVALS = 2, 10  # 2 configurations x 5 validation blocks
SELECTION = {'stm': ('stm_memory', True), 'parity': ('accuracy', True), 'narma': ('nrmse', False)}


def _tuning():
    return importlib.import_module('qrc_thresher.tuning')


def _wq():
    return importlib.import_module('qrc_thresher.reservoirs.windowed_qrc')


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _raw(*, tuning: bool = True) -> dict:
    """One tiny config for every task: the task is a command argument (D013), so the three
    records share the config hash, as they do for configs/comparative.yaml."""
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding='utf-8'))
    raw['experiment_name'] = 'tiny_tuning'
    raw['task'].update({'name': 'stm', 'length': 100, 'delay_max': 5, 'parity_window': 2})
    raw['reservoir'].update({'n_qubits': 2, 'depth': 1, 'readout': 'z_only', 'window': 1})
    raw['training']['washout'] = WASHOUT
    raw['seeds']['n_seeds'] = len(PAIRS)
    raw['baseline']['enabled'] = ['esn', 'random_features']
    if tuning:
        raw['tuning'] = copy.deepcopy(TINY_TUNING)
        raw['baseline'].pop('esn_grid')
    return raw


def _write(tmp: Path, **kwargs) -> Path:
    path = tmp / 'configs' / 'tiny.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_raw(**kwargs)), encoding='utf-8')
    return path


def _rows(path: Path) -> list:
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope='module')
def tuned(tmp_path_factory):
    """One tuned working directory: the tiny config and its three records (one per task)."""
    import os

    tmp = tmp_path_factory.mktemp('tuned')
    cfg_path = _write(tmp)
    cfg = load_config(cfg_path)
    tuning = _tuning()
    cwd = os.getcwd()
    os.chdir(tmp)
    try:
        records = {task: tuning.tune_config(cfg, task, cfg_path) for task in TASKS}
    finally:
        os.chdir(cwd)
    return {'dir': tmp, 'cfg_path': cfg_path, 'cfg': cfg, 'records': records}


class TestConfig:
    def test_a_tuning_block_is_validated_and_excludes_baseline_esn_grid(self) -> None:
        raw = _raw()
        cfg = AlphaLiteConfig.model_validate(raw)
        assert cfg.tuning.qrc.depth == [1, 2] and cfg.tuning.rks.window == [2]
        assert cfg.tuning.esn.input_scaling == [0.1, 1.0]
        both = _raw()
        both['baseline']['esn_grid'] = TINY_TUNING['esn']
        with pytest.raises(ValueError, match='esn_grid'):
            AlphaLiteConfig.model_validate(both)
        unknown = _raw()
        unknown['tuning']['qrc']['shots'] = [10]
        with pytest.raises(ValueError):
            AlphaLiteConfig.model_validate(unknown)
        assert load_config(DEFAULT_CONFIG).tuning is None

    def test_the_run_paths_refuse_a_tuning_config_without_its_record(self, tmp_path,
                                                                     monkeypatch) -> None:
        from qrc_thresher.commands.run import run_handler
        from qrc_thresher.engine import ParallelRunner

        cfg_path = _write(tmp_path)
        monkeypatch.chdir(tmp_path)
        cfg = load_config(cfg_path)
        assert run_handler('stm', str(cfg_path)) != 0
        manifests = ParallelRunner(config=cfg, max_workers=1).run_seeds('stm', config_path=cfg_path)
        assert all(not m.success and 'tuning' in (m.failure_reason or '') for m in manifests)


class TestRecord:
    def test_the_record_is_written_under_the_config_hash_with_the_schema(self, tuned) -> None:
        from qrc_thresher.proof.run_manifest import _config_hash

        tuning = _tuning()
        config_hash = _config_hash(tuned['cfg_path'])
        for task in TASKS:
            record = tuned['records'][task]
            path = tuned['dir'] / 'results' / 'tuning' / config_hash / f'{task}.json'
            assert path.exists()
            assert tuning.record_path(config_hash, task) == Path('results') / 'tuning' / (
                config_hash
            ) / f'{task}.json'
            on_disk = json.loads(path.read_text(encoding='utf-8'))
            assert on_disk == json.loads(json.dumps(record))
            assert record['config_hash'] == config_hash and record['task'] == task
            assert record['washout'] == WASHOUT
            assert record['reservoir'] == {'n_qubits': 2, 'readout': 'z_only',
                                           'backend': 'default.qubit'}
            # PI ruling 3: scope, folds, seeds and the record's own hash.
            assert record['selection_scope'] == 'train_cv'
            assert record['cv_folds'] == 5
            assert record['seeds'] == [list(p) for p in PAIRS]
            without = {k: v for k, v in record.items() if k != 'record_sha256'}
            canonical = json.dumps(without, sort_keys=True, ensure_ascii=True)
            assert record['record_sha256'] == hashlib.sha256(canonical.encode()).hexdigest()
            assert re.fullmatch(r'\d{8}T\d{6}\d*Z', record['sweep_id']), record['sweep_id']
            name, higher = SELECTION[task]
            assert record['selection_metric'] == {'name': name, 'higher_is_better': higher}
            assert record['grids'] == TINY_TUNING
            for model in MODELS:
                assert set(record[model]) == {f'{t}/{r}' for t, r in PAIRS}
                for key, entry in record[model].items():
                    assert (entry['task_seed'], entry['reservoir_seed']) == tuple(
                        int(v) for v in key.split('/')
                    )
                    assert re.fullmatch('[0-9a-f]{64}', entry['circuit_hash'])
                    assert (entry['n_configs'], entry['n_validation_evals']) == (N_CONFIGS, N_EVALS)
                    assert len(entry['configs']) == N_CONFIGS
                    winner = entry['configs'][entry['winner_index']]
                    assert winner['circuit_hash'] == entry['circuit_hash']
                    assert not winner['degenerate']
                    for c in entry['configs']:
                        assert {'circuit_hash', 'score', 'degenerate', 'reason'} <= set(c)
                        assert c['degenerate'] or math.isfinite(c['score'])
            qrc = next(iter(record['qrc'].values()))
            assert {'depth', 'window', 'encoding_scale'} <= set(qrc)
            esn = next(iter(record['esn'].values()))
            assert {'spectral_radius', 'input_scaling', 'leak_rate'} <= set(esn)
            rks = next(iter(record['rks'].values()))
            assert {'sigma', 'window'} <= set(rks)
            json.dumps(record, allow_nan=False)

    def test_the_three_tuners_log_equal_budgets(self, tuned) -> None:
        for task in TASKS:
            budgets = {
                (model, key): (entry['n_configs'], entry['n_validation_evals'])
                for model in MODELS
                for key, entry in tuned['records'][task][model].items()
            }
            assert set(budgets.values()) == {(N_CONFIGS, N_EVALS)}, (task, budgets)

    def test_the_winner_is_the_best_score_with_ties_to_grid_order(self, tuned) -> None:
        for task in TASKS:
            _, higher = SELECTION[task]
            for model in MODELS:
                for entry in tuned['records'][task][model].values():
                    scores = [c['score'] for c in entry['configs'] if not c['degenerate']]
                    best = max(scores) if higher else min(scores)
                    first = next(i for i, c in enumerate(entry['configs'])
                                 if not c['degenerate'] and c['score'] == best)
                    assert entry['winner_index'] == first

    def test_load_record_and_design_for_pair(self, tuned, monkeypatch) -> None:
        tuning = _tuning()
        monkeypatch.chdir(tuned['dir'])
        record = tuning.load_record(tuned['cfg_path'], 'stm')
        assert record == json.loads(json.dumps(tuned['records']['stm']))
        design = tuning.design_for_pair(record, 'qrc', 43, 138)
        assert design == record['qrc']['43/138']
        with pytest.raises(KeyError, match='44'):
            tuning.design_for_pair(record, 'qrc', 44, 139)


class TestSelection:
    def test_selection_uses_validation_blocks_only(self, tmp_path, monkeypatch) -> None:
        from qrc_thresher.tasks import stm

        tuning = _tuning()
        cfg_path = _write(tmp_path)
        cfg = load_config(cfg_path)
        monkeypatch.chdir(tmp_path)
        first = tuning.tune_config(cfg, 'stm', cfg_path)
        original = stm.generate_stm

        def scrambled(*args, **kwargs):
            ds = original(*args, **kwargs)
            targets = ds.targets.copy()
            tail = targets[ds.train_end:]
            targets[ds.train_end:] = np.random.default_rng(0).normal(size=tail.shape)
            return dataclasses.replace(ds, targets=targets)

        monkeypatch.setattr(stm, 'generate_stm', scrambled)
        second = tuning.tune_config(cfg, 'stm', cfg_path)
        for model in MODELS:
            for key in first[model]:
                a, b = first[model][key], second[model][key]
                assert a['winner_index'] == b['winner_index'], (model, key)
                assert [c['score'] for c in a['configs']] == [c['score'] for c in b['configs']]
                assert a['circuit_hash'] == b['circuit_hash']

    def test_one_feature_matrix_per_configuration(self, tmp_path, monkeypatch) -> None:
        from qrc_thresher.baselines import esn as esn_module
        from qrc_thresher.baselines import random_features

        tuning, wq = _tuning(), _wq()
        cfg_path = _write(tmp_path)
        cfg = load_config(cfg_path)
        monkeypatch.chdir(tmp_path)
        calls = {'qrc': [], 'esn': [], 'rks': []}
        originals = {
            'qrc': wq.WindowedReservoir.features,
            'esn': esn_module.ESN.states,
            'rks': random_features.extract_rks_features,
        }

        def spy_qrc(self, u, *args, **kwargs):
            calls['qrc'].append(self.circuit_hash)
            return originals['qrc'](self, u, *args, **kwargs)

        def spy_esn(self, u, *args, **kwargs):
            calls['esn'].append(self.weight_hash())
            return originals['esn'](self, u, *args, **kwargs)

        def spy_rks(u, params, *args, **kwargs):
            calls['rks'].append(wq.rks_circuit_hash(params))
            return originals['rks'](u, params, *args, **kwargs)

        monkeypatch.setattr(wq.WindowedReservoir, 'features', spy_qrc)
        monkeypatch.setattr(esn_module.ESN, 'states', spy_esn)
        monkeypatch.setattr(random_features, 'extract_rks_features', spy_rks)
        record = tuning.tune_config(cfg, 'parity', cfg_path)
        for model in MODELS:
            expected = sorted(c['circuit_hash'] for e in record[model].values()
                              for c in e['configs'])
            assert sorted(calls[model]) == expected, model  # once each, nothing more

    def test_the_selection_score_is_the_harness_readout_on_the_blocks(self, tuned) -> None:
        from sklearn.model_selection import KFold

        from qrc_thresher.readout import fit_ridge_cv
        from qrc_thresher.tasks.stm import generate_stm

        wq = _wq()
        cfg = tuned['cfg']
        for key, entry in tuned['records']['stm']['qrc'].items():
            task_seed, reservoir_seed = (int(v) for v in key.split('/'))
            ds = generate_stm(length=100, delay_max=5, train_frac=0.7,
                              rng=np.random.default_rng(task_seed))
            rows = np.arange(WASHOUT, ds.train_end)
            for c in entry['configs']:
                raw = cfg.model_dump()
                raw['reservoir'].update({'depth': c['depth'], 'window': c['window'],
                                         'encoding_scale': c['encoding_scale']})
                reservoir = wq.reservoir_from_config(AlphaLiteConfig.model_validate(raw),
                                                     reservoir_seed)
                assert reservoir.circuit_hash == c['circuit_hash']
                X = reservoir.features(ds.u)
                scores = []
                for fit_idx, val_idx in KFold(n_splits=cfg.training.cv_folds).split(rows):
                    fit_rows, val_rows = rows[fit_idx], rows[val_idx]
                    model = fit_ridge_cv(X[fit_rows], ds.targets[fit_rows],
                                         cfg.training.ridge_alphas, cfg.training.cv_folds)
                    pred = model.predict(X[val_rows])
                    scores.append(memory_capacity(pred[:, 1:], ds.targets[val_rows][:, 1:]))
                assert c['score'] == pytest.approx(float(np.mean(scores)), abs=1e-8)


class TestDeployment:
    def test_run_and_the_engine_deploy_the_validated_design(self, tuned, monkeypatch) -> None:
        from qrc_thresher.commands.run import run_handler
        from qrc_thresher.engine import ParallelRunner

        monkeypatch.chdir(tuned['dir'])
        for task in TASKS:
            record = tuned['records'][task]
            runs = Path('results') / 'runs.csv'
            runs.unlink(missing_ok=True)
            assert run_handler(task, str(tuned['cfg_path'])) == 0
            manifests = ParallelRunner(config=tuned['cfg'], max_workers=1).run_seeds(
                task, config_path=tuned['cfg_path']
            )
            assert all(m.success for m in manifests), [m.failure_reason for m in manifests]
            rows = [r for r in _rows(runs) if r['task_name'] == task]
            assert len(rows) == 2 * len(PAIRS)  # run and the engine, every pair
            for r in rows:
                design = record['qrc'][f"{r['task_seed']}/{r['reservoir_seed']}"]
                assert r['circuit_hash'] == design['circuit_hash']
                assert r['design'] == 'tuned' and r['sweep_id'] == record['sweep_id']
                assert r['tuning_record_sha'] == record['record_sha256']  # ruling 3
                assert (r['n_configs'], r['n_validation_evals']) == (str(N_CONFIGS), str(N_EVALS))

    def test_run_can_deploy_another_tasks_design(self, tuned, monkeypatch) -> None:
        from qrc_thresher.commands.run import run_handler

        monkeypatch.chdir(tuned['dir'])
        (Path('results') / 'runs.csv').unlink(missing_ok=True)
        stm_record, parity_record = tuned['records']['stm'], tuned['records']['parity']
        assert run_handler('parity', str(tuned['cfg_path']), design_task='stm') == 0
        rows = _rows(Path('results') / 'runs.csv')
        assert len(rows) == len(PAIRS)
        for r in rows:
            key = f"{r['task_seed']}/{r['reservoir_seed']}"
            assert r['task_name'] == 'parity' and r['design'] == 'tuned'
            assert r['circuit_hash'] == stm_record['qrc'][key]['circuit_hash']
            assert r['sweep_id'] == stm_record['sweep_id']
        assert parity_record['task'] == 'parity'  # the parity record is not what was deployed

    @pytest.mark.parametrize('name', ['no_entangle', 'haar', 'phase_random'])
    def test_ablation_inherits_the_design_with_the_suffix_hash(self, name, tuned,
                                                               monkeypatch) -> None:
        from qrc_thresher.commands.ablation import ablation_handler

        tuning = _tuning()
        record = tuned['records']['parity']
        monkeypatch.chdir(tuned['dir'])
        (Path('results') / 'runs.csv').unlink(missing_ok=True)
        assert ablation_handler(name, 'parity', str(tuned['cfg_path'])) == 0
        rows = _rows(Path('results') / 'runs.csv')
        assert [(int(r['task_seed']), int(r['reservoir_seed'])) for r in rows] == PAIRS
        for r in rows:
            design = record['qrc'][f"{r['task_seed']}/{r['reservoir_seed']}"]
            ablated = tuning.tuned_reservoir(tuned['cfg'], design, int(r['reservoir_seed']),
                                             ablation=name)
            assert r['circuit_hash'] == ablated.circuit_hash
            assert r['design'] == 'inherited' and r['sweep_id'] == record['sweep_id']
            assert r['tuning_record_sha'] == record['record_sha256']
            assert (r['n_configs'], r['n_validation_evals']) == ('1', '0')
            if name == 'no_entangle':
                assert r['circuit_hash'] == _sha(f"{design['circuit_hash']},entangle=False")

    def test_baseline_deploys_the_records_esn_and_rks(self, tuned, monkeypatch) -> None:
        from qrc_thresher.commands.baseline import run_baselines

        names = {'stm': ('esn', 'rks'), 'parity': ('esn_parity', 'rks_parity'),
                 'narma': ('esn_narma', 'rks_narma')}
        monkeypatch.chdir(tuned['dir'])
        for task in TASKS:
            record = tuned['records'][task]
            (Path('results') / 'runs.csv').unlink(missing_ok=True)
            manifests = run_baselines(tuned['cfg'], task, tuned['cfg_path'])
            assert all(m.success for m in manifests), [m.failure_reason for m in manifests]
            rows = _rows(Path('results') / 'runs.csv')
            assert {r['task_name'] for r in rows} == set(names[task])
            assert len(rows) == 2 * len(PAIRS)
            for r in rows:
                model = 'esn' if r['task_name'].startswith('esn') else 'rks'
                entry = record[model][f"{r['task_seed']}/{r['reservoir_seed']}"]
                assert r['circuit_hash'] == entry['circuit_hash'], (task, r['task_name'])
                assert r['design'] == 'tuned' and r['sweep_id'] == record['sweep_id']
                assert r['tuning_record_sha'] == record['record_sha256']
                assert (r['n_configs'], r['n_validation_evals']) == (str(N_CONFIGS), str(N_EVALS))

    def test_the_tune_command(self, tmp_path, monkeypatch) -> None:
        from click.testing import CliRunner

        from qrc_thresher.cli import cli
        from qrc_thresher.proof.run_manifest import _config_hash

        cfg_path = _write(tmp_path)
        monkeypatch.chdir(tmp_path)
        result = CliRunner().invoke(cli, ['tune', 'stm', '--config', str(cfg_path)])
        assert result.exit_code == 0, result.output
        record = tmp_path / 'results' / 'tuning' / _config_hash(cfg_path) / 'stm.json'
        assert record.exists()


class TestQRCGrid:
    def test_the_tuner_varies_depth_window_and_scale(self, tuned) -> None:
        wq = _wq()
        for key, entry in tuned['records']['stm']['qrc'].items():
            reservoir_seed = int(key.split('/')[1])
            points = {(c['depth'], c['window'], c['encoding_scale']) for c in entry['configs']}
            assert points == {(1, 2, math.pi / 2), (2, 2, math.pi / 2)}
            for c in entry['configs']:
                raw = tuned['cfg'].model_dump()
                raw['reservoir'].update({'depth': c['depth'], 'window': c['window'],
                                         'encoding_scale': c['encoding_scale']})
                tuned_cfg = AlphaLiteConfig.model_validate(raw)
                built = wq.reservoir_from_config(tuned_cfg, reservoir_seed)
                assert built.circuit_hash == c['circuit_hash']
                assert (built.depth, built.window, built.encoding_scale) == (
                    c['depth'], c['window'], c['encoding_scale']
                )

    def test_each_depth_is_its_own_draw_from_the_seed(self) -> None:
        from qrc_thresher.reservoirs.pennylane_qrc import build_reservoir_params

        shallow = build_reservoir_params(n_qubits=4, depth=2, readout='z_only',
                                         backend='default.qubit', rng=np.random.default_rng(137))
        deep = build_reservoir_params(n_qubits=4, depth=3, readout='z_only',
                                      backend='default.qubit', rng=np.random.default_rng(137))
        assert np.array_equal(shallow.thetas, deep.thetas[:2])  # thetas first ...
        assert not np.array_equal(shallow.phis, deep.phis[:2])  # ... then phis: a new draw


class TestESN:
    def _esn(self):
        return importlib.import_module('qrc_thresher.baselines.esn')

    def test_the_bias_is_drawn_after_w_in_and_leaves_w_and_w_in_unchanged(self) -> None:
        esn = self._esn()
        n = 4
        draw = esn.draw_reservoir(n, 137)
        rng = np.random.default_rng(137)
        W = rng.normal(0.0, 1.0, size=(n, n))
        Win = rng.uniform(-1.0, 1.0, size=(n, 1))
        b = rng.uniform(-1.0, 1.0, size=n)
        assert np.array_equal(draw.W_unit, esn.scale_recurrent(W, 1.0))
        assert np.array_equal(draw.Win_unit, Win)
        assert draw.b_unit.shape == (n,) and np.array_equal(draw.b_unit, b)

    def test_the_bias_enters_the_update_scaled_by_input_scaling(self) -> None:
        esn = self._esn()
        params = esn.ESNParams(spectral_radius=0.9, input_scaling=0.5, leak_rate=0.3)
        model = esn.build_esn(esn.draw_reservoir(4, 137), params)
        assert np.array_equal(model.b, 0.5 * esn.draw_reservoir(4, 137).b_unit)
        u = np.random.default_rng(42).uniform(-1.0, 1.0, size=50)
        x = np.zeros(4)
        expected = []
        for u_t in u:
            x = 0.7 * x + 0.3 * np.tanh(model.W @ x + model.Win[:, 0] * u_t + model.b)
            expected.append(x.copy())
        np.testing.assert_allclose(model.states(u), np.array(expected), rtol=0, atol=1e-14)

    def test_weight_hash_covers_the_bias(self) -> None:
        esn = self._esn()
        params = esn.ESNParams(spectral_radius=0.9, input_scaling=0.5, leak_rate=0.3)
        model = esn.build_esn(esn.draw_reservoir(4, 137), params)
        other = dataclasses.replace(model, b=model.b + 1e-9)
        assert model.weight_hash() != other.weight_hash()
        again = esn.build_esn(esn.draw_reservoir(4, 137), params)
        assert model.weight_hash() == again.weight_hash()

    def test_the_validation_score_excludes_k_zero(self) -> None:
        esn = self._esn()
        ds = generate_stm(length=300, delay_max=5, train_frac=0.7, rng=np.random.default_rng(42))
        cfg = load_config(DEFAULT_CONFIG)
        grid = {'spectral_radius': [0.9], 'input_scaling': [0.1, 1.0], 'leak_rate': [1.0]}
        search = esn.tune_esn(ds.u, ds.targets, ds.train_end, esn.draw_reservoir(4, 137), grid,
                              50, cfg.training.ridge_alphas, cfg.training.cv_folds, task='stm')
        assert search.score_name == 'stm_memory' and search.higher_is_better is True
        from sklearn.model_selection import KFold

        from qrc_thresher.readout import fit_ridge_cv

        rows = np.arange(50, ds.train_end)
        for record in search.configs:
            X = esn.build_esn(esn.draw_reservoir(4, 137), esn.ESNParams(**record['params'])).states(
                ds.u
            )
            scores = []
            for fit_idx, val_idx in KFold(n_splits=cfg.training.cv_folds).split(rows):
                fit_rows, val_rows = rows[fit_idx], rows[val_idx]
                model = fit_ridge_cv(X[fit_rows], ds.targets[fit_rows], cfg.training.ridge_alphas,
                                     cfg.training.cv_folds)
                pred = model.predict(X[val_rows])
                scores.append(memory_capacity(pred[:, 1:], ds.targets[val_rows][:, 1:]))
            assert record['score'] == pytest.approx(float(np.mean(scores)), abs=1e-10)
        narma = esn.tune_esn(ds.u, ds.targets[:, 0], ds.train_end, esn.draw_reservoir(4, 137),
                             grid, 50, cfg.training.ridge_alphas, cfg.training.cv_folds,
                             task='narma')
        assert narma.score_name == 'nrmse' and narma.higher_is_better is False


class TestRKS:
    def test_std_of_w_is_sigma_over_sqrt_d_and_features_use_the_window(self) -> None:
        from qrc_thresher.baselines.random_features import build_rks_params, extract_rks_features

        for sigma, d in ((0.5, 1), (1.0, 2), (3.0, 4)):
            params = build_rks_params(n_features=20000, sigma=sigma, rng=np.random.default_rng(3),
                                      window=d)
            assert params.W.shape == (20000, d) and params.b.shape == (20000,)
            assert (params.sigma, params.window, params.n_features) == (sigma, d, 20000)
            assert float(params.W.std()) == pytest.approx(sigma / math.sqrt(d), rel=0.02)
        params = build_rks_params(n_features=6, sigma=1.0, rng=np.random.default_rng(3), window=3)
        u = np.random.default_rng(5).uniform(-1.0, 1.0, size=8)
        X = np.zeros((8, 3))
        for j in range(3):
            X[j:, j] = u[: 8 - j]  # column j is u_{t-j}, 0 where t - j < 0
        expected = np.cos(X @ params.W.T + params.b)
        assert np.array_equal(extract_rks_features(u, params), expected)

    def test_at_d_one_the_stream_positions_of_the_d010_draw_are_unchanged(self) -> None:
        from qrc_thresher.baselines.random_features import build_rks_params

        F = 10
        new = build_rks_params(n_features=F, sigma=1.0, rng=np.random.default_rng([137, 3]),
                               window=1)
        rng = np.random.default_rng([137, 3])
        old_W = rng.normal(0.0, 1.0 / F, size=F)  # D010's draw: N(0, (sigma / F)^2), shape (F,)
        old_b = rng.uniform(0.0, 2.0 * np.pi, size=F)
        np.testing.assert_allclose(new.W.ravel(), old_W * F, rtol=1e-12)
        assert np.array_equal(new.b, old_b)

    def test_rks_from_config_and_the_hash_cover_the_window(self) -> None:
        wq = _wq()
        cfg = load_config(DEFAULT_CONFIG)
        base = wq.rks_from_config(cfg, 137)
        assert (base.sigma, base.window, base.n_features) == (1.0, 1, 4)
        tuned = wq.rks_from_config(cfg, 137, sigma=2.0, window=2)
        assert (tuned.sigma, tuned.window) == (2.0, 2) and tuned.W.shape == (4, 2)
        assert wq.rks_circuit_hash(base) != wq.rks_circuit_hash(tuned)
        h = hashlib.sha256()
        h.update(f'rks:n_features=4,sigma={2.0!r},window=2:'.encode())
        h.update(np.ascontiguousarray(tuned.W).tobytes())
        h.update(np.ascontiguousarray(tuned.b).tobytes())
        assert wq.rks_circuit_hash(tuned) == h.hexdigest()
