"""G0.7 wiring for the windowed reservoir (docs/DECISIONS.md D010).

Only the model plumbing changes: `qrc-thresher gate` takes --config (default
configs/alpha_lite.yaml) and the model no_entangle, the G0.7 feature map is built through the
config helper, and the model details record the window and the ablation. The protocol file,
evaluate(), the clause and permutation functions, test_preregistration.py and
test_gate_g07.py are untouched. configs/windowed_w2.yaml and windowed_w4.yaml differ from
alpha_lite.yaml only in reservoir.window and experiment_name.

These tests never run a full G0.7 evaluation: the evaluator is replaced by a stub that
records what it was given.
"""

from __future__ import annotations

import importlib
import inspect
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from qrc_thresher.config import load_config
from qrc_thresher.reservoirs.pennylane_qrc import build_reservoir_params, extract_features

REPO_ROOT = Path(__file__).parent.parent
CONFIGS = REPO_ROOT / 'configs'
DEFAULT_CONFIG = CONFIGS / 'alpha_lite.yaml'
WINDOWED = {'windowed_w2': 2, 'windowed_w4': 4}


def _wq():
    """Import the windowed reservoir module (absent until CP3b)."""
    return importlib.import_module('qrc_thresher.reservoirs.windowed_qrc')


def _g07():
    return importlib.import_module('qrc_thresher.gates.g07')


class TestGateCommand:
    def test_takes_a_config_and_the_no_entangle_model(self, tmp_path, monkeypatch) -> None:
        from click.testing import CliRunner

        from qrc_thresher.cli import cli

        gate = importlib.import_module('qrc_thresher.commands.gate')
        signature = inspect.signature(gate._evaluate_gate_g07)
        calls = []

        def stub(*args, **kwargs):
            calls.append(signature.bind(*args, **kwargs).arguments)
            # The six evidence keys the real _evaluate_gate_g07 returns.
            evidence = {
                'message': 'stub',
                'stm_clause': 'PASS',
                'parity_clause': 'PASS',
                'measurement_label': 'exact (oracle upper bound)',
                'json': 'stub.json',
                'figure': 'stub.png',
            }
            return 'PASS', evidence, []

        monkeypatch.setattr(gate, '_evaluate_gate_g07', stub)
        monkeypatch.chdir(tmp_path)  # nothing is written under the repo's results/
        (tmp_path / 'configs').mkdir()
        local_default = tmp_path / 'configs' / 'alpha_lite.yaml'
        shutil.copy(DEFAULT_CONFIG, local_default)

        runner = CliRunner()
        w2 = CONFIGS / 'windowed_w2.yaml'
        given = runner.invoke(cli, ['gate', 'G0.7', '--config', str(w2), '--model', 'no_entangle'])
        assert given.exit_code == 0, given.output
        default = runner.invoke(cli, ['gate', 'G0.7'])
        assert default.exit_code == 0, default.output

        assert len(calls) == 2
        assert Path(calls[0]['config_path']).resolve() == w2.resolve()
        assert calls[0]['model'] == 'no_entangle'
        assert Path(calls[1]['config_path']).resolve() == local_default.resolve()
        assert calls[1].get('model', 'pennylane_qrc') == 'pennylane_qrc'


class TestModelPlumbing:
    def test_the_models_are_the_reservoir_its_ablation_the_tuned_design_and_the_esns(self) -> None:
        assert set(_g07().MODELS) == {
            'pennylane_qrc', 'no_entangle', 'tuned_qrc', 'esn_linear', 'esn_nonlinear'
        }

    @pytest.mark.parametrize(
        'model, ablation', [('pennylane_qrc', None), ('no_entangle', 'no_entangle')]
    )
    def test_details_carry_the_window_and_the_ablation(self, model, ablation, monkeypatch) -> None:
        g07, wq = _g07(), _wq()
        cfg = load_config(CONFIGS / 'windowed_w2.yaml')
        captured = {}

        def capture(feature_map, protocol, seed_pairs, **kwargs):
            captured.update(feature_map=feature_map, **kwargs)
            return {}

        monkeypatch.setattr(g07, 'evaluate', capture)
        g07.evaluate_config(cfg, model=model)
        assert captured['model_name'] == model
        assert captured['model_details']['window'] == 2
        assert captured['model_details']['ablation'] == ablation
        u = np.random.default_rng(0).uniform(-1.0, 1.0, size=6)
        expected = wq.reservoir_from_config(cfg, 137, ablation=ablation).features(u)
        assert np.array_equal(captured['feature_map'](u, 137), expected)

    def test_the_default_model_is_todays_reservoir(self) -> None:
        cfg = load_config(DEFAULT_CONFIG)
        feature_map = _g07().qrc_feature_map(cfg)
        u = np.random.default_rng(0).uniform(-1.0, 1.0, size=6)
        params = build_reservoir_params(
            n_qubits=4, depth=3, readout='z_only', backend='default.qubit',
            rng=np.random.default_rng(137),
        )
        assert np.array_equal(feature_map(u, 137), extract_features(u, params))


class TestWindowedConfigs:
    @pytest.mark.parametrize('name, window', sorted(WINDOWED.items()))
    def test_differ_from_alpha_lite_only_in_window_and_name(self, name, window) -> None:
        base = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding='utf-8'))
        other = yaml.safe_load((CONFIGS / f'{name}.yaml').read_text(encoding='utf-8'))
        assert base['reservoir']['window'] == 1
        assert other['reservoir']['window'] == window
        assert other['experiment_name'] != base['experiment_name']
        for raw in (base, other):
            raw.pop('experiment_name')
            raw['reservoir'].pop('window')
        assert other == base
        assert load_config(CONFIGS / f'{name}.yaml').reservoir.window == window


def _stub_record(cfg, pairs):
    """A tuning record with one design per pair, as `tune stm` writes it (D011)."""
    import math

    from qrc_thresher.proof.run_manifest import _config_hash

    designs = {}
    for i, (task_seed, reservoir_seed) in enumerate(pairs):
        depth, window, scale = [(1, 1, math.pi), (2, 2, math.pi / 2), (3, 2, math.pi / 4)][i % 3]
        designs[f'{task_seed}/{reservoir_seed}'] = {
            'task_seed': task_seed, 'reservoir_seed': reservoir_seed, 'depth': depth,
            'window': window, 'encoding_scale': scale, 'circuit_hash': 'h' * 64,
            'n_configs': 60, 'n_validation_evals': 300, 'winner_index': 0, 'configs': [],
        }
    return {'config_hash': _config_hash(CONFIGS / 'comparative.yaml'), 'task': 'stm',
            'sweep_id': '20260923T000000000000Z', 'washout': 50,
            'reservoir': {'n_qubits': cfg.reservoir.n_qubits, 'readout': cfg.reservoir.readout,
                          'backend': cfg.reservoir.backend},
            'selection_metric': {'name': 'stm_memory', 'higher_is_better': True},
            'grids': {}, 'qrc': designs, 'esn': {}, 'rks': {}}


class TestTunedModel:
    """G0.7 v1 on the tuned design (D011): model tuned_qrc, design_STM(pair) from the record."""

    def test_the_feature_map_is_design_stm_of_the_pair(self, monkeypatch) -> None:
        import copy

        g07, wq = _g07(), _wq()
        cfg = load_config(DEFAULT_CONFIG)  # alpha_lite's 3 pairs
        pairs = [(42 + i, 137 + i) for i in range(3)]
        record = _stub_record(cfg, pairs)
        for entry in record['qrc'].values():  # the stub's hashes become the real ones
            raw = cfg.model_dump()
            raw['reservoir'].update({k: entry[k] for k in ('depth', 'window', 'encoding_scale')})
            from qrc_thresher.config import AlphaLiteConfig

            entry['circuit_hash'] = wq.reservoir_from_config(
                AlphaLiteConfig.model_validate(raw), entry['reservoir_seed']
            ).circuit_hash
        captured = {}

        def capture(feature_map, protocol, seed_pairs, **kwargs):
            captured.update(feature_map=feature_map, seed_pairs=list(seed_pairs), **kwargs)
            return {}

        monkeypatch.setattr(g07, 'evaluate', capture)
        g07.evaluate_config(cfg, model='tuned_qrc', tuning_record=copy.deepcopy(record))
        assert captured['model_name'] == 'tuned_qrc'
        assert [tuple(p) for p in captured['seed_pairs']] == pairs
        details = captured['model_details']
        assert details['sweep_id'] == record['sweep_id']
        assert details['tuning_config_hash'] == record['config_hash']
        assert details['designs'] == {
            key: {k: entry[k] for k in ('depth', 'window', 'encoding_scale', 'circuit_hash')}
            for key, entry in record['qrc'].items()
        }
        u = np.random.default_rng(0).uniform(-1.0, 1.0, size=6)
        for entry in record['qrc'].values():
            raw = cfg.model_dump()
            raw['reservoir'].update({k: entry[k] for k in ('depth', 'window', 'encoding_scale')})
            from qrc_thresher.config import AlphaLiteConfig

            expected = wq.reservoir_from_config(
                AlphaLiteConfig.model_validate(raw), entry['reservoir_seed']
            ).features(u)
            assert np.array_equal(captured['feature_map'](u, entry['reservoir_seed']), expected)

    def test_a_record_missing_a_pair_or_from_another_reservoir_is_refused(self) -> None:
        g07 = _g07()
        cfg = load_config(DEFAULT_CONFIG)
        short = _stub_record(cfg, [(42, 137), (43, 138)])
        with pytest.raises(ValueError, match='44'):
            g07.evaluate_config(cfg, model='tuned_qrc', tuning_record=short)
        with pytest.raises(ValueError, match='tuning_record'):
            g07.evaluate_config(cfg, model='tuned_qrc')
        other = load_config(CONFIGS / 'windowed_w2.yaml')
        raw = other.model_dump()
        raw['reservoir']['n_qubits'] = 5
        from qrc_thresher.config import AlphaLiteConfig

        five = AlphaLiteConfig.model_validate(raw)
        record = _stub_record(cfg, [(42 + i, 137 + i) for i in range(3)])  # n_qubits 4
        with pytest.raises(ValueError, match='n_qubits'):
            g07.evaluate_config(five, model='tuned_qrc', tuning_record=record)

    def test_the_gate_command_takes_a_tuning_config(self, tmp_path, monkeypatch) -> None:
        from click.testing import CliRunner

        from qrc_thresher.cli import cli

        gate = importlib.import_module('qrc_thresher.commands.gate')
        signature = inspect.signature(gate._evaluate_gate_g07)
        calls = []

        def stub(*args, **kwargs):
            calls.append(signature.bind(*args, **kwargs).arguments)
            evidence = {'message': 'stub', 'stm_clause': 'PASS', 'parity_clause': 'PASS',
                        'measurement_label': 'exact (oracle upper bound)', 'json': 'stub.json',
                        'figure': 'stub.png'}
            return 'PASS', evidence, []

        monkeypatch.setattr(gate, '_evaluate_gate_g07', stub)
        monkeypatch.chdir(tmp_path)
        comparative = CONFIGS / 'comparative.yaml'
        result = CliRunner().invoke(cli, ['gate', 'G0.7', '--config', str(DEFAULT_CONFIG),
                                          '--model', 'tuned_qrc', '--tuning-config',
                                          str(comparative)])
        assert result.exit_code == 0, result.output
        assert calls[0]['model'] == 'tuned_qrc'
        assert Path(calls[0]['config_path']).resolve() == DEFAULT_CONFIG.resolve()
        assert Path(calls[0]['tuning_config']).resolve() == comparative.resolve()
