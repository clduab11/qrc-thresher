"""One washout for every model (defect D16; docs/DECISIONS.md D012).

``training.washout`` replaces ``baseline.esn_washout`` (refused). Every writer drops rows
[0, washout) from training and tuning: the engine (three tasks), the run command (three), the
ablation command, the baseline command (stm, parity, narma), the tuner and the ESN smoke check.
Proof: the rows [0, washout) of every task's targets are marked with a sentinel before the writer
runs, and a spy on readout.fit_ridge_cv, readout.fit_ridge_cv_batched, every module-level import
of them and sklearn's RidgeCV.fit shows that no fitted target contains the sentinel, so no
zero-padded STM row, no NARMA-10 row 0..9 and no pre-window parity row reaches any fit.
"""

from __future__ import annotations

import dataclasses
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError
from sklearn.linear_model import RidgeCV

from qrc_thresher.config import AlphaLiteConfig, load_config

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG = REPO_ROOT / 'configs' / 'alpha_lite.yaml'
SENTINEL = 12345.0
WASHOUT = 20  # the tiny configs: T = 100, train_end = 70, 50 training rows after the washout
TASKS = ['stm', 'parity', 'narma']
TINY_TUNING = {
    'qrc': {'depth': [1], 'window': [1, 2], 'encoding_scale': [3.141592653589793]},
    'esn': {'spectral_radius': [0.9], 'input_scaling': [0.1, 1.0], 'leak_rate': [1.0]},
    'rks': {'sigma': [0.5, 2.0], 'window': [1]},
}


def _raw() -> dict:
    return yaml.safe_load(DEFAULT_CONFIG.read_text(encoding='utf-8'))


def _tiny(tmp_path: Path, task: str, *, tuning: bool = False, n_seeds: int = 1) -> Path:
    raw = _raw()
    raw['experiment_name'] = f'tiny_washout_{task}'
    raw['task'].update({'name': task, 'length': 100, 'delay_max': 5, 'parity_window': 2})
    raw['reservoir'].update({'n_qubits': 2, 'depth': 1, 'readout': 'z_only', 'window': 2})
    raw['training']['washout'] = WASHOUT
    raw['seeds']['n_seeds'] = n_seeds
    raw['baseline']['enabled'] = ['esn', 'random_features']
    if tuning:
        raw['tuning'] = TINY_TUNING
        raw['baseline'].pop('esn_grid')
    else:
        raw['baseline']['esn_grid'] = TINY_TUNING['esn']
    path = tmp_path / 'configs' / f'{task}.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw), encoding='utf-8')
    return path


def _mark(ds, washout: int):
    """The dataset with rows [0, washout) of its targets replaced by the sentinel."""
    targets = np.array(ds.targets, dtype=np.float64, copy=True)
    targets[:washout] = SENTINEL
    return dataclasses.replace(ds, targets=targets)


def _mark_task_generators(monkeypatch, washout: int) -> None:
    from qrc_thresher.tasks import narma10, stm, temporal_parity

    for module, name in ((stm, 'generate_stm'), (temporal_parity, 'generate_parity'),
                         (narma10, 'generate_narma10')):
        original = getattr(module, name)

        def marked(*args, _original=original, **kwargs):
            return _mark(_original(*args, **kwargs), washout)

        monkeypatch.setattr(module, name, marked)
    # Module-level imports of the generators (the health check imports inside functions).
    for mod in list(sys.modules.values()):
        if not (getattr(mod, '__name__', '') or '').startswith('qrc_thresher.'):
            continue
        for name, source in (('generate_stm', stm), ('generate_parity', temporal_parity),
                             ('generate_narma10', narma10)):
            if hasattr(mod, name) and mod is not source:
                monkeypatch.setattr(mod, name, getattr(source, name))


def _spy_on_fits(monkeypatch) -> list:
    """Record every target array that reaches a readout fit, on every path."""
    from qrc_thresher import readout

    seen: list = []
    original_cv, original_batched = readout.fit_ridge_cv, readout.fit_ridge_cv_batched

    def spy_cv(X, y, *args, **kwargs):
        seen.append(('fit_ridge_cv', np.asarray(y, dtype=np.float64)))
        return original_cv(X, y, *args, **kwargs)

    def spy_batched(X, Y, *args, **kwargs):
        seen.append(('fit_ridge_cv_batched', np.asarray(Y, dtype=np.float64)))
        return original_batched(X, Y, *args, **kwargs)

    original_fit = RidgeCV.fit

    def spy_fit(self, X, y, *args, **kwargs):
        seen.append(('RidgeCV.fit', np.asarray(y, dtype=np.float64)))
        return original_fit(self, X, y, *args, **kwargs)

    monkeypatch.setattr(readout, 'fit_ridge_cv', spy_cv)
    monkeypatch.setattr(readout, 'fit_ridge_cv_batched', spy_batched)
    monkeypatch.setattr(RidgeCV, 'fit', spy_fit)
    for mod in list(sys.modules.values()):
        if not (getattr(mod, '__name__', '') or '').startswith('qrc_thresher.'):
            continue
        if mod is readout:
            continue
        if getattr(mod, 'fit_ridge_cv', None) is original_cv:
            monkeypatch.setattr(mod, 'fit_ridge_cv', spy_cv)
        if getattr(mod, 'fit_ridge_cv_batched', None) is original_batched:
            monkeypatch.setattr(mod, 'fit_ridge_cv_batched', spy_batched)
    return seen


def _assert_no_washout_row_was_fitted(seen: list, expected_fits: int) -> None:
    assert len(seen) >= expected_fits, [name for name, _ in seen]
    for name, y in seen:
        assert not np.any(y == SENTINEL), f'{name} fitted a target row below the washout'
        assert np.all(np.isfinite(y)), name


def _import_writers() -> None:
    for name in ('qrc_thresher.engine', 'qrc_thresher.commands.run',
                 'qrc_thresher.commands.ablation', 'qrc_thresher.commands.baseline',
                 'qrc_thresher.gates.g07', 'qrc_thresher.baselines.esn',
                 'qrc_thresher.proof.benchmark_health', 'qrc_thresher.tuning'):
        importlib.import_module(name)


class TestConfig:
    def test_training_washout_is_the_registered_value_and_esn_washout_is_refused(self) -> None:
        cfg = load_config(DEFAULT_CONFIG)
        assert cfg.training.washout == 50
        raw = _raw()
        assert 'esn_washout' not in raw['baseline']
        raw['baseline']['esn_washout'] = 50
        with pytest.raises(ValidationError, match='esn_washout'):
            AlphaLiteConfig.model_validate(raw)
        assert not hasattr(cfg.baseline, 'esn_washout')

    @pytest.mark.parametrize(
        'change, match',
        [
            ({'training': {'washout': 19}}, 'delay_max'),          # K = 20
            ({'task': {'parity_window': 60}, 'training': {'washout': 50}}, 'parity_window'),
            ({'task': {'name': 'narma'}, 'training': {'washout': 9}}, 'NARMA'),
            ({'training': {'washout': 345}}, 'cv_folds'),          # 350 - 345 < 2 * 5
        ],
    )
    def test_the_washout_must_cover_the_padded_rows_and_leave_training_rows(
        self, change, match
    ) -> None:
        raw = _raw()
        for section, values in change.items():
            raw[section].update(values)
        with pytest.raises(ValidationError, match=match):
            AlphaLiteConfig.model_validate(raw)

    def test_the_washout_must_cover_the_window(self) -> None:
        # Every task can be run from one config (D013), so the NARMA-10 minimum of 10 always
        # applies alongside window - 1 (CP4b washout ruling: structural minima only, all of them).
        raw = _raw()
        raw['reservoir'].update({'n_qubits': 4, 'window': 4})
        raw['task']['delay_max'] = 1
        raw['training']['washout'] = 2
        with pytest.raises(ValidationError, match='window'):
            AlphaLiteConfig.model_validate(raw)
        raw['training']['washout'] = 9
        with pytest.raises(ValidationError, match='NARMA'):
            AlphaLiteConfig.model_validate(raw)
        raw['training']['washout'] = 10
        assert AlphaLiteConfig.model_validate(raw).training.washout == 10


class TestWriters:
    @pytest.mark.parametrize('task', TASKS)
    def test_the_engine(self, task, tmp_path, monkeypatch) -> None:
        from qrc_thresher.engine import ParallelRunner

        _import_writers()
        cfg_path = _tiny(tmp_path, task)
        monkeypatch.chdir(tmp_path)
        _mark_task_generators(monkeypatch, WASHOUT)
        seen = _spy_on_fits(monkeypatch)
        cfg = load_config(cfg_path)
        manifests = ParallelRunner(config=cfg, max_workers=1).run_seeds(task, config_path=cfg_path)
        assert [m.success for m in manifests] == [True], [m.failure_reason for m in manifests]
        _assert_no_washout_row_was_fitted(seen, expected_fits=1)

    @pytest.mark.parametrize('task', TASKS)
    def test_the_run_command(self, task, tmp_path, monkeypatch) -> None:
        from qrc_thresher.commands.run import run_handler

        _import_writers()
        cfg_path = _tiny(tmp_path, task)
        monkeypatch.chdir(tmp_path)
        _mark_task_generators(monkeypatch, WASHOUT)
        seen = _spy_on_fits(monkeypatch)
        assert run_handler(task, str(cfg_path)) == 0
        _assert_no_washout_row_was_fitted(seen, expected_fits=1)

    @pytest.mark.parametrize('name', ['no_entangle', 'phase_random', 'haar'])
    def test_the_ablation_command(self, name, tmp_path, monkeypatch) -> None:
        from qrc_thresher.commands.ablation import ablation_handler

        _import_writers()
        cfg_path = _tiny(tmp_path, 'parity')
        monkeypatch.chdir(tmp_path)
        _mark_task_generators(monkeypatch, WASHOUT)
        seen = _spy_on_fits(monkeypatch)
        assert ablation_handler(name, 'parity', str(cfg_path)) == 0
        _assert_no_washout_row_was_fitted(seen, expected_fits=1)

    @pytest.mark.parametrize('task', TASKS)
    def test_the_baseline_command_tunes_and_fits_after_the_washout(
        self, task, tmp_path, monkeypatch
    ) -> None:
        from qrc_thresher.commands.baseline import run_baselines

        _import_writers()
        cfg_path = _tiny(tmp_path, task)
        monkeypatch.chdir(tmp_path)
        _mark_task_generators(monkeypatch, WASHOUT)
        seen = _spy_on_fits(monkeypatch)
        manifests = run_baselines(load_config(cfg_path), task, cfg_path)
        assert manifests and all(m.success for m in manifests), [
            m.failure_reason for m in manifests
        ]
        # In-line ESN tuning (2 configurations x 5 blocks) plus the deployed fit.
        _assert_no_washout_row_was_fitted(seen, expected_fits=11)

    @pytest.mark.parametrize('task', TASKS)
    def test_the_tuner(self, task, tmp_path, monkeypatch) -> None:
        from qrc_thresher import tuning

        _import_writers()
        cfg_path = _tiny(tmp_path, task, tuning=True)
        monkeypatch.chdir(tmp_path)
        _mark_task_generators(monkeypatch, WASHOUT)
        seen = _spy_on_fits(monkeypatch)
        record = tuning.tune_config(load_config(cfg_path), task, cfg_path)
        assert record['washout'] == WASHOUT
        # Three models x 2 configurations x 5 validation blocks.
        _assert_no_washout_row_was_fitted(seen, expected_fits=30)

    def test_the_esn_smoke_check_reads_the_config_washout(self, monkeypatch) -> None:
        from qrc_thresher.proof import benchmark_health

        _import_writers()
        _mark_task_generators(monkeypatch, 50)
        seen = _spy_on_fits(monkeypatch)
        report = benchmark_health.check_esn_smoke()
        assert report['status'] == 'PASS', report
        assert report['washout'] == 50
        _assert_no_washout_row_was_fitted(seen, expected_fits=1)
        source = Path(benchmark_health.__file__).read_text(encoding='utf-8')
        assert 'training.washout' in source  # no hard-coded 50
