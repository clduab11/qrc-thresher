"""The stateful-to-smoothed rename (defect D2) and the GRU drop (defect D20); docs/DECISIONS.md
D018.

- ``reservoirs.smoothed_qrc.extract_features_smoothed``, the old ``extract_features_stateful`` and
  the 'smoothed' and 'stateful' plugins are all bit-identical (features and state_trace) to a
  reference built here from ``pennylane_qrc.extract_features`` with the original formula;
- the old function and the 'stateful' plugin stay as a deprecated shim that returns identical
  values and warns with a DeprecationWarning naming extract_features_smoothed on every call, not
  at import (a green guard at CP5a: importing stateful_qrc has never warned);
- README no longer calls the module memory-carrying;
- 'gru' is refused by the config and is absent from the builtin baseline plugins.
"""

from __future__ import annotations

import importlib
import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

REPO_ROOT = Path(__file__).parent.parent
README = REPO_ROOT / 'README.md'
GONE = ('memory-carrying', 'carried reservoir state', 'persistent reservoir state')


def _params_and_input():
    from qrc_thresher.reservoirs.pennylane_qrc import build_reservoir_params

    rng = np.random.default_rng(2026)
    params = build_reservoir_params(n_qubits=2, depth=1, readout='z_only',
                                    backend='default.qubit', rng=rng)
    return params, rng.uniform(-1.0, 1.0, size=7)


def _stateful_quietly(u, params, carry_depth):
    from qrc_thresher.reservoirs.stateful_qrc import extract_features_stateful

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        return extract_features_stateful(u, params, carry_depth=carry_depth)


def _reference(u, params, carry_depth):
    """HEAD's formula on the memoryless features: state[t] = mean(base[max(0, t-d):t+1]),
    features[t] = 0.5 * base[t] + 0.5 * state[t]."""
    from qrc_thresher.reservoirs.pennylane_qrc import extract_features

    base = extract_features(u, params)
    features = np.array(base, copy=True)
    state = np.zeros_like(features)
    for t in range(len(features)):
        state[t] = np.mean(base[max(0, t - carry_depth):t + 1], axis=0)
        features[t] = 0.5 * base[t] + 0.5 * state[t]
    return features, state


def _assert_equal(result, reference) -> None:
    features, state = reference
    assert np.array_equal(result.features, features)
    assert np.array_equal(result.state_trace, state)


class TestSmoothed:
    @pytest.mark.parametrize('carry_depth', [1, 3])
    def test_smoothed_output_is_bit_identical_to_the_reference(self, carry_depth) -> None:
        from qrc_thresher.reservoirs.smoothed_qrc import (
            SmoothedQRCResult,
            extract_features_smoothed,
        )

        params, u = _params_and_input()
        reference = _reference(u, params, carry_depth)
        new = extract_features_smoothed(u, params, carry_depth=carry_depth)
        assert isinstance(new, SmoothedQRCResult)
        _assert_equal(new, reference)
        _assert_equal(_stateful_quietly(u, params, carry_depth), reference)  # the shim too
        assert new.features.shape == (len(u), 2)

    def test_smoothed_refuses_carry_depth_below_one(self) -> None:
        from qrc_thresher.reservoirs.smoothed_qrc import extract_features_smoothed

        with pytest.raises(ValueError, match='carry_depth'):
            extract_features_smoothed(u=[], params=None, carry_depth=0)  # type: ignore[arg-type]

    def test_the_smoothed_and_stateful_plugins_match_the_reference(self) -> None:
        from qrc_thresher.plugins.registry import create_registry_hub

        hub = create_registry_hub(load_builtin=True, load_entry_points_flag=False)
        assert 'smoothed' in hub.reservoirs.available()
        assert 'stateful' in hub.reservoirs.available()  # the shim stays; nothing is deleted
        params, u = _params_and_input()
        reference = _reference(u, params, 2)
        _assert_equal(hub.reservoirs.get('smoothed')(u, params, carry_depth=2), reference)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            _assert_equal(hub.reservoirs.get('stateful')(u, params, carry_depth=2), reference)

    def test_stateful_qrc_result_stays_importable(self) -> None:
        from qrc_thresher.reservoirs.smoothed_qrc import SmoothedQRCResult

        from qrc_thresher.reservoirs.stateful_qrc import StatefulQRCResult

        assert StatefulQRCResult is SmoothedQRCResult


class TestDeprecatedShim:
    def test_extract_features_stateful_warns_on_every_call_and_returns_identical_values(self):
        from qrc_thresher.reservoirs.smoothed_qrc import extract_features_smoothed

        from qrc_thresher.reservoirs.stateful_qrc import extract_features_stateful

        params, u = _params_and_input()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            first = extract_features_stateful(u, params, carry_depth=2)
            second = extract_features_stateful(u, params, carry_depth=2)
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(deprecations) == 2  # two calls give two warnings
        assert all('extract_features_smoothed' in str(w.message) for w in deprecations)
        reference = _reference(u, params, 2)
        _assert_equal(extract_features_smoothed(u, params, carry_depth=2), reference)
        for old in (first, second):
            _assert_equal(old, reference)

    def test_the_stateful_plugin_warns_on_every_call(self) -> None:
        from qrc_thresher.plugins.registry import create_registry_hub

        hub = create_registry_hub(load_builtin=True, load_entry_points_flag=False)
        params, u = _params_and_input()
        plugin = hub.reservoirs.get('stateful')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            plugin(u, params, carry_depth=1)
            plugin(u, params, carry_depth=1)
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(deprecations) == 2
        assert all('extract_features_smoothed' in str(w.message) for w in deprecations)

    def test_importing_stateful_qrc_does_not_warn(self) -> None:
        # A green guard at CP5a: the shim warns on every call, never at import.
        import qrc_thresher.reservoirs.stateful_qrc as module

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            importlib.reload(module)
        assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]


class TestDocsAndGru:
    def test_readme_no_longer_calls_the_module_memory_carrying(self) -> None:
        text = README.read_text(encoding='utf-8')
        present = [phrase for phrase in GONE if phrase in text]
        assert present == [], present

    def test_gru_is_refused_by_the_config(self) -> None:
        from qrc_thresher.config import AlphaLiteConfig

        config = REPO_ROOT / 'configs' / 'alpha_lite.yaml'
        raw = yaml.safe_load(config.read_text(encoding='utf-8'))
        raw['baseline']['enabled'] = ['esn', 'gru']
        with pytest.raises(ValidationError, match='gru'):
            AlphaLiteConfig.model_validate(raw)
        raw['baseline']['enabled'] = ['esn']
        AlphaLiteConfig.model_validate(raw)

    def test_gru_is_absent_from_the_builtin_baseline_plugins(self) -> None:
        from qrc_thresher.commands import baseline
        from qrc_thresher.plugins.registry import create_registry_hub

        hub = create_registry_hub(load_builtin=True, load_entry_points_flag=False)
        assert 'gru' not in hub.baselines.available()
        assert 'gru' not in baseline._NOT_IN_RUN_PATH
