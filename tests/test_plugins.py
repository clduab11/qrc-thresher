"""Tests for plugin registry and builtin registrations."""

from __future__ import annotations

from qrc_thresher.plugins.registry import PluginRegistry, create_registry_hub


class TestPluginRegistry:
    """Core registry behavior tests."""

    def test_register_and_get(self) -> None:
        reg = PluginRegistry('example.group')
        fn = lambda: None  # noqa: E731
        reg.register('fn', fn)
        assert reg.get('fn') is fn

    def test_duplicate_register_raises(self) -> None:
        reg = PluginRegistry('example.group')
        reg.register('x', object())
        try:
            reg.register('x', object())
            assert False, 'expected ValueError for duplicate plugin name'
        except ValueError:
            assert True


class TestRegistryHub:
    """Hub-level registration tests."""

    def test_builtin_hub_has_core_groups(self) -> None:
        hub = create_registry_hub(load_builtin=True, load_entry_points_flag=False)
        assert 'stm' in hub.tasks.available()
        assert 'parity' in hub.tasks.available()
        assert 'narma' in hub.tasks.available()
        assert 'G0' in hub.gates.available()
        assert 'stateful' in hub.reservoirs.available()
        assert 'stm_mc' in hub.viz.available()
        assert 'depolarizing' in hub.noise_models.available()
