"""Plugin SDK and registry utilities for qrc_thresher."""

from qrc_thresher.plugins.protocols import (
    BaselinePlugin,
    GatePlugin,
    NoiseModelPlugin,
    ReservoirPlugin,
    TaskPlugin,
    VizPlugin,
)
from qrc_thresher.plugins.registry import (
    PluginRegistry,
    RegistryHub,
    create_registry_hub,
    get_registry_hub,
)


def register_builtin_plugins(hub: RegistryHub) -> None:
    """Lazily register builtins to avoid importing optional deps at import-time."""
    from qrc_thresher.plugins.builtin import register_builtin_plugins as _register

    _register(hub)

__all__ = [
    'TaskPlugin',
    'ReservoirPlugin',
    'BaselinePlugin',
    'GatePlugin',
    'VizPlugin',
    'NoiseModelPlugin',
    'PluginRegistry',
    'RegistryHub',
    'register_builtin_plugins',
    'create_registry_hub',
    'get_registry_hub',
]
