"""Plugin registry and entry-point discovery for qrc_thresher."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any, Dict, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class PluginRegistry(Generic[T]):
    """Name -> plugin object registry with entry-point loading support."""

    group: str
    plugins: Dict[str, T] = field(default_factory=dict)

    def register(self, name: str, plugin: T, override: bool = False) -> None:
        """Register a plugin object under a stable name."""
        if not override and name in self.plugins:
            raise ValueError(f'Plugin "{name}" already registered in {self.group}')
        self.plugins[name] = plugin

    def get(self, name: str) -> T:
        """Get plugin by name or raise KeyError."""
        try:
            return self.plugins[name]
        except KeyError as exc:
            available = ', '.join(sorted(self.plugins))
            raise KeyError(
                f'Plugin "{name}" not found in {self.group}. Available: [{available}]'
            ) from exc

    def has(self, name: str) -> bool:
        """Return True when plugin exists."""
        return name in self.plugins

    def available(self) -> list[str]:
        """Return sorted list of registered plugin names."""
        return sorted(self.plugins)

    def load_entry_points(self) -> None:
        """Load plugins from Python entry points for this registry's group."""
        for ep in _entry_points_for_group(self.group):
            try:
                plugin = ep.load()
                self.register(ep.name, plugin, override=False)
                logger.debug('Loaded plugin %s from entry point %s', ep.name, ep.value)
            except ValueError:
                logger.debug('Skipping duplicate plugin %s in group %s', ep.name, self.group)
            except Exception as exc:
                logger.warning(
                    'Failed loading plugin %s from group %s: %s',
                    ep.name,
                    self.group,
                    exc,
                )


@dataclass
class RegistryHub:
    """Collection of plugin registries used by qrc_thresher."""

    tasks: PluginRegistry[Any] = field(default_factory=lambda: PluginRegistry('qrc_thresher.tasks'))
    reservoirs: PluginRegistry[Any] = field(
        default_factory=lambda: PluginRegistry('qrc_thresher.reservoirs')
    )
    baselines: PluginRegistry[Any] = field(
        default_factory=lambda: PluginRegistry('qrc_thresher.baselines')
    )
    gates: PluginRegistry[Any] = field(default_factory=lambda: PluginRegistry('qrc_thresher.gates'))
    viz: PluginRegistry[Any] = field(default_factory=lambda: PluginRegistry('qrc_thresher.viz'))
    noise_models: PluginRegistry[Any] = field(
        default_factory=lambda: PluginRegistry('qrc_thresher.noise_models')
    )

    def load_entry_points(self) -> None:
        """Load all registries from entry points."""
        self.tasks.load_entry_points()
        self.reservoirs.load_entry_points()
        self.baselines.load_entry_points()
        self.gates.load_entry_points()
        self.viz.load_entry_points()
        self.noise_models.load_entry_points()


def _entry_points_for_group(group: str) -> list[Any]:
    """Compatibility helper for importlib.metadata.entry_points APIs."""
    eps = entry_points()
    if hasattr(eps, 'select'):
        return list(eps.select(group=group))
    return list(eps.get(group, []))


_DEFAULT_HUB: RegistryHub | None = None


def create_registry_hub(
    load_builtin: bool = True,
    load_entry_points_flag: bool = True,
) -> RegistryHub:
    """Create a fresh registry hub and optionally populate it."""
    hub = RegistryHub()

    if load_builtin:
        from qrc_thresher.plugins import register_builtin_plugins

        register_builtin_plugins(hub)

    if load_entry_points_flag:
        hub.load_entry_points()

    return hub


def get_registry_hub(refresh: bool = False) -> RegistryHub:
    """Return singleton registry hub used by the application."""
    global _DEFAULT_HUB
    if _DEFAULT_HUB is None or refresh:
        _DEFAULT_HUB = create_registry_hub(load_builtin=True, load_entry_points_flag=True)
    return _DEFAULT_HUB


def plugin_inventory() -> dict[str, list[str]]:
    """Return current plugin inventory by extension group."""
    hub = get_registry_hub(refresh=False)
    return {
        'tasks': hub.tasks.available(),
        'reservoirs': hub.reservoirs.available(),
        'baselines': hub.baselines.available(),
        'gates': hub.gates.available(),
        'viz': hub.viz.available(),
        'noise_models': hub.noise_models.available(),
    }
