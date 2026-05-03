"""Builtin plugin registrations for qrc_thresher."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

from qrc_thresher.plugins.registry import RegistryHub


def _lazy_plugin(module_name: str, attr_name: str) -> Callable[..., Any]:
    """Create a lazy-loading plugin callable.

    The target module is imported only when the plugin is invoked, so plugin
    registration remains robust even when optional runtime dependencies are
    not installed in the current environment.
    """

    def _plugin(*args: Any, **kwargs: Any) -> Any:
        module = import_module(module_name)
        fn = getattr(module, attr_name)
        return fn(*args, **kwargs)

    _plugin.__name__ = attr_name
    _plugin.__qualname__ = attr_name
    return _plugin


def register_builtin_plugins(hub: RegistryHub) -> None:
    """Register first-party implementations into the plugin hub."""
    # Tasks
    hub.tasks.register('stm', _lazy_plugin('qrc_thresher.tasks.stm', 'generate_stm'))
    hub.tasks.register(
        'parity',
        _lazy_plugin('qrc_thresher.tasks.temporal_parity', 'generate_parity'),
    )
    hub.tasks.register('narma', _lazy_plugin('qrc_thresher.tasks.narma10', 'generate_narma10'))

    # Reservoirs
    hub.reservoirs.register(
        'pennylane',
        _lazy_plugin('qrc_thresher.reservoirs.pennylane_qrc', 'extract_features'),
    )
    hub.reservoirs.register(
        'stateful',
        _lazy_plugin('qrc_thresher.reservoirs.stateful_qrc', 'extract_features_stateful'),
    )
    hub.reservoirs.register(
        'phase_random',
        _lazy_plugin('qrc_thresher.reservoirs.ablations', 'extract_features_phase_random'),
    )
    hub.reservoirs.register(
        'no_entangle',
        _lazy_plugin('qrc_thresher.reservoirs.ablations', 'extract_features_no_entangle'),
    )
    hub.reservoirs.register(
        'haar',
        _lazy_plugin('qrc_thresher.reservoirs.ablations', 'extract_features_haar'),
    )

    # Baselines
    hub.baselines.register(
        'esn_grid_search',
        _lazy_plugin('qrc_thresher.baselines.esn', 'grid_search_esn'),
    )
    hub.baselines.register(
        'esn_predict',
        _lazy_plugin('qrc_thresher.baselines.esn', 'train_predict_esn'),
    )
    hub.baselines.register(
        'random_features_extract',
        _lazy_plugin('qrc_thresher.baselines.random_features', 'extract_rks_features'),
    )
    hub.baselines.register(
        'random_features_train',
        _lazy_plugin('qrc_thresher.baselines.random_features', 'train_rks'),
    )
    hub.baselines.register('gru', _lazy_plugin('qrc_thresher.baselines.gru', 'train_gru'))

    # Gates
    hub.gates.register('G0', _lazy_plugin('qrc_thresher.commands.gate', '_evaluate_gate_g0'))
    hub.gates.register('G0.5', _lazy_plugin('qrc_thresher.commands.gate', '_evaluate_gate_g05'))
    hub.gates.register('G1', _lazy_plugin('qrc_thresher.commands.gate', '_evaluate_gate_g1'))
    hub.gates.register('G2', _lazy_plugin('qrc_thresher.commands.gate', '_evaluate_gate_g2'))
    hub.gates.register('G2.5', _lazy_plugin('qrc_thresher.commands.gate', '_evaluate_gate_g25'))
    hub.gates.register('G3', _lazy_plugin('qrc_thresher.commands.gate', '_evaluate_gate_g3'))
    hub.gates.register('G4', _lazy_plugin('qrc_thresher.commands.gate', '_evaluate_gate_g4'))
    hub.gates.register('G5', _lazy_plugin('qrc_thresher.commands.gate', '_evaluate_gate_g5'))
    hub.gates.register('G6', _lazy_plugin('qrc_thresher.commands.gate', '_evaluate_gate_g6'))
    hub.gates.register('G7', _lazy_plugin('qrc_thresher.commands.gate', '_evaluate_gate_g7'))

    # Visualizations
    hub.viz.register('stm_mc', _lazy_plugin('qrc_thresher.viz.plots', 'plot_stm_mc'))
    hub.viz.register('comparison', _lazy_plugin('qrc_thresher.viz.plots', 'plot_comparison'))
    hub.viz.register(
        'stm_delay_heatmap',
        _lazy_plugin('qrc_thresher.viz.plots', 'plot_stm_delay_heatmap'),
    )
    hub.viz.register(
        'runtime_breakdown',
        _lazy_plugin('qrc_thresher.viz.plots', 'plot_runtime_breakdown'),
    )
    hub.viz.register(
        'gate_decision_tree',
        _lazy_plugin('qrc_thresher.viz.plots', 'plot_gate_decision_tree'),
    )
    hub.viz.register(
        'metric_correlation',
        _lazy_plugin('qrc_thresher.viz.plots', 'plot_metric_correlation_matrix'),
    )

    # Noise models
    hub.noise_models.register(
        'depolarizing',
        _lazy_plugin('qrc_thresher.reservoirs.noise_models', 'build_depolarizing_noise_model'),
    )
    hub.noise_models.register(
        'relaxation',
        _lazy_plugin('qrc_thresher.reservoirs.noise_models', 'build_relaxation_noise_model'),
    )
