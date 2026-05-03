"""Configuration models for qrc_thresher (pydantic v2).

All configuration flows through these validated models.
No untyped dicts cross module boundaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field


class TaskConfig(BaseModel):
    """Task configuration."""

    name: Literal['stm', 'parity', 'narma']
    length: int = Field(ge=100)
    train_frac: float = Field(gt=0, lt=1)
    delay_max: Optional[int] = None
    parity_window: Optional[int] = None


class ReservoirConfig(BaseModel):
    """Quantum reservoir configuration."""

    backend: Literal['default.qubit', 'lightning.qubit']
    n_qubits: int = Field(ge=2, le=12)
    depth: int = Field(ge=1, le=10)
    readout: Literal['z_only', 'z_and_zz'] = 'z_only'


class BaselineConfig(BaseModel):
    """Classical baseline configuration."""

    enabled: List[Literal['esn', 'random_features', 'gru']]
    esn_grid: Optional[Dict[str, List[float]]] = None
    rks_dim: Optional[int] = None


class AblationConfig(BaseModel):
    """Ablation study configuration."""

    name: Literal['phase_random', 'no_entangle', 'random_features', 'haar']


class TrainingConfig(BaseModel):
    """Training and cross-validation configuration."""

    ridge_alphas: List[float]
    cv_folds: int = Field(ge=2, le=10)


class ProofConfig(BaseModel):
    """Proof layer configuration."""

    log_entanglement: bool = True
    log_circuit_hash: bool = True
    log_artifacts: bool = False


class SeedsConfig(BaseModel):
    """Seed configuration for reproducibility."""

    task_seed: int
    reservoir_seed: int
    n_seeds: int = Field(ge=1, le=20)


class GateThresholds(BaseModel):
    """Gate threshold configuration for pre-registered effect size criteria."""

    G1_stm_mc_threshold: float = 1.0
    G1_margin_pct: float = 0.20
    G2_accuracy_threshold: float = 0.70
    G2_random_features_max: float = 0.60
    G25_effect_se: float = 1.0


class AlphaLiteConfig(BaseModel):
    """Top-level experiment configuration."""

    experiment_name: str
    task: TaskConfig
    reservoir: ReservoirConfig
    baseline: BaselineConfig
    ablation: Optional[AblationConfig] = None
    training: TrainingConfig
    proof: ProofConfig
    seeds: SeedsConfig
    gates: Optional[GateThresholds] = None


def load_config(path: Path) -> AlphaLiteConfig:
    """Load and validate config from YAML file.

    Args:
        path: Path to YAML config file.

    Returns:
        Validated AlphaLiteConfig instance.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValidationError: If config fails pydantic validation.
    """
    if not path.exists():
        raise FileNotFoundError(f'Config file not found: {path}')
    with path.open('r') as f:
        raw = yaml.safe_load(f)
    if raw.get('_overlay'):
        raise ValueError(f'Overlay-only config cannot be loaded directly: {path}')
    return AlphaLiteConfig.model_validate(raw)


def load_config_with_overlays(
    base_path: Path,
    overlays: Optional[List[Path]] = None,
) -> AlphaLiteConfig:
    """Load config with optional overlay files.

    Args:
        base_path: Path to base YAML config file
        overlays: List of overlay YAML files to apply in order

    Returns:
        Merged AlphaLiteConfig
    """
    base_cfg = load_config(base_path)
    base_dict = base_cfg.model_dump()

    for overlay_path in overlays or []:
        overlay_dict = _load_yaml_raw(overlay_path)
        base_dict = _deep_merge(base_dict, overlay_dict)

    return AlphaLiteConfig.model_validate(base_dict)


def _load_yaml_raw(path: Path) -> Dict:
    """Load YAML file as raw dict without pydantic validation."""
    if not path.exists():
        raise FileNotFoundError(f'Config file not found: {path}')
    with path.open('r') as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f'Config must be a dict: {path}')
    return {k: v for k, v in raw.items() if k != '_overlay'}


def _deep_merge(base: Dict, overlay: Dict) -> Dict:
    """Deep merge overlay dict into base dict."""
    result = base.copy()
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
