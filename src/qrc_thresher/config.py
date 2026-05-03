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


class SeedsConfig(BaseModel):
    """Seed configuration for reproducibility."""

    task_seed: int
    reservoir_seed: int
    n_seeds: int = Field(ge=1, le=20)


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
    return AlphaLiteConfig.model_validate(raw)
