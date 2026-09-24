"""Configuration models for qrc_thresher (pydantic v2).

All configuration flows through these validated models.
No untyped dicts cross module boundaries.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TaskConfig(BaseModel):
    """Task configuration."""

    name: Literal['stm', 'parity', 'narma']
    length: int = Field(ge=100)
    train_frac: float = Field(gt=0, lt=1)
    delay_max: Optional[int] = None
    parity_window: Optional[int] = None


class ReservoirConfig(BaseModel):
    """Quantum reservoir configuration.

    ``window`` is the input window w of design (a) (docs/DECISIONS.md D003, D010): qubit j
    re-uploads u_{t-(j mod w)} at every layer. w = 1 is today's memoryless circuit. A window
    outside 1..n_qubits is refused. ``encoding_scale`` is the angle-encoding scale alpha of
    RY(alpha * u) (D011); pi is today's circuit and the untuned default.
    """

    backend: Literal['default.qubit', 'lightning.qubit']
    n_qubits: int = Field(ge=2, le=12)
    depth: int = Field(ge=1, le=10)
    readout: Literal['z_only', 'z_and_zz'] = 'z_only'
    window: int = Field(default=1, ge=1)
    encoding_scale: float = Field(default=math.pi, gt=0)

    @model_validator(mode='after')
    def _window_fits_the_register(self) -> 'ReservoirConfig':
        if self.window > self.n_qubits:
            raise ValueError(
                f'reservoir.window must be in 1..n_qubits = 1..{self.n_qubits}; got {self.window}'
            )
        return self


class ESNGridConfig(BaseModel):
    """ESN hyperparameter grid. The readout penalty is not part of it: every model uses
    the harness readout over training.ridge_alphas (docs/DECISIONS.md D009)."""

    model_config = ConfigDict(extra='forbid')

    spectral_radius: List[float] = Field(min_length=1)
    input_scaling: List[float] = Field(min_length=1)
    leak_rate: List[float] = Field(min_length=1)


class BaselineConfig(BaseModel):
    """Classical baseline configuration.

    ``esn_grid`` is the in-line ESN search of a config without a ``tuning`` block (D009); with
    a tuning block the ESN grid is ``tuning.esn`` and ``esn_grid`` must be absent (D011). The
    washout is ``training.washout`` for every model (D012), so ``esn_washout`` is refused.
    """

    model_config = ConfigDict(extra='forbid')

    enabled: List[Literal['esn', 'random_features', 'gru']]
    esn_grid: Optional[ESNGridConfig] = None
    rks_dim: Optional[int] = None


class AblationConfig(BaseModel):
    """Ablation study configuration."""

    name: Literal['phase_random', 'no_entangle', 'random_features', 'haar']


class TrainingConfig(BaseModel):
    """Training and cross-validation configuration.

    ``washout`` rows [0, washout) are dropped from training and tuning for every model (D012);
    the registered value is 50. ``AlphaLiteConfig`` checks it against the task and reservoir.
    """

    ridge_alphas: List[float]
    cv_folds: int = Field(ge=2, le=10)
    washout: int = Field(default=50, ge=0)


class QRCGridConfig(BaseModel):
    """QRC tuning grid: depth x window x encoding_scale (D011)."""

    model_config = ConfigDict(extra='forbid')

    depth: List[int] = Field(min_length=1)
    window: List[int] = Field(min_length=1)
    encoding_scale: List[float] = Field(min_length=1)


class RKSGridConfig(BaseModel):
    """RKS tuning grid: bandwidth sigma x input window d (D011, defect D13)."""

    model_config = ConfigDict(extra='forbid')

    sigma: List[float] = Field(min_length=1)
    window: List[int] = Field(min_length=1)


class TuningConfig(BaseModel):
    """The three tuning grids of D011. Its presence triggers `qrc-thresher tune`."""

    model_config = ConfigDict(extra='forbid')

    qrc: QRCGridConfig
    esn: ESNGridConfig
    rks: RKSGridConfig


class ProofConfig(BaseModel):
    """Proof layer configuration."""

    log_entanglement: bool = True
    log_circuit_hash: bool = True
    log_artifacts: bool = False


MEASUREMENT_LABELS: Dict[str, str] = {
    'exact': 'exact (oracle upper bound)',
}


class MeasurementConfig(BaseModel):
    """Measurement model for reservoir readouts (docs/DECISIONS.md D004).

    'exact' means exact expectation values. They are an oracle upper bound on what a
    device could measure and are never used for a headline claim. Finite-shot models
    are added when the shot path lands.
    """

    model: Literal['exact'] = 'exact'


def measurement_label(model: str) -> str:
    """Return the report label for a measurement model.

    Args:
        model: Measurement model name, e.g. 'exact'.

    Returns:
        Label used in every report, e.g. 'exact (oracle upper bound)'.

    Raises:
        ValueError: If the model is unknown.
    """
    try:
        return MEASUREMENT_LABELS[model]
    except KeyError:
        raise ValueError(f'Unknown measurement model: {model!r}') from None


class SeedsConfig(BaseModel):
    """Seed configuration for reproducibility."""

    task_seed: int
    reservoir_seed: int
    n_seeds: int = Field(ge=1, le=20)


NARMA_ZERO_ROWS = 10  # NARMA-10's first ten targets are exactly zero (D012)


class AlphaLiteConfig(BaseModel):
    """Top-level experiment configuration.

    The comparative gates' thresholds live in configs/gates/COMPARATIVE.v1.yaml (D013, D014),
    not here: the former ``gates`` block is gone.
    """

    experiment_name: str
    task: TaskConfig
    reservoir: ReservoirConfig
    measurement: MeasurementConfig = Field(default_factory=MeasurementConfig)
    baseline: BaselineConfig
    ablation: Optional[AblationConfig] = None
    training: TrainingConfig
    proof: ProofConfig
    seeds: SeedsConfig
    tuning: Optional[TuningConfig] = None

    @model_validator(mode='after')
    def _washout_covers_the_padded_rows(self) -> 'AlphaLiteConfig':
        """D012: the washout must cover every zero-padded target row and leave training rows.

        Structural minima only: K (task.delay_max), the NARMA-10 zero rows (every task can be
        run from one config, so this always applies), parity_window - 1, window - 1, and at
        least 2 * cv_folds rows in [washout, train_end). Every violation is reported at once.
        The NARMA transient beyond the zero rows is the registered value's job, not a validator's.
        """
        w = self.training.washout
        task = self.task
        problems = []
        if task.delay_max is not None and w < task.delay_max:
            problems.append(f'at least task.delay_max = {task.delay_max}')
        if w < NARMA_ZERO_ROWS:
            problems.append(f'at least the {NARMA_ZERO_ROWS} NARMA-10 zero rows')
        if task.parity_window is not None and w < task.parity_window - 1:
            problems.append(f'at least task.parity_window - 1 = {task.parity_window - 1}')
        if w < self.reservoir.window - 1:
            problems.append(f'at least reservoir.window - 1 = {self.reservoir.window - 1}')
        train_end = int(task.length * task.train_frac)
        if train_end - w < 2 * self.training.cv_folds:
            problems.append(
                f'small enough to leave 2 * cv_folds = {2 * self.training.cv_folds} rows in '
                f'[washout, train_end = {train_end}) (it leaves {train_end - w})'
            )
        if problems:
            raise ValueError(
                f'training.washout = {w} must be ' + ' and '.join(problems) + ' (D012)'
            )
        return self

    @model_validator(mode='after')
    def _one_esn_grid(self) -> 'AlphaLiteConfig':
        """D011: with a tuning block the ESN grid is tuning.esn; baseline.esn_grid must go."""
        if self.tuning is not None and self.baseline.esn_grid is not None:
            raise ValueError(
                'baseline.esn_grid and tuning.esn are both present; with a tuning block the ESN '
                'grid is tuning.esn (docs/DECISIONS.md D011)'
            )
        return self


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
