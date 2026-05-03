"""Plugin protocol interfaces for qrc_thresher extension points."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class TaskPlugin(Protocol):
    """Interface for task-generation plugins."""

    name: str

    def __call__(
        self,
        length: int,
        train_frac: float,
        rng: np.random.Generator,
        **kwargs: Any,
    ) -> Any:
        """Generate a dataset object for a task."""


class ReservoirPlugin(Protocol):
    """Interface for reservoir feature-extraction plugins."""

    name: str

    def __call__(self, u: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Transform input sequence into feature matrix."""


class BaselinePlugin(Protocol):
    """Interface for baseline model plugins."""

    name: str

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Train/evaluate a baseline model and return result."""


class GatePlugin(Protocol):
    """Interface for gate evaluator plugins."""

    name: str

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[str, dict, list]:
        """Evaluate a gate and return (verdict, evidence, run_ids)."""


class VizPlugin(Protocol):
    """Interface for visualization plugins."""

    name: str

    def __call__(self, *args: Any, **kwargs: Any) -> list[str]:
        """Generate visualization artifacts and return output paths."""


class NoiseModelPlugin(Protocol):
    """Interface for noise model plugins."""

    name: str

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Build and return a backend-specific noise model object."""
