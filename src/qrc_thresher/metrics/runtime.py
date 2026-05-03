"""Per-stage runtime measurement utilities."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Dict, Generator

logger = logging.getLogger(__name__)


class StageTimer:
    """Accumulates per-stage timing information."""

    def __init__(self) -> None:
        self._times: Dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Generator[None, None, None]:
        """Context manager that times a named stage.

        Args:
            name: Stage name (e.g., 'feature_extraction').

        Yields:
            None.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._times[name] = elapsed
            logger.debug('Stage "%s" completed in %.3f s', name, elapsed)

    def to_dict(self) -> Dict[str, float]:
        """Return copy of timing data.

        Returns:
            Dict mapping stage name to elapsed seconds.
        """
        return dict(self._times)
