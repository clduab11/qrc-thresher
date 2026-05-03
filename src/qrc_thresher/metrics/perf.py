"""Performance profiling helpers for benchmark sweeps."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True, slots=True)
class PerfRecord:
    """Single benchmark timing record."""

    name: str
    iterations: int
    total_seconds: float
    mean_seconds: float
    min_seconds: float
    max_seconds: float


def benchmark_callable(
    name: str,
    fn: Callable[[], None],
    iterations: int = 5,
) -> PerfRecord:
    """Benchmark a callable and summarize timing statistics."""
    if iterations < 1:
        raise ValueError(f'iterations must be >= 1, got {iterations}')

    samples: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)

    total = sum(samples)
    return PerfRecord(
        name=name,
        iterations=iterations,
        total_seconds=total,
        mean_seconds=total / iterations,
        min_seconds=min(samples),
        max_seconds=max(samples),
    )


def write_perf_report(records: list[PerfRecord], out_path: Path) -> Path:
    """Write performance records to JSON report file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'records': [
            {
                'name': r.name,
                'iterations': r.iterations,
                'total_seconds': r.total_seconds,
                'mean_seconds': r.mean_seconds,
                'min_seconds': r.min_seconds,
                'max_seconds': r.max_seconds,
            }
            for r in records
        ]
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return out_path
