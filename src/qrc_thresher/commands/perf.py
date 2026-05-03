"""Performance benchmark command implementation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from qrc_thresher.metrics.perf import benchmark_callable, write_perf_report


def perf_handler(iterations: int = 5) -> int:
    """Run lightweight performance profiling and write benchmark report."""
    rng = np.random.default_rng(123)
    u = rng.uniform(-1, 1, size=100)

    def _numpy_baseline() -> None:
        _ = np.fft.fft(u)

    def _linear_algebra() -> None:
        a = rng.normal(0, 1, size=(64, 64))
        _ = np.linalg.svd(a, compute_uv=False)

    records = [
        benchmark_callable('fft_100', _numpy_baseline, iterations=iterations),
        benchmark_callable('svd_64', _linear_algebra, iterations=iterations),
    ]

    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out_path = Path('results') / 'benchmarks' / f'perf_{ts}.json'
    write_perf_report(records, out_path)
    print(f'Performance report written to {out_path}')
    return 0
