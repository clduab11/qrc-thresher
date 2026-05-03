"""Tests for performance profiling helpers."""

from __future__ import annotations

import json

from qrc_thresher.metrics.perf import benchmark_callable, write_perf_report


def test_benchmark_callable_returns_valid_record() -> None:
    rec = benchmark_callable('noop', lambda: None, iterations=3)
    assert rec.name == 'noop'
    assert rec.iterations == 3
    assert rec.mean_seconds >= 0.0


def test_write_perf_report(tmp_path) -> None:
    rec = benchmark_callable('noop', lambda: None, iterations=2)
    out = tmp_path / 'perf.json'
    write_perf_report([rec], out)
    payload = json.loads(out.read_text(encoding='utf-8'))
    assert 'records' in payload
    assert len(payload['records']) == 1
