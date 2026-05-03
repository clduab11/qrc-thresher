"""Health command implementation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def health_handler(out_dir: str) -> int:
    """Handle health command. Returns exit code."""
    from qrc_thresher.proof.benchmark_health import run_benchmark_health
    from qrc_thresher.proof.source_health import run_source_health

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    source = run_source_health()
    bench = run_benchmark_health()

    overall = source['overall'] == 'PASS' and bench['overall'] == 'PASS'
    report = {
        'version': '1.0',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'checks': {
            'env': source['env']['status'],
            'imports': source['imports'],
            'git': source['git']['status'],
            'config': bench['checks'].get('config', 'FAIL'),
            'tasks': bench['checks'].get('tasks', 'FAIL'),
            'qrc_smoke': bench['checks'].get('qrc_smoke', 'FAIL'),
            'esn_smoke': bench['checks'].get('esn_smoke', 'FAIL'),
            'metrics': bench['checks'].get('metrics', 'FAIL'),
            'manifest': bench['checks'].get('manifest', 'FAIL'),
            'reproducibility': bench['checks'].get('reproducibility', 'FAIL'),
        },
        'details': {
            'source': source,
            'benchmark': bench['details'],
        },
        'overall': 'PASS' if overall else 'FAIL',
    }

    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    report_path = out_path / f'{ts}.json'
    with report_path.open('w') as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return 0 if overall else 1
