"""Noise sweep command implementation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from qrc_thresher.reservoirs.noise_models import (
    build_depolarizing_noise_model,
    build_relaxation_noise_model,
)


def noise_sweep_handler() -> int:
    """Build example noise models and write a sweep summary artifact."""
    depol = build_depolarizing_noise_model(probability=0.002)
    relax = build_relaxation_noise_model(t1=45e3, t2=60e3)

    payload = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'models': {
            'depolarizing': str(type(depol)),
            'relaxation': str(type(relax)),
        },
    }

    out_dir = Path('results') / 'noise'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'noise_sweep.latest.json'
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'Noise sweep summary written to {out_path}')
    return 0
