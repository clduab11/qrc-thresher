"""Source-level health checks (environment and imports).

Checks Python version, required package imports, version ranges, and git state.
"""

from __future__ import annotations

import importlib
import logging
import subprocess
import sys
from typing import Any, Dict

logger = logging.getLogger(__name__)

_REQUIRED_PACKAGES = [
    'pennylane',
    'qiskit',
    'sklearn',
    'numpy',
    'scipy',
    'pandas',
    'matplotlib',
    'pydantic',
    'yaml',
    'pytest',
    'reservoirpy',
    'click',
]

_MIN_PYTHON = (3, 11)


def check_python_version() -> Dict[str, Any]:
    """Check that Python version is >= 3.11."""
    vi = sys.version_info
    ok = (vi.major, vi.minor) >= _MIN_PYTHON
    return {
        'status': 'PASS' if ok else 'FAIL',
        'version': f'{vi.major}.{vi.minor}.{vi.micro}',
        'required': f'>={_MIN_PYTHON[0]}.{_MIN_PYTHON[1]}',
    }


def check_imports() -> Dict[str, str]:
    """Check that all required packages are importable."""
    results: Dict[str, str] = {}
    for pkg in _REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
            results[pkg] = 'PASS'
        except ImportError as exc:
            results[pkg] = f'FAIL: {exc}'
    return results


def check_git_state() -> Dict[str, Any]:
    """Check that git repo is detected and commit hash is readable."""
    try:
        commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return {'status': 'PASS', 'commit': commit}
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return {'status': 'FAIL', 'error': str(exc)}


def check_numpy_rng() -> Dict[str, str]:
    """Check that numpy.random.Generator is available."""
    try:
        import numpy as np

        rng = np.random.default_rng(42)
        _ = rng.uniform(0, 1)
        return {'status': 'PASS'}
    except Exception as exc:
        return {'status': f'FAIL: {exc}'}


def run_source_health() -> Dict[str, Any]:
    """Run all source-level health checks.

    Returns:
        Dict with status per check and overall status.
    """
    python_check = check_python_version()
    imports_check = check_imports()
    git_check = check_git_state()
    rng_check = check_numpy_rng()

    imports_all_pass = all(v == 'PASS' for v in imports_check.values())
    overall = (
        python_check['status'] == 'PASS'
        and imports_all_pass
        and git_check['status'] == 'PASS'
        and rng_check['status'] == 'PASS'
    )

    return {
        'env': python_check,
        'imports': imports_check,
        'git': git_check,
        'numpy_rng': rng_check,
        'overall': 'PASS' if overall else 'FAIL',
    }
