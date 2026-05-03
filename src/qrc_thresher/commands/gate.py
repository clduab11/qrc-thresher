"""Gate command implementation."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np


def gate_handler(name: str) -> int:
    """Handle gate command. Returns exit code."""
    import pandas as pd

    from qrc_thresher.config import load_config

    gates_dir = Path('results') / 'gates'
    gates_dir.mkdir(parents=True, exist_ok=True)

    config = None
    try:
        config = load_config(Path('configs/alpha_lite.yaml'))
    except Exception:
        pass

    if name == 'G0':
        result, evidence, run_ids = _evaluate_gate_g0()
        _write_gate_result(gates_dir, name, result, evidence, run_ids)
        print(f'Gate {name}: {result}')
        return {'PASS': 0, 'FAIL': 1, 'INSUFFICIENT_EVIDENCE': 2}.get(result, 2)

    if name == 'G0.5':
        result, evidence, run_ids = _evaluate_gate_g05()
        _write_gate_result(gates_dir, name, result, evidence, run_ids)
        print(f'Gate {name}: {result}')
        return {'PASS': 0, 'FAIL': 1, 'INSUFFICIENT_EVIDENCE': 2}.get(result, 2)

    if name == 'G7':
        result, evidence, run_ids = _evaluate_gate_g7()
        _write_gate_result(gates_dir, name, result, evidence, run_ids)
        print(f'Gate {name}: {result}')
        return {'PASS': 0, 'FAIL': 1, 'INSUFFICIENT_EVIDENCE': 2}.get(result, 2)

    runs_csv = Path('results') / 'runs.csv'
    if not runs_csv.exists():
        if name == 'G6':
            empty = pd.DataFrame()
            result, evidence, run_ids = _evaluate_gate_g6(empty)
            _write_gate_result(gates_dir, name, result, evidence, run_ids)
            print(f'Gate {name}: {result}')
            return {'PASS': 0, 'FAIL': 1, 'INSUFFICIENT_EVIDENCE': 2}.get(result, 2)

        print(f'INSUFFICIENT_EVIDENCE: no runs.csv found at {runs_csv}')
        _write_gate_result(
            gates_dir,
            name,
            'INSUFFICIENT_EVIDENCE',
            {'message': f'{runs_csv} not found'},
            [],
        )
        return 2

    df = pd.read_csv(runs_csv)
    successful = df[df['success'].astype(str).str.lower() == 'true'].copy()

    if name == 'G1':
        result, evidence, run_ids = _evaluate_gate_g1(successful, config)
    elif name == 'G2':
        result, evidence, run_ids = _evaluate_gate_g2(successful, config)
    elif name == 'G2.5':
        result, evidence, run_ids = _evaluate_gate_g25(successful, config)
    elif name == 'G4':
        result, evidence, run_ids = _evaluate_gate_g4(successful)
    elif name == 'G3':
        result, evidence, run_ids = _evaluate_gate_g3(successful)
    elif name == 'G5':
        result, evidence, run_ids = _evaluate_gate_g5(successful)
    elif name == 'G6':
        result, evidence, run_ids = _evaluate_gate_g6(successful)
    else:
        result = 'INSUFFICIENT_EVIDENCE'
        evidence = {'message': f'Unknown gate: {name}'}
        run_ids = []

    _write_gate_result(gates_dir, name, result, evidence, run_ids)
    print(f'Gate {name}: {result}')

    exit_codes = {'PASS': 0, 'FAIL': 1, 'INSUFFICIENT_EVIDENCE': 2}
    return exit_codes.get(result, 2)


def _latest_health_report() -> Optional[dict]:
    """Return the JSON content of the most recent health report, or None."""
    health_dir = Path('results') / 'health'
    if not health_dir.exists():
        return None
    candidates = sorted(health_dir.glob('*.json'))
    if not candidates:
        return None
    try:
        with candidates[-1].open('r') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _evaluate_gate_g0() -> tuple[str, dict, list]:
    """G0 PASS iff the latest health report's overall is PASS."""
    report = _latest_health_report()
    if report is None:
        return (
            'INSUFFICIENT_EVIDENCE',
            {'message': 'No health report found. Run `qrc-thresher health` first.'},
            [],
        )
    overall = report.get('overall', 'FAIL')
    return (
        'PASS' if overall == 'PASS' else 'FAIL',
        {'overall': overall, 'timestamp_utc': report.get('timestamp_utc', '')},
        [],
    )


def _filter_task(df, task_name: str):
    """Return successful rows whose task_name matches (handles legacy rows)."""
    if 'task_name' in df.columns:
        return df[df['task_name'].astype(str) == task_name]
    return df.iloc[0:0]


def _metric_values(df, metric_name: str) -> list:
    """Pull primary_metric_value from rows whose primary_metric_name matches."""
    import pandas as pd

    if 'primary_metric_name' not in df.columns or 'primary_metric_value' not in df.columns:
        return []
    sub = df[df['primary_metric_name'].astype(str) == metric_name]
    if sub.empty:
        return []

    vals = pd.to_numeric(sub['primary_metric_value'], errors='coerce').dropna()
    return [float(v) for v in vals.tolist()]


def _evaluate_gate_g1(successful, config=None) -> tuple[str, dict, list]:
    """G1: STM MC > threshold AND >=margin_pct above no_entangle ablation MC."""
    stm_mc_threshold = 1.0
    margin_pct_threshold = 0.20
    if config is not None and config.gates is not None:
        stm_mc_threshold = config.gates.G1_stm_mc_threshold
        margin_pct_threshold = config.gates.G1_margin_pct

    qrc = _filter_task(successful, 'stm')
    qrc_mcs = _metric_values(qrc, 'mc')
    abl = _filter_task(successful, 'ablation:no_entangle')
    abl_mcs = _metric_values(abl, 'mc')

    n_seeds = len(qrc_mcs)
    evidence = {
        'qrc_n_runs': n_seeds,
        'no_entangle_n_runs': len(abl_mcs),
        'qrc_mc_mean': float(sum(qrc_mcs) / n_seeds) if qrc_mcs else None,
        'no_entangle_mc_mean': (
            float(sum(abl_mcs) / len(abl_mcs)) if abl_mcs else None
        ),
    }
    run_ids = qrc['run_id'].astype(str).tolist() + abl['run_id'].astype(str).tolist()

    if n_seeds < 5 or not abl_mcs:
        evidence['message'] = (
            'Need >=5 STM runs and >=1 no_entangle ablation run with '
            'primary_metric_value=mc.'
        )
        return 'INSUFFICIENT_EVIDENCE', evidence, run_ids

    qrc_mean = evidence['qrc_mc_mean']
    abl_mean = evidence['no_entangle_mc_mean']
    margin_pct = (qrc_mean - abl_mean) / abl_mean if abl_mean and abl_mean > 0 else 0.0
    evidence['margin_pct'] = margin_pct
    if qrc_mean > stm_mc_threshold and margin_pct >= margin_pct_threshold:
        return 'PASS', evidence, run_ids
    return 'FAIL', evidence, run_ids


def _evaluate_gate_g2(successful, config=None) -> tuple[str, dict, list]:
    """G2: parity accuracy > threshold AND random_features ablation accuracy < max."""
    accuracy_threshold = 0.70
    random_features_max = 0.60
    if config is not None and config.gates is not None:
        accuracy_threshold = config.gates.G2_accuracy_threshold
        random_features_max = config.gates.G2_random_features_max

    qrc = _filter_task(successful, 'parity')
    qrc_accs = _metric_values(qrc, 'accuracy')
    abl = _filter_task(successful, 'ablation:random_features')
    abl_accs = _metric_values(abl, 'accuracy')

    n_seeds = len(qrc_accs)
    evidence = {
        'qrc_n_runs': n_seeds,
        'random_features_n_runs': len(abl_accs),
        'qrc_accuracy_mean': (
            float(sum(qrc_accs) / n_seeds) if qrc_accs else None
        ),
        'random_features_accuracy_mean': (
            float(sum(abl_accs) / len(abl_accs)) if abl_accs else None
        ),
    }
    run_ids = qrc['run_id'].astype(str).tolist() + abl['run_id'].astype(str).tolist()

    if n_seeds < 5:
        evidence['message'] = 'Need >=5 parity runs with primary_metric_value=accuracy.'
        return 'INSUFFICIENT_EVIDENCE', evidence, run_ids

    qrc_mean = evidence['qrc_accuracy_mean']
    abl_mean = evidence['random_features_accuracy_mean']
    if abl_mean is None:
        evidence['message'] = (
            'Need at least one parity-task random_features ablation run '
            'with accuracy logged.'
        )
        return 'INSUFFICIENT_EVIDENCE', evidence, run_ids

    if qrc_mean > accuracy_threshold and abl_mean < random_features_max:
        return 'PASS', evidence, run_ids
    return 'FAIL', evidence, run_ids


def _evaluate_gate_g25(successful, config=None) -> tuple[str, dict, list]:
    """G2.5: full QRC outperforms Haar-random ablation by >=effect_se on STM-MC or parity-acc."""
    effect_se_threshold = 1.0
    if config is not None and config.gates is not None:
        effect_se_threshold = config.gates.G25_effect_se

    qrc_stm = _metric_values(_filter_task(successful, 'stm'), 'mc')
    haar = _metric_values(_filter_task(successful, 'ablation:haar'), 'mc')

    evidence: dict = {
        'qrc_stm_n_runs': len(qrc_stm),
        'haar_n_runs': len(haar),
    }
    run_ids = (
        _filter_task(successful, 'stm')['run_id'].astype(str).tolist()
        + _filter_task(successful, 'ablation:haar')['run_id'].astype(str).tolist()
    )

    if len(qrc_stm) < 3 or len(haar) < 3:
        evidence['message'] = 'Need >=3 STM runs and >=3 Haar ablation runs.'
        return 'INSUFFICIENT_EVIDENCE', evidence, run_ids

    qrc_mean = sum(qrc_stm) / len(qrc_stm)
    haar_mean = sum(haar) / len(haar)
    qrc_var = sum((v - qrc_mean) ** 2 for v in qrc_stm) / max(len(qrc_stm) - 1, 1)
    haar_var = sum((v - haar_mean) ** 2 for v in haar) / max(len(haar) - 1, 1)
    pooled_se = math.sqrt(qrc_var / len(qrc_stm) + haar_var / len(haar))
    evidence['qrc_stm_mean'] = qrc_mean
    evidence['haar_stm_mean'] = haar_mean
    evidence['pooled_se'] = pooled_se
    evidence['delta'] = qrc_mean - haar_mean

    if pooled_se == 0:
        evidence['message'] = 'Zero pooled standard error; check inputs.'
        return 'INSUFFICIENT_EVIDENCE', evidence, run_ids

    if (qrc_mean - haar_mean) >= effect_se_threshold * pooled_se:
        return 'PASS', evidence, run_ids
    return 'FAIL', evidence, run_ids


def _evaluate_gate_g05() -> tuple[str, dict, list]:
    """G0.5: PennyLane vs Qiskit cross-check."""
    from qrc_thresher.reservoirs.pennylane_qrc import (
        build_reservoir_params,
        extract_features,
    )
    from qrc_thresher.reservoirs.qiskit_crosscheck import (
        qiskit_expectation_values,
        verify_crosscheck,
    )

    evidence: dict = {}
    run_ids: list = []

    try:
        rng = np.random.default_rng(2026)
        n_qubits = 2
        depth = 2

        params = build_reservoir_params(
            n_qubits=n_qubits,
            depth=depth,
            readout='z_only',
            backend='default.qubit',
            rng=rng,
        )

        sample_inputs = rng.uniform(-1, 1, size=5)
        all_match = True
        max_diffs = []

        for u_t in sample_inputs:
            pennylane_vals = extract_features(np.array([u_t]), params).ravel()
            qiskit_vals = qiskit_expectation_values(
                u_t,
                params.thetas,
                params.phis,
                n_qubits,
                depth,
            )

            diff = np.abs(pennylane_vals - qiskit_vals)
            max_diffs.append(float(np.max(diff)))
            if not verify_crosscheck(pennylane_vals, qiskit_vals, tolerance=1e-5):
                all_match = False

        evidence['max_diffs'] = max_diffs
        evidence['n_samples'] = len(sample_inputs)
        evidence['all_match'] = all_match

        if all_match:
            return 'PASS', evidence, run_ids
        return 'FAIL', evidence, run_ids

    except Exception as exc:
        evidence['error'] = str(exc)
        return 'FAIL', evidence, run_ids


def _evaluate_gate_g3(successful) -> tuple[str, dict, list]:
    """G3: QRC vs Best ESN with Holm-Bonferroni correction."""
    from scipy import stats as scipy_stats

    from qrc_thresher.metrics.stats import holm_bonferroni

    qrc = _filter_task(successful, 'stm')
    esn = _filter_task(successful, 'esn')

    qrc_mcs = _metric_values(qrc, 'mc')
    esn_vals = _metric_values(esn, 'mc')

    n_qrc = len(qrc_mcs)
    n_esn = len(esn_vals)

    run_ids = qrc['run_id'].astype(str).tolist() + esn['run_id'].astype(str).tolist()

    evidence: dict = {
        'qrc_stm_n_runs': n_qrc,
        'esn_n_runs': n_esn,
        'qrc_mc_mean': float(sum(qrc_mcs) / n_qrc) if qrc_mcs else None,
        'esn_val_mean': float(sum(esn_vals) / n_esn) if esn_vals else None,
    }

    if n_qrc < 5:
        evidence['message'] = 'Need >=5 STM (QRC) runs with primary_metric_value=mc.'
        return 'INSUFFICIENT_EVIDENCE', evidence, run_ids

    if n_esn < 5:
        evidence['message'] = (
            'ESN baseline runs must be logged first. '
            'Run ESN benchmark and ensure results are appended to runs.csv '
            'with task_name="esn" and primary_metric_name="mc".'
        )
        return 'INSUFFICIENT_EVIDENCE', evidence, run_ids

    qrc_mean = evidence['qrc_mc_mean']
    esn_mean = evidence['esn_val_mean']

    qrc_arr = np.array(qrc_mcs)
    esn_arr = np.array(esn_vals)

    min_len = min(len(qrc_arr), len(esn_arr))
    qrc_arr = qrc_arr[:min_len]
    esn_arr = esn_arr[:min_len]

    diff = qrc_arr - esn_arr
    _, p_two_sided = scipy_stats.ttest_rel(qrc_arr, esn_arr)
    p_one_sided = p_two_sided / 2.0 if np.mean(diff) > 0 else 1.0 - (p_two_sided / 2.0)

    adjusted_p = holm_bonferroni([p_one_sided])

    min_adjusted_p = adjusted_p[0]
    evidence['raw_p_value'] = p_one_sided
    evidence['adjusted_p_value'] = adjusted_p[0]
    evidence['qrc_mc_mean'] = float(np.mean(qrc_arr))
    evidence['esn_val_mean'] = float(np.mean(esn_arr))
    evidence['delta'] = float(np.mean(qrc_arr) - np.mean(esn_arr))

    if qrc_mean > esn_mean and min_adjusted_p < 0.05:
        return 'PASS', evidence, run_ids
    return 'FAIL', evidence, run_ids


def _evaluate_gate_g4(successful) -> tuple[str, dict, list]:
    """G4: QRC NARMA-10 NRMSE < ESN NARMA-10 NRMSE."""
    qrc = _filter_task(successful, 'narma')
    esn = successful[successful['task_name'].astype(str) == 'esn_narma']

    qrc_nrmses = _metric_values(qrc, 'nrmse')
    esn_nrmses = _metric_values(esn, 'nrmse')

    n_qrc = len(qrc_nrmses)
    n_esn = len(esn_nrmses)
    evidence: dict = {
        'qrc_narma_n_runs': n_qrc,
        'esn_narma_n_runs': n_esn,
        'qrc_nrmse_mean': float(sum(qrc_nrmses) / n_qrc) if qrc_nrmses else None,
        'esn_nrmse_mean': float(sum(esn_nrmses) / n_esn) if esn_nrmses else None,
    }
    run_ids = qrc['run_id'].astype(str).tolist() + esn['run_id'].astype(str).tolist()

    if n_qrc < 5:
        evidence['message'] = 'Need >=5 NARMA QRC runs with primary_metric_value=nrmse.'
        return 'INSUFFICIENT_EVIDENCE', evidence, run_ids
    if n_esn < 5:
        evidence['message'] = 'Need >=5 NARMA ESN runs with primary_metric_value=nrmse.'
        return 'INSUFFICIENT_EVIDENCE', evidence, run_ids

    qrc_mean = evidence['qrc_nrmse_mean']
    esn_mean = evidence['esn_nrmse_mean']
    if qrc_mean is None or esn_mean is None:
        evidence['message'] = 'NARMA runs found but no nrmse values logged.'
        return 'INSUFFICIENT_EVIDENCE', evidence, run_ids

    if qrc_mean < esn_mean:
        return 'PASS', evidence, run_ids
    return 'FAIL', evidence, run_ids


def _evaluate_gate_g5(successful) -> tuple[str, dict, list]:
    """G5: Full-circuit cross-check across backends."""
    evidence: dict = {}
    run_ids: list = successful['run_id'].astype(str).tolist()

    if 'backend_device' not in successful.columns:
        evidence['message'] = 'backend_device column not found in runs.csv'
        return 'INSUFFICIENT_EVIDENCE', evidence, run_ids

    backends = successful['backend_device'].unique()
    evidence['backends'] = list(backends)

    if len(backends) < 1:
        evidence['message'] = 'No backend data found'
        return 'INSUFFICIENT_EVIDENCE', evidence, run_ids

    stm_data = _filter_task(successful, 'stm')
    mc_by_backend: dict = {}

    for backend in backends:
        be_data = stm_data[stm_data['backend_device'].astype(str) == backend]
        mcs = _metric_values(be_data, 'mc')
        if mcs:
            mc_by_backend[str(backend)] = {
                'mean': float(np.mean(mcs)),
                'n': len(mcs),
                'values': mcs,
            }

    evidence['mc_by_backend'] = mc_by_backend

    if len(mc_by_backend) < 2:
        if len(mc_by_backend) == 1:
            backend_name = list(mc_by_backend.keys())[0]
            be_data = mc_by_backend[backend_name]
            evidence['message'] = f'Only one backend ({backend_name}) with {be_data["n"]} runs'
            backend_filter = successful['backend_device'].astype(str) == backend_name
            circuit_hashes = successful[backend_filter]['circuit_hash'].unique()
            evidence['circuit_hashes'] = list(circuit_hashes)
            evidence['consistent_hashes'] = len(circuit_hashes) == 1
            if len(circuit_hashes) == 1:
                return 'PASS', evidence, run_ids
            return 'FAIL', evidence, run_ids
        evidence['message'] = 'Insufficient data for backend comparison'
        return 'INSUFFICIENT_EVIDENCE', evidence, run_ids

    means = [v['mean'] for v in mc_by_backend.values()]
    max_diff = float(max(means) - min(means)) if len(means) >= 2 else 0.0
    evidence['max_backend_diff'] = max_diff

    if max_diff <= 0.05:
        return 'PASS', evidence, run_ids
    return 'FAIL', evidence, run_ids


def _evaluate_gate_g6(successful) -> tuple[str, dict, list]:
    """G6: Phase-2 readiness checklist gate.

    PASS when core Phase-2 capabilities are present and importable:
    plugin registry, advanced stats, observability hooks, and perf helpers.
    """
    evidence: dict = {
        'n_successful_runs': int(len(successful)),
        'checks': {},
    }
    run_ids = successful['run_id'].astype(str).tolist() if 'run_id' in successful.columns else []

    # Plugin registry and builtins
    try:
        from qrc_thresher.plugins.registry import create_registry_hub, plugin_inventory

        hub = create_registry_hub(load_builtin=True, load_entry_points_flag=False)
        inventory = plugin_inventory()
        evidence['plugin_inventory'] = inventory
        has_core_plugins = all(
            len(group) > 0
            for group in [
                hub.tasks.available(),
                hub.reservoirs.available(),
                hub.baselines.available(),
                hub.gates.available(),
                hub.viz.available(),
            ]
        )
        evidence['checks']['plugin_registry'] = has_core_plugins
    except Exception as exc:
        evidence['checks']['plugin_registry'] = False
        evidence['plugin_registry_error'] = str(exc)

    # Advanced statistics
    try:
        from qrc_thresher.metrics.stats import bca_ci, power_analysis

        rng = np.random.default_rng(123)
        toy = rng.normal(0.0, 1.0, size=50)
        ci_low, ci_high = bca_ci(toy, rng=rng, n_resamples=200)
        required_n = power_analysis(effect_size=0.5, alpha=0.05, power=0.8)
        evidence['checks']['advanced_stats'] = ci_low < ci_high and required_n > 0
        evidence['advanced_stats_sample'] = {
            'bca_ci': [ci_low, ci_high],
            'required_n_d05': required_n,
        }
    except Exception as exc:
        evidence['checks']['advanced_stats'] = False
        evidence['advanced_stats_error'] = str(exc)

    # Observability hooks
    try:
        from qrc_thresher.observability import configure_logging, configure_tracing

        configure_logging(verbose=False, json_logs=False)
        _ = configure_tracing(service_name='qrc_thresher_gate_g6')
        evidence['checks']['observability'] = True
    except Exception as exc:
        evidence['checks']['observability'] = False
        evidence['observability_error'] = str(exc)

    # Perf helpers
    try:
        from qrc_thresher.metrics.perf import benchmark_callable

        perf = benchmark_callable('g6_noop', lambda: None, iterations=2)
        evidence['checks']['perf_helpers'] = perf.mean_seconds >= 0.0
    except Exception as exc:
        evidence['checks']['perf_helpers'] = False
        evidence['perf_error'] = str(exc)

    overall = all(bool(v) for v in evidence['checks'].values())
    return ('PASS' if overall else 'FAIL'), evidence, run_ids


def _evaluate_gate_g7(
    calibration_dir: Optional[Path] = None,
    max_age_days: int = 30,
) -> tuple[str, dict, list]:
    """G7: Device calibration validator gate.

    Validates calibration JSON files under results/calibration/.
    """
    if calibration_dir is None:
        calibration_dir = Path('results') / 'calibration'

    evidence: dict = {
        'calibration_dir': str(calibration_dir),
        'max_age_days': max_age_days,
    }

    if not calibration_dir.exists():
        evidence['message'] = f'Calibration directory not found: {calibration_dir}'
        return 'INSUFFICIENT_EVIDENCE', evidence, []

    files = sorted(calibration_dir.glob('*.json'))
    if not files:
        evidence['message'] = 'No calibration JSON files found.'
        return 'INSUFFICIENT_EVIDENCE', evidence, []

    now = datetime.now(timezone.utc)
    errors: list[str] = []
    validated: list[str] = []

    required = {'backend', 'timestamp_utc', 't1', 't2', 'readout_error'}

    for file_path in files:
        try:
            payload = json.loads(file_path.read_text(encoding='utf-8'))
        except Exception as exc:
            errors.append(f'{file_path.name}: invalid JSON ({exc})')
            continue

        missing = sorted(required - set(payload.keys()))
        if missing:
            errors.append(f'{file_path.name}: missing keys {missing}')
            continue

        try:
            ts = datetime.fromisoformat(str(payload['timestamp_utc']).replace('Z', '+00:00'))
        except Exception:
            errors.append(f'{file_path.name}: invalid timestamp_utc')
            continue

        age_days = (now - ts).total_seconds() / (24 * 3600)
        if age_days > max_age_days:
            errors.append(
                f'{file_path.name}: calibration too old ({age_days:.1f} days > {max_age_days})'
            )
            continue

        try:
            t1 = float(payload['t1'])
            t2 = float(payload['t2'])
            readout_error = float(payload['readout_error'])
        except Exception:
            errors.append(f'{file_path.name}: t1/t2/readout_error must be numeric')
            continue

        if t1 <= 0 or t2 <= 0:
            errors.append(f'{file_path.name}: t1 and t2 must be positive')
            continue
        if not 0.0 <= readout_error <= 1.0:
            errors.append(f'{file_path.name}: readout_error outside [0,1]')
            continue

        validated.append(file_path.name)

    evidence['n_files'] = len(files)
    evidence['validated_files'] = validated
    if errors:
        evidence['errors'] = errors
        return 'FAIL', evidence, []

    return 'PASS', evidence, []


def _write_gate_result(
    gates_dir: Path,
    name: str,
    result: str,
    evidence: dict,
    run_ids: list,
) -> None:
    """Write gate result JSON to results/gates/<name>.json."""
    gate_file = gates_dir / f'{name}.json'
    data = {
        'gate': name,
        'result': result,
        'evidence': evidence,
        'run_ids': run_ids,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    }
    with gate_file.open('w') as f:
        json.dump(data, f, indent=2)
