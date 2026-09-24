"""Gate command implementation."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

FAMILY_NAMES = ('family', 'G1', 'G2', 'G2.5', 'G3', 'G4')
_EXIT = {'PASS': 0, 'FAIL': 1, 'INSUFFICIENT_EVIDENCE': 2}


def gate_handler(
    name: str,
    model: str = 'pennylane_qrc',
    config_path: str = 'configs/alpha_lite.yaml',
    tuning_config: str = 'configs/comparative.yaml',
) -> int:
    """Handle gate command. Returns exit code.

    ``config_path`` is the experiment config (default: the cwd-relative
    configs/alpha_lite.yaml); ``model`` and ``tuning_config`` are used by G0.7 only.
    'family' and its members G1, G2, G2.5, G3, G4 evaluate the comparative family of
    ``config_path`` as a unit (D013, D014; PI ruling 7).
    """
    import pandas as pd

    gates_dir = Path('results') / 'gates'
    gates_dir.mkdir(parents=True, exist_ok=True)

    if name in FAMILY_NAMES:
        return _family_command(name, Path(config_path), gates_dir)

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

    if name == 'G0.7':
        # G0.7 writes a new timestamped JSON and figure per evaluation (never overwrites).
        result, evidence, run_ids = _evaluate_gate_g07(
            config_path=Path(config_path), model=model, tuning_config=Path(tuning_config)
        )
        print(f'Gate {name}: {result}')
        print(f"  {evidence['message']}")
        print(f"  json: {evidence['json']}")
        print(f"  figure: {evidence['figure']}")
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

    if name == 'G5':
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


def _family_command(name: str, config_path: Path, gates_dir: Path) -> int:
    """Evaluate the whole family, write the record and the member views, print one member."""
    from qrc_thresher.gates import comparative

    result, error = comparative.evaluate_config(
        config_path, Path('results') / 'runs.csv', gates_dir
    )
    if error is not None:
        print(f'Gate {name}: {comparative.INSUFFICIENT}')
        print(f'  {error}')
        return 2
    paths = comparative.write_family_report(result, gates_dir)
    members = result['members']
    shown = comparative.MEMBERS if name == 'family' else (name,)
    for member in shown:
        entry = members[member]
        line = (f"Gate {member}: {entry['result']}  (raw p = {entry['raw_p']:.4g}, Holm-adjusted "
                f"p = {entry['adjusted_p']:.4g}, n = {entry['n_pairs']})")
        print(line)
        if entry['message']:
            print(f"  {entry['message']}")
        print(f"  json: {paths[member].as_posix()}")
    print(f"  family: {paths['family'].as_posix()}")
    if name == 'family':
        worst = max((members[m]['result'] for m in comparative.MEMBERS), key=_EXIT.__getitem__)
        return _EXIT[worst]
    return _EXIT[members[name]['result']]


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


def _family_member(name: str, config_path=None) -> tuple[str, dict, list]:
    """One member's view of the comparative family (D013, D014; PI ruling 7).

    The family is always evaluated as a unit (m = 5). The entry points G1, G2, G2.5, G3 and G4
    are kept as thin wrappers over this function so ``pyproject`` stays untouched (debt).
    Returns (result, member dict, run_ids) without writing anything; the gate command writes
    the family JSON and the member views.
    """
    from qrc_thresher.gates import comparative

    path = Path(config_path) if config_path is not None else Path('configs') / 'comparative.yaml'
    result, error = comparative.evaluate_config(
        path, Path('results') / 'runs.csv', Path('results') / 'gates'
    )
    if error is not None:
        return comparative.INSUFFICIENT, {'gate': name, 'message': error}, []
    member = result['members'][name]
    run_ids = list(member['comparison'].get('run_ids', []))
    return member['result'], member, run_ids


def _evaluate_gate_g1(successful=None, config=None, config_path=None) -> tuple[str, dict, list]:
    """G1 (D014): thin wrapper over the comparative family; see ``_family_member``."""
    return _family_member('G1', config_path)


def _evaluate_gate_g2(successful=None, config=None, config_path=None) -> tuple[str, dict, list]:
    """G2 (D014): thin wrapper over the comparative family; see ``_family_member``."""
    return _family_member('G2', config_path)


def _evaluate_gate_g25(successful=None, config=None, config_path=None) -> tuple[str, dict, list]:
    """G2.5 (D014): thin wrapper over the comparative family; see ``_family_member``."""
    return _family_member('G2.5', config_path)


# G0.5 cases (docs/DECISIONS.md D010): (n_qubits, depth, seed), each at every distinct window
# in {1, 2, n} and both readouts, over G05_N_STEPS steps that include the zero-padded rows, at
# the encoding scale pi.
G05_TRIPLES = ((2, 2, 2026), (4, 3, 137), (5, 4, 7))
G05_N_STEPS = 6
G05_READOUTS = ('z_only', 'z_and_zz')
# The registered scale cases (D011): scale in {pi/4, pi/2, 3pi/4} on (4, 3, 137) at every window
# in {1, 2, 4} with both readouts, 18 cases, separate from G05_TRIPLES.
G05_SCALES = (math.pi / 4, math.pi / 2, 3 * math.pi / 4)
G05_SCALE_CASES = tuple(
    (4, 3, 137, window, readout, scale)
    for scale in G05_SCALES
    for window in (1, 2, 4)
    for readout in G05_READOUTS
)


def _g05_case(
    n_qubits: int, depth: int, seed: int, window: int, readout: str,
    encoding_scale: float = math.pi,
) -> dict:
    """One G0.5 case: PennyLane (the production reservoir) against an independent Qiskit build.

    The angles come from build_reservoir_params with default_rng(seed); the inputs are the next
    G05_N_STEPS draws from that generator, Uniform(-1, 1); both sides encode at encoding_scale.
    """
    from qrc_thresher.reservoirs import windowed_qrc
    from qrc_thresher.reservoirs.pennylane_qrc import build_reservoir_params
    from qrc_thresher.reservoirs.qiskit_crosscheck import qiskit_features, verify_crosscheck

    rng = np.random.default_rng(seed)
    params = build_reservoir_params(
        n_qubits=n_qubits, depth=depth, readout=readout, backend='default.qubit', rng=rng
    )
    u = rng.uniform(-1.0, 1.0, size=G05_N_STEPS)
    reservoir = windowed_qrc.WindowedReservoir(
        params=params, window=window, reservoir_seed=seed, encoding_scale=encoding_scale
    )
    pennylane_vals = np.asarray(reservoir.features(u), dtype=np.float64)
    qiskit_vals = np.asarray(
        qiskit_features(
            u, params.thetas, params.phis, n_qubits, depth, window, readout,
            encoding_scale=encoding_scale,
        ),
        dtype=np.float64,
    )
    if pennylane_vals.shape != qiskit_vals.shape:
        raise ValueError(
            f'G0.5 shape mismatch: PennyLane {pennylane_vals.shape}, Qiskit {qiskit_vals.shape}'
        )
    max_abs_diff = float(np.max(np.abs(pennylane_vals - qiskit_vals)))
    return {
        'n_qubits': int(n_qubits),
        'depth': int(depth),
        'seed': int(seed),
        'window': int(window),
        'readout': str(readout),
        'encoding_scale': float(encoding_scale),
        'n_steps': int(G05_N_STEPS),
        'n_zero_padded_rows': int(window - 1),
        'n_features': int(pennylane_vals.shape[1]),
        'max_abs_diff': max_abs_diff,
        'passed': bool(verify_crosscheck(pennylane_vals, qiskit_vals)),
    }


def _evaluate_gate_g05() -> tuple[str, dict, list]:
    """G0.5: PennyLane vs Qiskit cross-check at CROSSCHECK_TOLERANCE, on the D010 cases."""
    from qrc_thresher.reservoirs.qiskit_crosscheck import CROSSCHECK_TOLERANCE

    evidence: dict = {'tolerance': float(CROSSCHECK_TOLERANCE), 'n_steps': int(G05_N_STEPS)}
    run_ids: list = []
    try:
        cases = [
            _g05_case(n_qubits, depth, seed, window, readout)
            for n_qubits, depth, seed in G05_TRIPLES
            for window in sorted({1, 2, n_qubits})
            for readout in G05_READOUTS
        ] + [
            _g05_case(n_qubits, depth, seed, window, readout, encoding_scale=scale)
            for n_qubits, depth, seed, window, readout, scale in G05_SCALE_CASES
        ]
        evidence['cases'] = cases
        evidence['n_cases'] = len(cases)
        evidence['max_abs_diff'] = max(c['max_abs_diff'] for c in cases) if cases else None
        evidence['all_match'] = bool(cases) and all(c['passed'] for c in cases)
        return ('PASS' if evidence['all_match'] else 'FAIL'), evidence, run_ids
    except Exception as exc:
        evidence['error'] = str(exc)
        return 'FAIL', evidence, run_ids


def _evaluate_gate_g07(
    config_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    model: str = 'pennylane_qrc',
    tuning_config: Optional[Path] = None,
) -> tuple[str, dict, list]:
    """G0.7: memory sanity gate (pre-registered v1) on the configured reservoir.

    ``model`` selects the configured PennyLane reservoir (default), its no_entangle ablation,
    the tuning record's design_STM ('tuned_qrc'; the record of ``tuning_config``, D011) or a
    fixed ESN preset ('esn_linear', 'esn_nonlinear'; docs/DECISIONS.md D009).

    The protocol is configs/gates/G0.7.v1.yaml. Every evaluation writes a new
    timestamped JSON and forgetting-curve figure under results/gates/, never
    overwriting an earlier one, and returns their paths as evidence.
    """
    from qrc_thresher.config import load_config
    from qrc_thresher.gates import g07

    cfg = load_config(config_path or Path('configs/alpha_lite.yaml'))
    tuning_record = None
    if model == 'tuned_qrc':
        from qrc_thresher.tuning import TuningRecordMissing, load_record

        tuning_path = Path(tuning_config or Path('configs') / 'comparative.yaml')
        try:
            tuning_record = load_record(tuning_path, 'stm')
        except TuningRecordMissing as exc:
            return 'INSUFFICIENT_EVIDENCE', {
                'message': str(exc), 'stm_clause': 'INSUFFICIENT_EVIDENCE',
                'parity_clause': 'INSUFFICIENT_EVIDENCE', 'measurement_label': '',
                'json': '', 'figure': '',
            }, []
    try:
        result = g07.evaluate_config(cfg, model=model, tuning_record=tuning_record)
    except ValueError as exc:
        if model != 'tuned_qrc':
            raise
        # A record that does not describe this config's pairs or reservoir, or a design whose
        # rebuilt hash is not the record's (D011; CP4b.1 item A1): refuse, write nothing.
        return 'INSUFFICIENT_EVIDENCE', {
            'message': str(exc), 'stm_clause': 'INSUFFICIENT_EVIDENCE',
            'parity_clause': 'INSUFFICIENT_EVIDENCE', 'measurement_label': '',
            'json': '', 'figure': '',
        }, []
    paths = g07.write_report(result, out_dir or Path('results') / 'gates')
    evidence = {
        'message': result['message'],
        'stm_clause': result['clauses']['stm']['result'],
        'parity_clause': result['clauses']['parity']['result'],
        'measurement_label': result['measurement_label'],
        'json': str(paths['json']),
        'figure': str(paths['figure']),
    }
    return result['result'], evidence, []


def _evaluate_gate_g3(successful=None, config=None, config_path=None) -> tuple[str, dict, list]:
    """G3 (D014): thin wrapper over the comparative family; see ``_family_member``."""
    return _family_member('G3', config_path)


def _evaluate_gate_g4(successful=None, config=None, config_path=None) -> tuple[str, dict, list]:
    """G4 (D014): thin wrapper over the comparative family; see ``_family_member``."""
    return _family_member('G4', config_path)


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
        mcs = _metric_values(be_data, 'stm_memory')  # k >= 1 (D013; PI ruling 10)
        if mcs:
            mc_by_backend[str(backend)] = {
                'mean': float(np.mean(mcs)),
                'n': len(mcs),
                'values': mcs,
            }

    evidence['stm_memory_by_backend'] = mc_by_backend

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
