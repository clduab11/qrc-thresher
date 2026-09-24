"""The comparative family G1, G2, G2.5, G3, G4 (docs/DECISIONS.md D013, D014).

The protocol is configs/gates/COMPARATIVE.v1.yaml, loaded into a frozen spec. The family is
evaluated as a unit on the successful rows of one config's runs.csv: every member is a paired
comparison (``metrics.paired.compare_arms``) of the tuned design's rows against its registered
comparator, the five one-sided paired-t p-values are Holm-adjusted together (m = 5; an
INSUFFICIENT member enters as 1.0), and a member PASSes iff its adjusted p <= alpha, its floor
holds and, for G1, the tuned design's G0.7 v1 verdict is PASS. The untuned default is reported
beside every member with identical statistics and never flips a verdict (PI rulings 1 and 5).

Arms are identified through the tuning records (``Designs``): a QRC arm is the rows of the
member's task whose circuit_hash is design_<task>(pair); an inherited arm is the ablation rows
whose hash is that design's hash plus the D010 suffix; the classical arms are the baseline rows
of the registered task name. The (member, task) -> row-selector mapping lives here, on
``task_names``, and nowhere else (PI ruling 2).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from qrc_thresher.config import measurement_label
from qrc_thresher.metrics.paired import PairedComparison, compare_arms
from qrc_thresher.metrics.stats import holm_bonferroni
from qrc_thresher.task_names import (
    ablation_task_name,
    parse_task_name,
    qrc_task_name,
    task_metric,
)

FAMILY_NAME = 'COMPARATIVE'
FAMILY_SIZE = 5
MEMBERS = ('G1', 'G2', 'G2.5', 'G3', 'G4')
PROTOCOL_FILE = Path('configs') / 'gates' / 'COMPARATIVE.v1.yaml'
_REPO_ROOT = Path(__file__).resolve().parents[3]
INSUFFICIENT = 'INSUFFICIENT_EVIDENCE'
Pair = Tuple[int, int]


# --- the registered protocol ------------------------------------------------------------------

class _Spec(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)


class MeasurementSpec(_Spec):
    model: Literal['exact']
    label: str


class PairingSpec(_Spec):
    keys: List[str]
    min_pairs: int = Field(ge=2)
    budget_equality: List[str]
    inherited_match: str
    duplicate_rule: str
    washout: int = Field(ge=0)


class WilcoxonSpec(_Spec):
    zero_method: Literal['wilcox']
    method: str


class BootstrapSpec(_Spec):
    method: Literal['bca']
    n_resamples: int = Field(ge=1)
    rng_seed: int = Field(ge=0)
    ci_level: float = Field(gt=0, lt=1)


class PowerSpec(_Spec):
    n_pairs: int
    detectable_d_z_at_80pct_power_one_sided: float


class StatisticsSpec(_Spec):
    decision: Literal['paired_t_one_sided']
    alpha: float = Field(gt=0, lt=1)
    correction: Literal['holm']
    family_size: Literal[5]
    holm_family: List[str]
    insufficient_member_p: float
    reported: List[str]
    wilcoxon: WilcoxonSpec
    bootstrap: BootstrapSpec
    d_z: str
    power: PowerSpec


class FloorSpec(_Spec):
    metric: str
    rule: Literal['mean_greater_than', 'mean_less_than']
    value: float


class MemberSpec(_Spec):
    clause_a: Optional[str] = None
    design: Literal['stm', 'parity', 'narma']
    task: Literal['stm', 'parity', 'narma']
    metric: str
    baseline: str
    direction: Literal['greater', 'less']
    readout: Optional[Literal['z_only', 'z_and_zz']] = None
    floor: Optional[FloorSpec] = None
    reported_beside: Optional[str] = None


class DefaultsSpec(_Spec):
    compared_qrc: Dict[str, Union[str, int, float]]
    reported_qrc: Dict[str, Union[str, int, float]]
    esn: Dict[str, str]
    one_candidate_row_per_pair: bool
    comparators: Dict[str, str]


class ComparativeProtocol(_Spec):
    """The COMPARATIVE v1 pre-registration, validated field by field."""

    family: Literal['COMPARATIVE']
    version: Literal[1]
    decision_records: List[str]
    experiment_config: str
    tuning_grids: str
    measurement: MeasurementSpec
    pairing: PairingSpec
    statistics: StatisticsSpec
    members: Dict[str, MemberSpec]
    defaults: DefaultsSpec
    source_path: str = ''
    sha256: str = ''


def protocol_sha256(raw: dict) -> str:
    """SHA-256 of the canonical JSON of a parsed protocol file (sorted keys)."""
    return hashlib.sha256(json.dumps(raw, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def load_protocol(path: Optional[Path] = None) -> ComparativeProtocol:
    """Load and validate the registered family protocol (cwd first, then the repo root)."""
    if path is None:
        candidates = [PROTOCOL_FILE, _REPO_ROOT / PROTOCOL_FILE]
        path = next((p for p in candidates if p.exists()), candidates[0])
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'COMPARATIVE protocol file not found: {path}')
    raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    protocol = ComparativeProtocol.model_validate(
        {**raw, 'source_path': _display_path(path), 'sha256': protocol_sha256(raw)}
    )
    if tuple(protocol.members) != MEMBERS:
        raise ValueError(f'the protocol must list the members {MEMBERS} in order')
    return protocol


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


# --- the designs --------------------------------------------------------------------------------

@dataclass
class Designs:
    """Which circuit hash each arm must carry (from the tuning records; D011, D013).

    Attributes:
        tuned: task -> pair -> circuit_hash of design_<task>(pair).
        default: design label ('default', 'default_w1') -> pair -> hash of that default QRC.
        ablation_hash: (design_hash, ablation_name, pair) -> the inherited ablation's hash.
    """

    tuned: Dict[str, Dict[Pair, str]]
    default: Dict[str, Dict[Pair, str]] = field(default_factory=dict)
    ablation_hash: Callable[[str, str, Pair], str] = None  # type: ignore[assignment]


def designs_from_records(cfg, records: Dict[str, dict]) -> Designs:
    """Designs from the three tuning records of a config (missing tasks are simply absent)."""
    from qrc_thresher.deploy import DEFAULT_QRC, DEFAULT_WINDOWS
    from qrc_thresher.tuning import seed_pairs, tuned_reservoir

    pairs = seed_pairs(cfg)
    by_hash: Dict[str, Tuple[dict, int]] = {}
    tuned: Dict[str, Dict[Pair, str]] = {}
    for task, record in records.items():
        tuned[task] = {}
        for ts, rs in pairs:
            entry = record['qrc'].get(f'{ts}/{rs}')
            if entry is None:
                continue
            tuned[task][(ts, rs)] = entry['circuit_hash']
            by_hash[entry['circuit_hash']] = (entry, rs)
    default: Dict[str, Dict[Pair, str]] = {}
    for window, label in DEFAULT_WINDOWS:
        default[label] = {}
        entry = {**DEFAULT_QRC, 'window': window}
        for ts, rs in pairs:
            h = tuned_reservoir(cfg, entry, rs).circuit_hash
            default[label][(ts, rs)] = h
            by_hash[h] = (entry, rs)

    def ablation_hash(design_hash: str, name: str, pair: Pair) -> str:
        entry, rs = by_hash[design_hash]
        return tuned_reservoir(cfg, entry, rs, ablation=name).circuit_hash

    return Designs(tuned=tuned, default=default, ablation_hash=ablation_hash)


# --- row selection (the one (member, task) -> rows mapping; PI ruling 2) ------------------------

def _rows(runs: pd.DataFrame, task_name: str, design: str) -> pd.DataFrame:
    return runs[(runs['task_name'].astype(str) == task_name)
                & (runs['design'].astype(str) == design)]


def _with_expected_hash(rows: pd.DataFrame, expected: Dict[Pair, str]) -> pd.DataFrame:
    """Rows whose circuit_hash is the expected one for their pair (other designs drop out)."""
    if rows.empty:
        return rows
    keep = [
        str(r['circuit_hash']) == expected.get((int(r['task_seed']), int(r['reservoir_seed'])))
        for _, r in rows.iterrows()
    ]
    return rows[pd.Series(keep, index=rows.index)]


def qrc_arm(runs: pd.DataFrame, task: str, hashes: Dict[Pair, str], design: str) -> pd.DataFrame:
    """The QRC rows of ``task`` carrying ``design`` whose hash is the design's for their pair."""
    return _with_expected_hash(_rows(runs, qrc_task_name(task), design), hashes)


def comparator_arm(
    runs: pd.DataFrame, baseline: str, task: str, designs: Designs,
    design_hashes: Dict[Pair, str], *, default_table: bool,
):
    """Arm B for a member: (rows, design label, match rule, expected_hash) by baseline kind."""
    parsed = parse_task_name(baseline)
    if parsed['kind'] == 'ablation':
        name = parsed['model']
        expected = {p: designs.ablation_hash(h, name, p) for p, h in design_hashes.items()}
        rows = _rows(runs, ablation_task_name(name), 'inherited')
        rows = rows[rows['primary_metric_name'].astype(str) == task_metric(task)]  # the task
        rows = _with_expected_hash(rows, expected)
        return rows, 'inherited', 'hash', (lambda design_hash, pair: designs.ablation_hash(
            design_hash, name, pair))
    if parsed['kind'] == 'baseline':
        if parsed['model'] == 'rks':
            # G2: the tuned RKS in both tables (a task-triviality floor; PI ruling 5).
            match = None if default_table else 'budget'
            return _rows(runs, baseline, 'tuned'), 'tuned', match, None
        label = 'default' if default_table else 'tuned'
        return _rows(runs, baseline, label), label, 'budget', None
    raise ValueError(f'unknown comparator {baseline!r} for the family')


# --- evaluation ---------------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return 'unknown'


def _sweep_for(sweep_id: Union[str, Dict[str, str], None], task: str) -> Optional[str]:
    if isinstance(sweep_id, dict):
        return sweep_id.get(task)
    return sweep_id


def _all_sweeps(sweep_id: Union[str, Dict[str, str], None]) -> List[str]:
    if isinstance(sweep_id, dict):
        return [str(v) for v in sweep_id.values() if v is not None]
    return [] if sweep_id is None else [str(sweep_id)]


def _floor(spec: Optional[FloorSpec], comparison: PairedComparison) -> Optional[dict]:
    if spec is None:
        return None
    observed = comparison.mean_a if comparison.status == 'OK' else None
    if observed is None:
        passed = None
    elif spec.rule == 'mean_greater_than':
        passed = bool(observed > spec.value)
    else:
        passed = bool(observed < spec.value)
    return {'metric': spec.metric, 'rule': spec.rule, 'value': spec.value,
            'observed': observed, 'passed': passed}


def _mean_report(rows: pd.DataFrame, design: str, task_name: str) -> dict:
    values = pd.to_numeric(rows['primary_metric_value'], errors='coerce').dropna()
    n = int(len(values))
    return {
        'design': design,
        'task_name': task_name,
        'n_rows': n,
        'mean': float(values.mean()) if n else None,
        'std': float(values.std(ddof=1)) if n > 1 else (0.0 if n == 1 else None),
    }


def _mc_k0_mean(rows: pd.DataFrame) -> Optional[float]:
    values = []
    for text in rows['secondary_metrics'].tolist() if 'secondary_metrics' in rows else []:
        try:
            value = json.loads(text) if isinstance(text, str) else (text or {})
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict) and 'mc_k0' in value and value['mc_k0'] is not None:
            values.append(float(value['mc_k0']))
    return float(sum(values) / len(values)) if values else None


def _check_g07(
    g07_tuned: Optional[dict], config_hash: str, sweep_id: Optional[str]
) -> Tuple[Optional[dict], Optional[str]]:
    """Ruling 4: the G0.7 tuned file must name this config's hash and sweep."""
    if not g07_tuned:
        return None, ('G1 clause (a) needs a G0.7 v1 evaluation of the tuned design '
                      '(model tuned_qrc); none found')
    details = g07_tuned.get('model_details') or {}
    file_hash = details.get('tuning_config_hash')
    file_sweep = details.get('sweep_id')
    entry = {
        'result': g07_tuned.get('result'),
        'json': g07_tuned.get('json'),
        'sha256': g07_tuned.get('sha256'),
        'config_hash': file_hash,
        'sweep_id': file_sweep,
    }
    if file_hash != config_hash:
        return entry, (f"G0.7 tuned_qrc file {g07_tuned.get('json')} was tuned for config_hash "
                       f"{file_hash}, not the rows' {config_hash}")
    if sweep_id is not None and file_sweep != sweep_id:
        return entry, (f"G0.7 tuned_qrc file {g07_tuned.get('json')} belongs to sweep "
                       f"{file_sweep}, "
                       f"not the rows' {sweep_id}")
    return entry, None


def evaluate_family(
    runs: pd.DataFrame,
    protocol: ComparativeProtocol,
    designs: Designs,
    *,
    g07_tuned: Optional[dict],
    readout: str,
    config_hash: str,
    sweep_id: Union[str, Dict[str, str]],
    tuning_record_sha: Union[str, Dict[str, str], None] = None,
    measurement_model: str = 'exact',
) -> dict:
    """Evaluate the five members in one pass with m = 5 (D013, D014).

    Args:
        runs: Successful manifest rows (any config, any order); rows of other configs or
            sweeps are ignored.
        protocol: The registered family.
        designs: The design hashes (``designs_from_records``, or synthetic in tests).
        g07_tuned: The newest G0.7 tuned_qrc evaluation as {'result', 'json', 'sha256',
            'model_details': {'tuning_config_hash', 'sweep_id'}}, or None.
        readout: The config's reservoir readout (G1b and G2.5 need z_only).
        config_hash: The config's hash; ``sweep_id`` a single stamp or one per design task
            (each `tune TASK` stamps its own record; a member's tuned rows carry the sweep of
            its design task, default-design rows any of the family's sweeps).
    """
    alpha = protocol.statistics.alpha
    successful = runs
    if 'success' in runs:
        successful = runs[runs['success'].astype(str).str.lower() == 'true']
    successful = successful[successful['config_hash'].astype(str) == str(config_hash)]
    successful = successful.sort_values(  # canonical order: the result must not depend on it
        ['task_name', 'design', 'task_seed', 'reservoir_seed', 'run_id'], kind='mergesort'
    ).reset_index(drop=True)
    members: Dict[str, dict] = {}
    raw_p: Dict[str, float] = {}

    for name in MEMBERS:
        spec = protocol.members[name]
        sweep = _sweep_for(sweep_id, spec.design)
        rows = successful
        if sweep is not None:
            rows = successful[successful['sweep_id'].astype(str) == str(sweep)]
        # Default-design rows are untuned: they carry the sweep of whichever record deployed
        # them, so the default table admits rows from any of the family's sweeps.
        default_rows = successful
        family_sweeps = _all_sweeps(sweep_id)
        if family_sweeps:
            default_rows = successful[successful['sweep_id'].astype(str).isin(family_sweeps)]
        entry: dict = {'gate': name, 'design': spec.design, 'task': spec.task,
                       'metric': spec.metric, 'direction': spec.direction,
                       'baseline': spec.baseline, 'sweep_id': sweep}
        messages: List[str] = []

        # The tuned comparison (verdict-bearing).
        tuned_hashes = designs.tuned.get(spec.design, {})
        arm_a = qrc_arm(rows, spec.task, tuned_hashes, 'tuned')
        arm_b, label_b, match, expected = comparator_arm(
            rows, spec.baseline, spec.task, designs, tuned_hashes, default_table=False
        )
        comparison = compare_arms(
            arm_a, arm_b, metric=spec.metric, direction=spec.direction,
            designs=('tuned', label_b), match=match, expected_hash=expected,
            min_pairs=protocol.pairing.min_pairs,
            n_resamples=protocol.statistics.bootstrap.n_resamples,
            rng_seed=protocol.statistics.bootstrap.rng_seed,
        )
        if comparison.status != 'OK':
            messages.append(f'{spec.baseline}: {comparison.reason}')
        if spec.readout is not None and readout != spec.readout:
            messages.append(
                f'{name} isolates its factor only under the {spec.readout} readout (D010, D014); '
                f'the config reads out {readout}'
            )
        g07_entry = None
        if spec.clause_a is not None:
            g07_entry, problem = _check_g07(g07_tuned, config_hash, sweep)
            if problem:
                messages.append(problem)
            entry['g07'] = g07_entry

        floor = _floor(spec.floor, comparison)
        insufficient = bool(messages)
        if insufficient:
            p = float(protocol.statistics.insufficient_member_p)
        else:
            p = float(comparison.p_one_sided)
        raw_p[name] = p
        baseline_better = False
        if comparison.status == 'OK':
            if spec.direction == 'greater':
                wrong_way = comparison.mean_diff < 0
            else:
                wrong_way = comparison.mean_diff > 0
            baseline_better = bool(wrong_way and comparison.p_two_sided <= alpha)

        # The default table (reported, never gated; PI rulings 1 and 5).
        default_label = str(protocol.defaults.compared_qrc['design'])
        default_hashes = designs.default.get(default_label, {})
        d_arm_a = qrc_arm(default_rows, spec.task, default_hashes, default_label)
        d_arm_b, d_label_b, d_match, d_expected = comparator_arm(
            default_rows, spec.baseline, spec.task, designs, default_hashes, default_table=True
        )
        default_comparison = compare_arms(
            d_arm_a, d_arm_b, metric=spec.metric, direction=spec.direction,
            designs=(default_label, d_label_b), match=d_match, expected_hash=d_expected,
            min_pairs=protocol.pairing.min_pairs,
            n_resamples=protocol.statistics.bootstrap.n_resamples,
            rng_seed=protocol.statistics.bootstrap.rng_seed,
        )
        reported_label = str(protocol.defaults.reported_qrc['design'])
        reported_rows = qrc_arm(
            default_rows, spec.task, designs.default.get(reported_label, {}), reported_label
        )

        reported: dict = {}
        if spec.reported_beside == 'stm_memory_margin_over_no_entangle':
            stm_a = qrc_arm(rows, 'stm', tuned_hashes, 'tuned')
            stm_b, lb, m, ex = comparator_arm(rows, ablation_task_name('no_entangle'), 'stm',
                                              designs, tuned_hashes, default_table=False)
            beside = compare_arms(stm_a, stm_b, metric='stm_memory', direction='greater',
                                  designs=('tuned', lb), match=m, expected_hash=ex,
                                  min_pairs=protocol.pairing.min_pairs,
                                  n_resamples=protocol.statistics.bootstrap.n_resamples,
                                  rng_seed=protocol.statistics.bootstrap.rng_seed)
            reported['stm_memory_margin_over_no_entangle'] = beside.to_dict()
        if spec.reported_beside == 'mc_k0':
            reported['mc_k0'] = {'qrc': _mc_k0_mean(arm_a), 'esn': _mc_k0_mean(arm_b)}

        entry.update({
            'comparison': comparison.to_dict(),
            'n_pairs': comparison.n_pairs,
            'raw_p': p,
            'floor': floor,
            'baseline_better': baseline_better,
            'p_two_sided': comparison.p_two_sided,
            'reported': reported,
            'default': default_comparison.to_dict(),
            'default_w1': _mean_report(reported_rows, reported_label, qrc_task_name(spec.task)),
            'message': '; '.join(messages) if messages else '',
        })
        members[name] = entry

    adjusted = holm_bonferroni([raw_p[m] for m in MEMBERS])
    for name, adj in zip(MEMBERS, adjusted):
        member = members[name]
        member['adjusted_p'] = float(adj)
        if member['message']:
            member['result'] = INSUFFICIENT
        else:
            passed = member['adjusted_p'] <= alpha
            if member['floor'] is not None and not member['floor']['passed']:
                passed = False
            if member.get('g07') is not None and member['g07'].get('result') != 'PASS':
                passed = False
            member['result'] = 'PASS' if passed else 'FAIL'
            if not passed:
                member['message'] = _fail_message(member, alpha)

    return {
        'family': FAMILY_NAME,
        'version': protocol.version,
        'protocol_path': protocol.source_path,
        'protocol_sha256': protocol.sha256,
        'alpha': alpha,
        'family_size': FAMILY_SIZE,
        'holm_family': list(protocol.statistics.holm_family),
        'config_hash': config_hash,
        'sweep_id': sweep_id,
        'tuning_record_sha': tuning_record_sha,
        'git_commit': _git_commit(),
        'measurement_model': measurement_model,
        'measurement_label': measurement_label(measurement_model),
        'readout': readout,
        'n_rows': int(len(successful)),
        'members': members,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    }


def _fail_message(member: dict, alpha: float) -> str:
    parts = []
    if member['adjusted_p'] > alpha:
        parts.append(f"Holm-adjusted one-sided p = {member['adjusted_p']:.4g} > {alpha}")
    if member['baseline_better']:
        parts.append(f"baseline better (two-sided p = {member['p_two_sided']:.4g})")
    if member['floor'] is not None and not member['floor']['passed']:
        f = member['floor']
        parts.append(f"floor {f['metric']} {f['rule']} {f['value']} not met ({f['observed']:.4g})")
    if member.get('g07') is not None and member['g07'].get('result') != 'PASS':
        parts.append(f"G0.7 v1 on the tuned design: {member['g07'].get('result')}")
    return '; '.join(parts) or 'FAIL'


# --- reports ------------------------------------------------------------------------------------

def _fresh(out_dir: Path, stem: str) -> Path:
    path, n = out_dir / f'{stem}.json', 1
    while path.exists():
        path, n = out_dir / f'{stem}.{n}.json', n + 1
    return path


def write_family_report(result: dict, out_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Write the family JSON (the record) and one view per member; never overwrite (ruling 7)."""
    out_dir = Path(out_dir) if out_dir is not None else Path('results') / 'gates'
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    family_path = _fresh(out_dir, f'{FAMILY_NAME}.v{result["version"]}.{stamp}')
    text = json.dumps(result, indent=2, allow_nan=False)
    with family_path.open('x', encoding='utf-8') as f:
        f.write(text)
    family_sha = hashlib.sha256(family_path.read_bytes()).hexdigest()
    paths = {'family': family_path}
    provenance = {k: result[k] for k in (
        'protocol_path', 'protocol_sha256', 'config_hash', 'sweep_id', 'tuning_record_sha',
        'git_commit', 'measurement_model', 'measurement_label', 'alpha', 'family_size',
        'holm_family', 'timestamp_utc',
    )}
    for name in MEMBERS:
        safe = re.sub(r'[^A-Za-z0-9._-]+', '_', name)
        gate_path = _fresh(out_dir, f'{safe}.{stamp}')
        view = {**result['members'][name], **provenance,
                'family_json': family_path.as_posix(), 'family_sha256': family_sha}
        with gate_path.open('x', encoding='utf-8') as f:
            json.dump(view, f, indent=2, allow_nan=False)
        paths[name] = gate_path
    return paths


def newest_g07_tuned(gates_dir: Path) -> Optional[dict]:
    """The newest results/gates/G0.7.tuned_qrc.*.json (discovery only; ruling 4 verifies it)."""
    candidates = sorted(Path(gates_dir).glob('G0.7.tuned_qrc.*.json'))
    if not candidates:
        return None
    path = candidates[-1]
    data = json.loads(path.read_text(encoding='utf-8'))
    return {
        'result': data.get('result'),
        'json': path.as_posix(),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'model_details': data.get('model_details') or {},
    }


def evaluate_config(
    config_path: Path, runs_csv: Path, gates_dir: Path
) -> Tuple[Optional[dict], Optional[str]]:
    """Evaluate the family for a config from its runs.csv and tuning records.

    Returns:
        (result or None, error) where error names the missing tuning record or runs.csv.
    """
    from qrc_thresher.config import load_config
    from qrc_thresher.proof.run_manifest import _config_hash
    from qrc_thresher.tuning import TuningRecordMissing, load_record

    cfg = load_config(Path(config_path))
    if cfg.tuning is None:
        return None, (f'{Path(config_path).as_posix()} has no tuning block, so no tuning record '
                      'and no comparative family (D011, D013)')
    records: Dict[str, dict] = {}
    for task in ('stm', 'parity', 'narma'):
        try:
            records[task] = load_record(Path(config_path), task)
        except TuningRecordMissing as exc:
            return None, str(exc)
    if not Path(runs_csv).exists():
        return None, f'no runs.csv at {Path(runs_csv).as_posix()}; run the sweep first'
    runs = pd.read_csv(
        runs_csv, keep_default_na=False,
        dtype={'sweep_id': str, 'tuning_record_sha': str, 'circuit_hash': str, 'config_hash': str},
    )
    protocol = load_protocol()
    designs = designs_from_records(cfg, records)
    sweeps = {task: r['sweep_id'] for task, r in records.items()}
    shas = {task: r['record_sha256'] for task, r in records.items()}
    if len(set(sweeps.values())) == 1:
        sweeps = next(iter(sweeps.values()))  # type: ignore[assignment]
        shas = next(iter(shas.values()))  # type: ignore[assignment]
    result = evaluate_family(
        runs, protocol, designs, g07_tuned=newest_g07_tuned(gates_dir),
        readout=cfg.reservoir.readout, config_hash=_config_hash(Path(config_path)),
        sweep_id=sweeps, tuning_record_sha=shas, measurement_model=cfg.measurement.model,
    )
    return result, None


__all__ = [
    'ComparativeProtocol',
    'Designs',
    'FAMILY_NAME',
    'FAMILY_SIZE',
    'INSUFFICIENT',
    'MEMBERS',
    'PROTOCOL_FILE',
    'comparator_arm',
    'designs_from_records',
    'evaluate_config',
    'evaluate_family',
    'load_protocol',
    'newest_g07_tuned',
    'protocol_sha256',
    'qrc_arm',
    'write_family_report',
]
