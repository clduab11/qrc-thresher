"""`qrc-thresher scorecard`: the plain-language scorecard generated from docs/evidence/*/
(docs/DECISIONS.md D019, ruling P4; D018 for the labels; CP5 item E.5).

The scorecard reads only the files listed in each evidence folder's MANIFEST.sha256, verifying
their hashes first. It finds the family record by name and content, its member views by content
(the view's ``gate`` is the member and its ``family_sha256`` the family file's SHA-256; exactly one
per member), G0.7 files keyed by (model, model_details.experiment_name) and legacy gates
``<name>.json`` or ``<name>.<stamp>.json``; for each key the newest stamp wins. One row per
registered gate: a gate with no evidence shows "missing", an INSUFFICIENT_EVIDENCE verdict shows
as such, and neither is ever a FAIL. Every number is read from the JSONs, never hand-copied. The
output is deterministic: the same evidence gives the same bytes and there is no generation
timestamp.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from qrc_thresher.config import MEASUREMENT_LABELS
from qrc_thresher.gates.comparative import VIEW_PROVENANCE_KEYS
from qrc_thresher.proof.evidence import (
    COMMIT_RE,
    MEMBERS,
    EvidenceError,
    kind_of,
    verify_manifest,
)

SCORECARD_COLUMNS = ('gate', 'plain-language claim', 'verdict', 'key numbers',
                     'measurement label', 'commit', 'config hash', 'evidence file')
GATES = ('G0', 'G0.5', 'G0.7', 'G1', 'G2', 'G2.5', 'G3', 'G4', 'G5', 'G6', 'G7')
LEGACY_GATES = ('G0.5', 'G5', 'G6', 'G7')
EXACT_LABEL = MEASUREMENT_LABELS['exact']  # one label table (D018, CP5b C1)
CLASSICAL_LABEL = MEASUREMENT_LABELS['classical']
MISSING = 'missing'
NOT_PUBLISHED = 'not published (health output stays local)'
_STAMP = re.compile(r'\.(\d{8}T\d{12}Z)(?:\.\d+)?\.json$')

# Plain-language claims, sourced from DECISIONS (D005/D008 for G0.7, D010/D011 for G0.5,
# D014 for G1-G4), never from BUILD_SPEC §15.
CLAIMS: Dict[str, str] = {
    'G0': 'The environment and harness are healthy (the health report; kept local).',
    'G0.5': 'The PennyLane reservoir and an independent Qiskit build of the same circuit give '
            'the same expectation values on every registered case, at one tolerance (D010, D011).',
    'G0.7': 'The model passes the memory sanity gate: on every seed pair the held-out linear '
            'readback of past inputs (k >= 1) and the window-2 parity beat a permutation null at '
            'the registered level (D005, D008).',
    'G1': '(a) The tuned STM design passes G0.7 v1, and (b) it beats its own no-entanglement '
          'ablation on parity accuracy, paired by seed, under the Holm-adjusted family rule '
          "(D014) (alpha_lite.yaml's 3 pairs; z_only; (b) chosen post hoc, D014).",
    'G2': 'The tuned parity design clears the accuracy floor and beats the tuned random kitchen '
          'sinks (RKS) baseline on parity accuracy, paired by seed (D014). A baseline comparison, '
          'not an entanglement claim.',
    'G2.5': 'The tuned STM design beats its matched Haar-random ablation on STM memory (k >= 1), '
            'paired by seed (D014).',
    'G3': 'The tuned STM design beats the tuned ESN on STM memory (k >= 1), paired by seed (D014).',
    'G4': 'The tuned NARMA design clears the NRMSE floor and beats the tuned ESN on NARMA-10 '
          'NRMSE (lower is better), paired by seed (D014).',
    'G5': 'Legacy gate, not registered in DECISIONS: its evaluator compares recorded STM memory '
          'across the backend_device values of runs.csv; it is not a full-circuit cross-backend '
          "check (the audit's defect register, deferred).",
    'G6': 'Legacy gate, not registered in DECISIONS: an importability checklist.',
    'G7': 'Legacy gate, not registered in DECISIONS: a calibration-file check.',
}
NOTES = (
    'G2 is a baseline comparison against the tuned random kitchen sinks, not an entanglement '
    'claim (D014).',
    'The parity-accuracy disclosure applies to G1 and G2: for a readout linear in the window '
    'inputs, a single-arm parity accuracy scatters on both sides of chance, so sub-0.5 '
    'single-arm accuracies in the default tables are not a finding; the paired margins cancel '
    "it by seed, and G2's registered accuracy floor sits above it (METHODOLOGY §1.2, D016).",
    'Every QRC number and verdict here is exact (oracle upper bound): exact expectation values, '
    'an upper bound on what a device could measure, and not a headline claim (D004); run 4 '
    "does not meet D004's finite-shot commitment, and finite-shot evidence is pending under a "
    'v2 protocol (D017).',
    'Classical comparators (ESN, RKS) are labelled by their kind, "classical, no measurement '
    'cost"; rows written before D018 recorded "exact" on classical arms.',
    'The ESN presets\' G0.7 rows show the classical label by kind; G0.7 v1 records "exact" '
    'by protocol.',
    'A gate without evidence in the folders read shows "missing"; INSUFFICIENT_EVIDENCE is '
    'shown as such. Neither is a FAIL.',
    'Each family member is followed by its default-design table, reported only and never '
    'gated (D014).',
)


def _g(value) -> str:
    if value is None:
        return 'n/a'
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f'{value:.4g}'
    return str(value)


def _cell(text: str) -> str:
    return str(text).replace('|', '\\|').replace('\n', ' ')


def _stamp(name: str) -> str:
    match = _STAMP.search(name)
    return match.group(1) if match else ''


def _short_commit(value) -> str:
    return str(value)[:7] if value else 'n/a'


def _short_hash(value) -> str:
    return str(value)[:8] if value else 'n/a'


# --- reading the folders -------------------------------------------------------------------------

class _Evidence:
    """Everything the scorecard cites, read from the manifests of the evidence folders."""

    def __init__(self, folders: Sequence[Path], docs_dir: Path) -> None:
        self.docs_dir = Path(docs_dir).resolve()
        self.families: List[dict] = []  # {data, ref, sha, stamp, views: {member: (ref, sha)}}
        self.g07: Dict[Tuple[str, str], dict] = {}
        self.legacy: Dict[str, dict] = {}
        for folder in sorted(Path(f) for f in folders):
            self._read_folder(folder)

    def _ref(self, folder: Path, name: str) -> str:
        try:
            return (folder / name).resolve().relative_to(self.docs_dir).as_posix()
        except ValueError:  # a folder outside docs/: cite it by its own name
            return f'{folder.name}/{name}'

    @staticmethod
    def _check_commit(value, ref: str) -> None:
        """A cited file's commit must be a hex hash: never 'unknown' or '-dirty' (CP5b C21)."""
        text = str(value or '')
        if (not text or text == 'unknown' or text.endswith('-dirty')
                or not COMMIT_RE.fullmatch(text)):
            raise EvidenceError(f'{ref}: its commit {text!r} is unknown, dirty or not a hash; '
                                'the scorecard cites clean commits only')

    def _read_folder(self, folder: Path) -> None:
        entries = verify_manifest(folder, complete=False)  # unlisted files are ignored
        loaded: Dict[str, Tuple[dict, str]] = {}
        for name, digest in entries.items():
            if not name.endswith('.json') or '/' in name:
                continue
            try:
                data = json.loads((folder / name).read_text(encoding='utf-8'))
            except (OSError, ValueError) as exc:
                raise EvidenceError(f'{folder.name}/{name}: unreadable JSON ({exc})') from exc
            loaded[name] = (data, digest)
        views = [(n, d, s) for n, (d, s) in loaded.items() if kind_of(d) == 'view']
        for name, (data, digest) in loaded.items():
            kind = kind_of(data)
            if kind == 'family' and re.match(r'COMPARATIVE\.v\d+\.', name):
                found: Dict[str, List[Tuple[str, str]]] = {m: [] for m in MEMBERS}
                for view_name, view, view_sha in views:
                    if view.get('family_sha256') == digest:
                        found[view['gate']].append((self._ref(folder, view_name), view_sha))
                bad = [f'{m}: {len(v)}' for m, v in found.items() if len(v) != 1]
                if bad:
                    raise EvidenceError(f'{folder.name}/{name}: expected exactly one view per '
                                        f'member; got {", ".join(bad)}')
                for view_name, view, _sha in views:
                    if view.get('family_sha256') != digest:
                        continue
                    member = data['members'].get(view['gate'])
                    if not isinstance(member, dict):
                        raise EvidenceError(f'{view_name}: its member is missing from the family')
                    # The view is what write_family_report writes: the member dict plus the
                    # family's provenance keys (CP5b.2 F1). (builder) A provenance key the view
                    # does not hold is tolerated: the pinned scorecard fixtures write a subset;
                    # one it holds must equal the family's value.
                    for key in VIEW_PROVENANCE_KEYS:
                        if key in view and view[key] != data.get(key):
                            raise EvidenceError(f'{view_name}: disagrees with its family on {key}')
                    for key, value in member.items():
                        if key in VIEW_PROVENANCE_KEYS:
                            continue
                        if key not in view or view[key] != value:
                            raise EvidenceError(f'{view_name}: disagrees with its family on {key}')
                self._check_commit(data.get('git_commit'), self._ref(folder, name))
                self.families.append({
                    'data': data, 'ref': self._ref(folder, name), 'sha': digest,
                    'stamp': _stamp(name), 'views': {m: v[0] for m, v in found.items()},
                    'key': (str(data.get('family')), str(data.get('version')),
                            str(data.get('config_hash'))),
                })
            elif kind == 'g07' and name.startswith('G0.7.'):
                details = data.get('model_details') or {}
                key = (str(data.get('model')), str(details.get('experiment_name')))
                entry = {'data': data, 'ref': self._ref(folder, name), 'sha': digest,
                         'stamp': _stamp(name)}
                self._check_commit(data.get('git_commit_hash') or (data.get('environment') or {})
                                   .get('git_commit_hash'), entry['ref'])
                if key not in self.g07 or entry['stamp'] > self.g07[key]['stamp']:
                    self.g07[key] = entry
            elif kind == 'legacy' and data.get('gate') in LEGACY_GATES:
                gate = str(data['gate'])
                if not (name == f'{gate}.json' or name.startswith(f'{gate}.')):
                    continue
                entry = {'data': data, 'ref': self._ref(folder, name), 'sha': digest,
                         'stamp': _stamp(name)}
                self._check_commit(data.get('git_commit_hash'), entry['ref'])
                if gate not in self.legacy or entry['stamp'] > self.legacy[gate]['stamp']:
                    self.legacy[gate] = entry

    def newest_families(self) -> List[dict]:
        """One family per (protocol, version, config_hash), the newest stamp of each, ordered by
        (config_hash, version) (CP5b ruling 13)."""
        newest: Dict[Tuple[str, str, str], dict] = {}
        for family in self.families:
            key = family['key']
            if key not in newest or family['stamp'] > newest[key]['stamp']:
                newest[key] = family
        return [newest[k] for k in sorted(newest, key=lambda k: (k[2], k[1], k[0]))]


# --- rows -------------------------------------------------------------------------------------

def _comparison_numbers(member: dict, comparison: dict, *, gated: bool) -> str:
    if comparison.get('status') != 'OK':
        return f"INSUFFICIENT_EVIDENCE: {comparison.get('reason') or member.get('message') or ''}"
    parts = [
        f"n_pairs = {comparison.get('n_pairs')}",
        f"mean A {_g(comparison.get('mean_a'))} vs B {_g(comparison.get('mean_b'))}",
        f"mean diff {_g(comparison.get('mean_diff'))} "
        f"[{_ci_label(comparison)} {_g(comparison.get('ci_low'))}, "
        f"{_g(comparison.get('ci_high'))}]",
        f"d_z {_g(comparison.get('d_z'))}",
    ]
    if gated:
        parts.append(f"raw p {_g(member.get('raw_p'))}; Holm p {_g(member.get('adjusted_p'))}")
        floor = member.get('floor')
        if floor:
            state = 'met' if floor.get('passed') else 'not met'
            parts.append(f"floor {floor.get('metric')} {_rule(floor.get('rule'))} "
                         f"{_g(floor.get('value'))} {state} (observed {_g(floor.get('observed'))})")
        if member.get('baseline_better'):
            parts.append(f"baseline better (two-sided p {_g(member.get('p_two_sided'))})")
    else:
        parts.append(f"one-sided p {_g(comparison.get('p_one_sided'))} (reported only, never in "
                     'the Holm family)')
    return '; '.join(parts)


def _ci_label(comparison: dict) -> str:
    """'BCa <level>' from comparison['ci_level'] (CP5b C22). Every compare_arms record carries
    ci_level (metrics/paired.py since its first commit); the 0.95 default, COMPARATIVE.v1's
    registered level, serves hand-built records only (CP5b ruling 19)."""
    level = comparison.get('ci_level', 0.95)
    return f'BCa {float(level):.0%}'


def _rule(rule) -> str:
    return {'mean_greater_than': '>', 'mean_less_than': '<'}.get(str(rule), str(rule))


def _member_label(member: dict, family: dict) -> str:
    """'a: <a>; b: <b>' when the arms differ, else the one label. A member without
    measurement_labels (written before CP5) is labelled by kind through comparative.arm_labels
    (CP5b C1); if its baseline name does not parse, the family label."""
    from qrc_thresher.gates.comparative import arm_labels

    labels = member.get('measurement_labels') or {}
    if not labels:
        try:
            labels = arm_labels(str(member.get('baseline')),
                                str(family.get('measurement_model') or 'exact'))
        except ValueError:
            labels = {}
    if labels and labels.get('a') != labels.get('b'):
        return f"a: {labels['a']}; b: {labels['b']}"
    if labels:
        return str(labels.get('a'))
    return str(family.get('measurement_label') or EXACT_LABEL)


def _member_rows(ev: _Evidence, gate: str) -> List[List[str]]:
    families = ev.newest_families()
    if not families:
        return [[gate, CLAIMS[gate], MISSING, 'n/a', 'n/a', 'n/a', 'n/a', 'n/a']]
    rows: List[List[str]] = []
    for family in families:
        rows += _family_member_rows(family, gate)
    return rows


def _family_member_rows(family: dict, gate: str) -> List[List[str]]:
    data = family['data']
    member = data['members'].get(gate)
    if member is None:
        return [[gate, CLAIMS[gate], MISSING, 'n/a', 'n/a', 'n/a', 'n/a', 'n/a']]
    ref, sha = family['views'][gate]
    verdict = str(member.get('result'))
    numbers = _comparison_numbers(member, member.get('comparison') or {}, gated=True)
    if gate == 'G1' and member.get('g07'):
        numbers += f"; G0.7 v1 on the tuned design: {member['g07'].get('result')}"
    rows = [[gate, CLAIMS[gate], verdict, numbers, _member_label(member, data),
             _short_commit(data.get('git_commit')), _short_hash(data.get('config_hash')),
             f'{ref} (sha256 {sha[:12]})']]
    default = member.get('default') or {}
    w1 = member.get('default_w1') or {}
    default_numbers = _comparison_numbers(member, default, gated=False)
    if w1:
        w1_status = w1.get('status')
        if w1_status is not None and w1_status != 'OK':
            default_numbers += f"; default_w1 {w1_status}: {w1.get('reason')}"
        else:
            default_numbers += (f"; default_w1 mean {_g(w1.get('mean'))} (std {_g(w1.get('std'))}, "
                                f"n_rows {w1.get('n_rows')})")
    status = default.get('status', 'n/a')
    verdict_cell = 'reported only (no verdict)' if status == 'OK' else f'reported only: {status}'
    rows.append([gate, f'{gate} default-design table (reported only, never gated; D014)',
                 verdict_cell, default_numbers,
                 _member_label(member, data), _short_commit(data.get('git_commit')),
                 _short_hash(data.get('config_hash')), f'{ref} (sha256 {sha[:12]})'])
    return rows


def _g07_rows(ev: _Evidence) -> List[List[str]]:
    if not ev.g07:
        return [['G0.7', CLAIMS['G0.7'], MISSING, 'n/a', 'n/a', 'n/a', 'n/a', 'n/a']]
    rows = []
    for (model, experiment), entry in sorted(ev.g07.items()):
        data = entry['data']
        details = data.get('model_details') or {}
        clauses = data.get('clauses') or {}
        numbers = (f"model {model}; experiment {experiment}; n_seeds = {data.get('n_seeds')}; "
                   f"STM clause {(clauses.get('stm') or {}).get('result')}; parity clause "
                   f"{(clauses.get('parity') or {}).get('result')}; protocol v"
                   f"{data.get('protocol_version')}; stamp {entry['stamp']}")
        label = CLASSICAL_LABEL if details.get('kind') == 'classical' else str(
            data.get('measurement_label') or EXACT_LABEL)
        commit = data.get('git_commit_hash') or (data.get('environment') or {}).get(
            'git_commit_hash')
        if data.get('config_hash'):
            config = _short_hash(data['config_hash'])
        elif details.get('tuning_config_hash'):
            config = f"{_short_hash(details['tuning_config_hash'])} (tuning config)"
        else:
            config = 'n/a'
        rows.append(['G0.7', CLAIMS['G0.7'], str(data.get('result')), numbers, label,
                     _short_commit(commit), config, f"{entry['ref']} (sha256 {entry['sha'][:12]})"])
    return rows


def _legacy_row(ev: _Evidence, gate: str) -> List[str]:
    entry = ev.legacy.get(gate)
    if entry is None:
        return [gate, CLAIMS[gate], MISSING, 'n/a', 'n/a', 'n/a', 'n/a', 'n/a']
    data = entry['data']
    evidence = data.get('evidence') or {}
    parts = []
    for key in ('n_cases', 'all_match', 'max_abs_diff', 'tolerance', 'message'):
        if key in evidence:
            parts.append(f'{key} {_g(evidence[key])}')
    return [gate, CLAIMS[gate], str(data.get('result')), '; '.join(parts) or 'n/a',
            str(data.get('measurement_label') or 'n/a'), _short_commit(data.get('git_commit_hash')),
            _short_hash(data.get('config_hash')), f"{entry['ref']} (sha256 {entry['sha'][:12]})"]


def build_scorecard(evidence_dirs: Sequence[Path], docs_dir: Path) -> str:
    """The scorecard markdown for the evidence folders (deterministic; no timestamp).

    Raises:
        EvidenceError: When a folder's MANIFEST.sha256 is missing or a listed file does not
            match it, or a family lacks exactly one view per member.
    """
    ev = _Evidence(evidence_dirs, docs_dir)
    rows: List[List[str]] = [['G0', CLAIMS['G0'], NOT_PUBLISHED, 'n/a', 'n/a', 'n/a', 'n/a', 'n/a']]
    rows.append(_legacy_row(ev, 'G0.5'))
    rows += _g07_rows(ev)
    for member in MEMBERS:
        rows += _member_rows(ev, member)
    for gate in ('G5', 'G6', 'G7'):
        rows.append(_legacy_row(ev, gate))

    folders = ', '.join(f'`{Path(f).name}`' for f in sorted(Path(d) for d in evidence_dirs))
    lines = ['# Scorecard', '',
             f'Generated by `qrc-thresher scorecard` from the evidence folder(s) {folders} under '
             '`docs/evidence/`, reading only the files listed in each folder\'s MANIFEST.sha256 '
             'after verifying their hashes. Every number is read from the cited file '
             '(docs/DECISIONS.md D019, ruling P4).', '',
             '| ' + ' | '.join(SCORECARD_COLUMNS) + ' |',
             '|' + '|'.join(' --- ' for _ in SCORECARD_COLUMNS) + '|']
    for row in rows:
        lines.append('| ' + ' | '.join(_cell(c) for c in row) + ' |')
    lines += ['', '## Notes', '']
    lines += [f'- {note}' for note in NOTES]
    lines.append('')
    return '\n'.join(lines)


def scorecard_handler(evidence: Sequence[str], out: Optional[str]) -> int:
    """CLI handler for `qrc-thresher scorecard`. Returns exit code."""
    docs_dir = Path('docs')
    folders = [Path(e) for e in evidence] if evidence else sorted(
        p for p in (docs_dir / 'evidence').glob('*') if p.is_dir()
    )
    if not folders:
        print('scorecard: no evidence folder under docs/evidence/')
        return 1
    try:
        text = build_scorecard(folders, docs_dir)
    except EvidenceError as exc:
        print(f'scorecard: refused; {exc}')
        return 1
    out_path = Path(out) if out else docs_dir / 'scorecard.md'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(text.encode('utf-8'))
    print(f'scorecard written to {out_path.as_posix()}')
    return 0


__all__ = ['CLAIMS', 'GATES', 'NOTES', 'SCORECARD_COLUMNS', 'EvidenceError', 'build_scorecard',
           'scorecard_handler']
