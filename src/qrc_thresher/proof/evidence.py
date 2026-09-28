"""`qrc-thresher evidence`: byte-exact export of gate records to docs/evidence/<short-commit>/
(docs/DECISIONS.md D019, ruling P4; CP5 item E.1).

The export takes one or more gate JSONs under results/gates. A family record brings along its
member views (found by content: the view's ``gate`` is a member and its ``family_sha256`` is the
SHA-256 of the family bytes), the G0.7 file it names (by basename, in the same folder), the three
tuning records behind it and the config file behind it (copied byte-exact into configs/). A G0.7
file brings its figure. Optionally the rows of the family's sweep are written as
runs.sanitised.csv, allowlisted columns only, every cell the identical text of its source.

Nothing is re-serialised, no recorded path is rewritten and no environment block is stripped:
every copy is ``shutil.copyfile``. Before anything is written, every hash is verified (views
against the family bytes, the G0.7 sha the family recorded, each tuning record's own sha and the
family's tuning_record_sha, the config's canonical hash against the family's config_hash) and
every gate JSON and row must carry the same clean commit ('unknown' and '-dirty' are refused).
The export is built in a staging folder outside the repository, scanned for local paths and
forbidden keys, and moved into place with one os.replace; an existing target is never touched.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

MEMBERS = ('G1', 'G2', 'G2.5', 'G3', 'G4')
TASKS = ('stm', 'parity', 'narma')
FORBIDDEN_KEYS = ('cli_command', 'git_branch', 'platform', 'artifact_paths', 'package_versions')
# The row columns that leave the repository (E.1 minimum plus CP5a ruling 10). Never
# cli_command, git_branch, platform, artifact_paths, package_versions, config_path,
# timestamps or failure_reason (an exception text can carry a path).
ROW_COLUMNS = (
    'run_id', 'success', 'git_commit_hash', 'config_hash', 'sweep_id', 'tuning_record_sha',
    'circuit_hash', 'task_seed', 'reservoir_seed', 'task_name', 'design', 'primary_metric_name',
    'primary_metric_value', 'measurement_model', 'n_configs', 'n_validation_evals',
    'secondary_metrics', 'python_version', 'backend_device', 'device', 'precision',
    'entanglement_metric', 'runtime_per_stage_seconds',
)
DROPPED_COLUMNS = ('cli_command', 'git_branch', 'platform', 'artifact_paths', 'package_versions',
                   'config_path', 'timestamp_utc', 'failure_reason')
TEXT_SUFFIXES = {'.json', '.csv', '.md', '.sha256', '.yaml', '.yml', '.gitattributes'}
MANIFEST = 'MANIFEST.sha256'
README = 'README.md'
ROWS_FILE = 'runs.sanitised.csv'
CP4C_SEQUENCE = (
    'qrc-thresher tune --config configs/comparative.yaml',
    'qrc-thresher run stm --config configs/comparative.yaml',
    'qrc-thresher run parity --config configs/comparative.yaml',
    'qrc-thresher run narma --config configs/comparative.yaml',
    'qrc-thresher run parity --config configs/comparative.yaml --design-task stm',
    'qrc-thresher run stm --config configs/comparative.yaml --design default',
    'qrc-thresher run parity --config configs/comparative.yaml --design default',
    'qrc-thresher run narma --config configs/comparative.yaml --design default',
    'qrc-thresher ablation no_entangle stm --config configs/comparative.yaml',
    'qrc-thresher ablation no_entangle parity --config configs/comparative.yaml --design-task stm',
    'qrc-thresher ablation no_entangle stm --config configs/comparative.yaml --design default',
    'qrc-thresher ablation no_entangle parity --config configs/comparative.yaml --design default',
    'qrc-thresher ablation haar stm --config configs/comparative.yaml',
    'qrc-thresher ablation haar stm --config configs/comparative.yaml --design default',
    'qrc-thresher baseline stm --config configs/comparative.yaml',
    'qrc-thresher baseline stm --config configs/comparative.yaml --design default',
    'qrc-thresher baseline parity --config configs/comparative.yaml',
    'qrc-thresher baseline parity --config configs/comparative.yaml --design default',
    'qrc-thresher baseline narma --config configs/comparative.yaml',
    'qrc-thresher baseline narma --config configs/comparative.yaml --design default',
    'qrc-thresher gate G0.7 --config configs/alpha_lite.yaml --model tuned_qrc '
    '--tuning-config configs/comparative.yaml',
    'qrc-thresher gate family --config configs/comparative.yaml',
    'qrc-thresher summary --phase cp4c',
)

# The sweep whose export carries the CP4c command sequence in its README: run 4 (D019).
RUN4_SWEEP_ID = '20260924T214530147999Z'
LEGACY_GATES = ('G0.5', 'G5', 'G6', 'G7')  # the only legacy JSONs the export accepts (E.1)

COMMIT_RE = re.compile(r'[0-9a-f]{7,40}')  # a clean git hash (CP5b.2 F5)

_DRIVE = re.compile(r'(?<![A-Za-z])[A-Za-z]:[\\/]')
_HOME = re.compile(r'/Users/|/home/')
_JSON_KEY = re.compile(r'"(?:' + '|'.join(FORBIDDEN_KEYS) + r')"\s*:')
_YAML_KEY = re.compile(r'^\s*(?:-\s+)?["\']?(?:' + '|'.join(FORBIDDEN_KEYS) + r')["\']?\s*:',
                       re.MULTILINE)
_ABS_CONFIG_PATH = re.compile(
    r'config_path["\']?\s*[:=]\s*["\']?(?:[A-Za-z]:[\\/]|/|\\\\)'  # drive, POSIX root or UNC
)


class EvidenceError(Exception):
    """The export or the scorecard refused; the message names the check that failed."""


# --- hashing and scanning -----------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def scan_text(text: str, suffix: Optional[str] = None) -> List[str]:
    """The privacy scan of one exported text file: a list of hits, empty when clean.

    Hits: a drive-letter path ('https://' does not match), a /Users/ or /home/ path, a
    forbidden key as a JSON object key, a YAML mapping key or a CSV header cell (the first line,
    parsed by csv.reader, for a .csv file or when ``suffix`` is unknown), and an absolute
    config_path value (drive letter, POSIX root or UNC). Prose mentions of the column names are
    allowed.
    """
    hits: List[str] = []
    for match in _DRIVE.finditer(text):
        hits.append(f'local path {match.group(0)!r}')
    for match in _HOME.finditer(text):
        hits.append(f'local path {match.group(0)!r}')
    for match in _JSON_KEY.finditer(text):
        hits.append(f'forbidden JSON key {match.group(0)!r}')
    for match in _YAML_KEY.finditer(text):
        hits.append(f'forbidden YAML key {match.group(0).strip()!r}')
    first = text.splitlines()[0] if text else ''
    if first and (suffix is None or suffix.lower() == '.csv'):
        try:
            cells = {cell.strip() for cell in next(csv.reader([first]))}
        except (csv.Error, StopIteration):
            cells = set()
        for name in FORBIDDEN_KEYS:
            if name in cells:
                hits.append(f'forbidden CSV column {name!r}')
    for match in _ABS_CONFIG_PATH.finditer(text):
        hits.append(f'absolute config_path {match.group(0)!r}')
    return hits


def _sort_key(root: Path):
    """Case-insensitive order by path parts: the committed order on every platform (C11)."""
    return lambda p: [s.lower() for s in Path(p).relative_to(root).parts]


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.name.lower() in TEXT_SUFFIXES


def scan_folder(folder: Path) -> Dict[str, List[str]]:
    """scan_text over every text file under ``folder`` (the PNG is skipped)."""
    hits: Dict[str, List[str]] = {}
    folder = Path(folder)
    for path in sorted((p for p in folder.rglob('*') if p.is_file()), key=_sort_key(folder)):
        if not is_text_file(path):
            continue
        try:
            text = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            hits[path.relative_to(folder).as_posix()] = ['not UTF-8']
            continue
        found = scan_text(text, path.suffix)
        if found:
            hits[path.relative_to(folder).as_posix()] = found
    return hits


def write_manifest(folder: Path) -> Dict[str, str]:
    """MANIFEST.sha256 in sha256sum format (LF, POSIX relative paths) over every file but
    itself."""
    folder = Path(folder)
    entries: Dict[str, str] = {}
    for path in sorted((p for p in folder.rglob('*') if p.is_file()), key=_sort_key(folder)):
        rel = path.relative_to(folder).as_posix()
        if rel == MANIFEST:
            continue
        entries[rel] = sha256_file(path)
    text = ''.join(f'{digest}  {name}\n' for name, digest in entries.items())
    (folder / MANIFEST).write_bytes(text.encode('utf-8'))
    return entries


def verify_manifest(folder: Path, *, complete: bool = True) -> Dict[str, str]:
    """Parse and verify MANIFEST.sha256: every listed file hashes as recorded and, when
    ``complete``, every file in the folder except the manifest is listed (the scorecard reads
    only listed files and ignores the rest).

    Raises:
        EvidenceError: On a missing manifest, a missing file, a hash mismatch or, when
            ``complete``, an unlisted file.
    """
    folder = Path(folder)
    manifest = folder / MANIFEST
    if not manifest.is_file():
        raise EvidenceError(f'{folder.name}: no {MANIFEST}')
    entries: Dict[str, str] = {}
    for line in manifest.read_bytes().decode('utf-8').splitlines():
        if not line.strip():
            continue
        digest, sep, name = line.partition('  ')
        if not sep or len(digest) != 64:
            raise EvidenceError(f'{folder.name}/{MANIFEST}: unreadable line {line!r}')
        entries[name] = digest
    for name, digest in entries.items():
        path = folder / name
        if not path.is_file():
            raise EvidenceError(f'{folder.name}: {name} is listed in {MANIFEST} but missing')
        if sha256_file(path) != digest:
            raise EvidenceError(f'{folder.name}: {name} does not match its {MANIFEST} hash')
    if complete:
        present = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
        unlisted = sorted(present - set(entries) - {MANIFEST})
        if unlisted:
            raise EvidenceError(f'{folder.name}: files not listed in {MANIFEST}: {unlisted}')
    return entries


# --- classification -----------------------------------------------------------------------------

def _load(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise EvidenceError(f'{Path(path).name}: unreadable JSON ({exc})') from exc


def kind_of(data: dict) -> str:
    """'family', 'view', 'g07' or 'legacy' from a gate JSON's content."""
    if data.get('family') == 'COMPARATIVE' and isinstance(data.get('members'), dict):
        return 'family'
    if 'family_sha256' in data and data.get('gate') in MEMBERS:
        return 'view'
    if data.get('gate') == 'G0.7':
        return 'g07'
    return 'legacy'


def _commit_of(kind: str, data: dict, name: str) -> str:
    if kind == 'family':
        value = data.get('git_commit')
    elif kind == 'g07':
        value = data.get('git_commit_hash') or (data.get('environment') or {}).get(
            'git_commit_hash')
    else:
        value = data.get('git_commit_hash')
    if not value:
        raise EvidenceError(f'{name}: no git commit recorded')
    return str(value)


def _check_commits(commits: Dict[str, str]) -> str:
    """Every gate JSON and row carries the same clean commit; returns it."""
    for name, commit in commits.items():
        if commit == 'unknown':
            raise EvidenceError(f"{name}: commit 'unknown' is refused")
        if commit.endswith('-dirty'):
            raise EvidenceError(f'{name}: commit {commit} is -dirty and refused')
        if not COMMIT_RE.fullmatch(commit):
            raise EvidenceError(f'{name}: commit {commit!r} is not a git hash')
    distinct = sorted(set(commits.values()))
    if len(distinct) != 1:
        raise EvidenceError(f'mixed commits across the exported files and rows: {distinct}')
    return distinct[0]


def _views_by_content(folder: Path, family_path: Path, family_sha: str) -> Dict[str, Path]:
    """Exactly one member view per member, found by content (never by a filename glob)."""
    found: Dict[str, List[Path]] = {m: [] for m in MEMBERS}
    for path in sorted(folder.glob('*.json')):
        if path == family_path:
            continue
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if kind_of(data) == 'view' and data.get('family_sha256') == family_sha:
            found[data['gate']].append(path)
    problems = [f'{m}: {len(paths)} view(s)' for m, paths in found.items() if len(paths) != 1]
    if problems:
        raise EvidenceError(
            f'{family_path.name}: expected exactly one member view per member with '
            f'family_sha256 {family_sha[:12]}...; got {", ".join(problems)}'
        )
    return {m: paths[0] for m, paths in found.items()}


def _sweeps(sweep_id) -> List[str]:
    if isinstance(sweep_id, dict):
        return sorted({str(v) for v in sweep_id.values() if v})
    return [str(sweep_id)] if sweep_id else []


# --- the export --------------------------------------------------------------------------------

def export_evidence(
    gate_jsons: Sequence[Path],
    *,
    out_root: Path,
    config_path: Optional[Path] = None,
    runs_csv: Optional[Path] = None,
    results_dir: Path = Path('results'),
) -> Path:
    """Export gate records to ``out_root/<short-commit>/``; returns the folder written.

    Args:
        gate_jsons: Family records, G0.7 files and legacy gate JSONs under results/gates.
        out_root: The evidence root (docs/evidence). The target folder must not exist.
        config_path: The config file behind the family, or None to use the tuning records'
            relative config_path. Its canonical hash must equal the family's config_hash.
        runs_csv: The runs.csv whose rows of the family's config and sweep are sanitised and
            exported; None exports no rows.
        results_dir: The results/ folder holding tuning/<config_hash>/<task>.json.

    Raises:
        EvidenceError: On any failed check; nothing is written under ``out_root`` then.
    """
    from qrc_thresher.proof.run_manifest import _config_hash
    from qrc_thresher.tuning import record_sha256

    results_dir = Path(results_dir)
    if config_path is not None and not Path(config_path).is_file():
        raise EvidenceError(f'--config {Path(config_path).as_posix()}: not a file')
    unique: List[Path] = []
    for given in gate_jsons:
        candidate = Path(given)
        if candidate.is_file() and any(candidate.resolve() == u.resolve() for u in unique):
            continue
        unique.append(candidate)
    gate_jsons = unique
    copies: List[Tuple[Path, str]] = []  # (source, relative destination)
    commits: Dict[str, str] = {}
    families: List[dict] = []
    readme_files: Dict[str, str] = {}  # destination -> recorded path
    sweep_note: List[str] = []
    for given in gate_jsons:
        src = Path(given)
        if not src.is_file():
            raise EvidenceError(f'{src.as_posix()}: not a file')
        data = _load(src)
        kind = kind_of(data)
        if kind == 'view':
            raise EvidenceError(f'{src.name}: a member view is exported with its family record, '
                                'not on its own')
        if kind == 'legacy' and data.get('gate') not in LEGACY_GATES:
            raise EvidenceError(f'{src.name}: gate {data.get("gate")!r} is not exported; the '
                                f'export takes family records, G0.7 files and the legacy gates '
                                f'{list(LEGACY_GATES)} only, and any health output stays out '
                                '(item E.1)')
        commits[src.name] = _commit_of(kind, data, src.name)
        copies.append((src, src.name))
        readme_files[src.name] = f'results/gates/{src.name}'
        if kind == 'g07':
            figure = data.get('figure')
            if figure:
                fig_path = src.parent / str(figure)
                if not fig_path.is_file():
                    raise EvidenceError(f'{src.name}: its figure {figure} is missing')
                copies.append((fig_path, fig_path.name))
                readme_files[fig_path.name] = f'results/gates/{fig_path.name}'
        if kind != 'family':
            continue

        family_bytes = src.read_bytes()
        family_sha = sha256_bytes(family_bytes)
        views = _views_by_content(src.parent, src, family_sha)
        for member, view in views.items():
            copies.append((view, view.name))
            readme_files[view.name] = f'results/gates/{view.name}'
            commits[view.name] = str(_load(view).get('git_commit') or commits[src.name])
        # The G0.7 file the family names (by basename, same folder), bound by its sha.
        g07_entry = (data['members'].get('G1') or {}).get('g07') or {}
        g07_json = g07_entry.get('json')
        if g07_json:
            g07_path = src.parent / Path(str(g07_json)).name
            if not g07_path.is_file():
                raise EvidenceError(f'{src.name}: the G0.7 file it names, {g07_path.name}, is '
                                    'missing from the same folder')
            if sha256_file(g07_path) != g07_entry.get('sha256'):
                raise EvidenceError(f'{src.name}: {g07_path.name} does not match the recorded '
                                    'members.G1.g07.sha256')
            g07_data = _load(g07_path)
            commits[g07_path.name] = _commit_of('g07', g07_data, g07_path.name)
            if all(g07_path != c[0] for c in copies):
                copies.append((g07_path, g07_path.name))
                readme_files[g07_path.name] = f'results/gates/{g07_path.name}'
                figure = g07_data.get('figure')
                if figure:
                    fig_path = g07_path.parent / str(figure)
                    if not fig_path.is_file():
                        raise EvidenceError(f'{g07_path.name}: its figure {figure} is missing')
                    copies.append((fig_path, Path(str(figure)).name))
                    readme_files[Path(str(figure)).name] = f'results/gates/{fig_path.name}'
        # The tuning records, each self-verifying and equal to the family's tuning_record_sha.
        config_hash = str(data.get('config_hash'))
        recorded_shas = data.get('tuning_record_sha')
        records: Dict[str, dict] = {}
        for task in TASKS:
            rec_path = results_dir / 'tuning' / config_hash / f'{task}.json'
            if not rec_path.is_file():
                raise EvidenceError(f'{src.name}: tuning record {rec_path.as_posix()} is missing')
            record = _load(rec_path)
            if record_sha256(record) != record.get('record_sha256'):
                raise EvidenceError(f'{rec_path.name}: record_sha256 does not recompute')
            expected = (recorded_shas.get(task) if isinstance(recorded_shas, dict)
                        else recorded_shas)
            if str(record.get('record_sha256')) != str(expected):
                raise EvidenceError(f'{rec_path.name}: record_sha256 differs from the family\'s '
                                    f'tuning_record_sha for {task}')
            records[task] = record
            copies.append((rec_path, f'tuning/{task}.json'))
            readme_files[f'tuning/{task}.json'] = f'results/tuning/{config_hash}/{task}.json'
        # The config, byte-exact, whose canonical hash is the family's config_hash.
        candidates: List[Path] = [Path(config_path)] if config_path is not None else []
        for record in records.values():
            recorded = str(record.get('config_path') or '')
            if recorded:
                candidates += [Path(recorded), results_dir.parent / recorded]
        cfg = next((c for c in candidates if c.is_file()), None)
        if cfg is None:
            raise EvidenceError(f'{src.name}: no config file found (tried '
                                f'{[c.as_posix() for c in candidates]})')
        if _config_hash(cfg) != config_hash:
            raise EvidenceError(f'{cfg.name}: its canonical config_hash {_config_hash(cfg)[:12]}'
                                f'... differs from the family\'s {config_hash[:12]}...')
        copies.append((cfg, f'configs/{cfg.name}'))
        readme_files[f'configs/{cfg.name}'] = f'configs/{cfg.name}'
        families.append({'data': data, 'path': src, 'config_hash': config_hash,
                         'sweeps': _sweeps(data.get('sweep_id')), 'config': cfg})

    rows_text: Optional[str] = None
    if runs_csv is not None:
        if len(families) != 1:
            raise EvidenceError('runs.csv is exported for exactly one family record')
        rows_text, row_commits, n_rows = _sanitise_rows(Path(runs_csv), families[0])
        commits.update(row_commits)
        sweep_note.append(f'{n_rows} rows of sweep(s) {", ".join(families[0]["sweeps"])}')

    commit = _check_commits(commits)
    short = commit[:7]
    out_root = Path(out_root)
    target = out_root / short
    if target.exists():
        raise EvidenceError(f'{target.as_posix()} exists; an evidence folder is never overwritten '
                            f'or merged ({short})')

    # The same file may be named twice (a G0.7 file given on the command line and named by the
    # family); two *different* files bound for one destination are refused.
    by_dest: Dict[str, Path] = {}
    deduped: List[Tuple[Path, str]] = []
    for source, dest in copies:
        resolved = source.resolve()
        if dest in by_dest:
            if by_dest[dest] != resolved:
                raise EvidenceError(f'two different files would be written to {dest}')
            continue
        by_dest[dest] = resolved
        deduped.append((source, dest))
    copies = deduped

    staging_root = Path(tempfile.mkdtemp(prefix='qrc-evidence-'))
    try:
        staged = staging_root / short
        staged.mkdir()
        for src, dest in copies:
            (staged / dest).parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copyfile(src, staged / dest)
            except OSError as exc:
                raise EvidenceError(f'{dest}: could not copy {src.as_posix()} ({exc})') from exc
            if sha256_file(staged / dest) != sha256_file(src):
                raise EvidenceError(f'{dest}: the staged copy differs from its source')
        if rows_text is not None:
            (staged / ROWS_FILE).write_bytes(rows_text.encode('utf-8'))
            readme_files[ROWS_FILE] = ('the rows of the family\'s config and sweep in '
                                       'results/runs.csv, allowlisted columns only')
        (staged / README).write_bytes(
            _readme(short, families, readme_files, staged, sweep_note).encode('utf-8')
        )
        write_manifest(staged)
        hits = scan_folder(staged)  # the manifest text included
        if hits:
            raise EvidenceError(f'the export would leak: {hits}')
        created_out_root = False
        if not out_root.exists():
            out_root.mkdir(parents=True)
            created_out_root = True
        try:
            os.replace(staged, target)
        except OSError as exc:
            if created_out_root:
                try:
                    out_root.rmdir()  # only when empty; never a tree removal
                except OSError:
                    pass
            raise EvidenceError(
                f'could not move the staged export into {target.as_posix()} ({exc}); nothing '
                'was written there and the export is not copied file by file; set TMPDIR/TEMP '
                f'to a folder on the same filesystem as {out_root.as_posix()}, outside the '
                'repository'
            ) from exc
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return target


def _sanitise_rows(runs_csv: Path, family: dict) -> Tuple[str, Dict[str, str], int]:
    """The rows of the family's config and sweep(s), allowlisted columns, cells as text, LF."""
    if not runs_csv.is_file():
        raise EvidenceError(f'{runs_csv.as_posix()}: not a file')
    with runs_csv.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    missing = [c for c in ROW_COLUMNS if c not in header]
    if missing:
        raise EvidenceError(f'{runs_csv.name}: allowlisted column(s) missing: {missing}')
    config_hash, sweeps = family['config_hash'], set(family['sweeps'])
    kept, commits = [], {}
    for row in rows:
        if str(row.get('config_hash', '')) != config_hash:
            continue
        if str(row.get('sweep_id', '')) not in sweeps:
            if str(row.get('success', '')).lower() == 'true':
                raise EvidenceError(
                    f"{runs_csv.name}: successful row {row.get('run_id')} of config "
                    f"{config_hash[:8]}... lies outside the family's sweep(s) "
                    f"{sorted(sweeps)}; the recorded n_rows counts every successful row of the "
                    "config, so a subset cannot be exported"
                )
            continue
        kept.append(row)
        commits[f"row {row.get('run_id')}"] = str(row.get('git_commit_hash', ''))
    if not kept:
        raise EvidenceError(f'{runs_csv.name}: no rows of config {config_hash[:8]}... and sweep(s) '
                            f'{sorted(sweeps)}')
    lines = [','.join(ROW_COLUMNS)]
    for row in kept:
        lines.append(','.join(_csv_cell(row.get(c, '')) for c in ROW_COLUMNS))
    return '\n'.join(lines) + '\n', commits, len(kept)


def _csv_cell(text: str) -> str:
    text = '' if text is None else str(text)
    if any(ch in text for ch in ',"\n\r'):
        return '"' + text.replace('"', '""') + '"'
    return text


def _readme(short: str, families: List[dict], files: Dict[str, str], staged: Path,
            notes: Iterable[str]) -> str:
    lines = [f'# Evidence exported at commit {short}', '']
    lines.append('Byte-exact copies of gate records from `results/`, exported by '
                 '`qrc-thresher evidence` (docs/DECISIONS.md D019, ruling P4). Nothing here is '
                 're-serialised; every recorded path inside the files is as recorded.')
    lines.append('')
    for family in families:
        data = family['data']
        lines.append(f"- Family: `{family['path'].name}`; protocol {data.get('family')} "
                     f"v{data.get('version')}; config_hash `{family['config_hash']}`; "
                     f"sweep_id {', '.join(family['sweeps']) or 'none'}; "
                     f"commit {data.get('git_commit')}.")
    for note in notes:
        lines.append(f'- Rows: {note}.')
    for family in families:  # one pointer per family that is not run 4 (CP5b ruling 18)
        if RUN4_SWEEP_ID in family['sweeps']:
            continue
        cfg_name = f'configs/{family["config"].name}'
        lines.append(f'- Reproduction: `qrc-thresher tune --config {cfg_name}`, then the run, '
                     'ablation, baseline and gate commands of that config (BUILD_SPEC App. F.2); '
                     'the full command sequence is recorded only for run 4 (D019).')
    lines += ['', '## Files', '', '| file | sha256 | recorded as |', '|---|---|---|']
    for path in sorted((p for p in staged.rglob('*') if p.is_file()), key=_sort_key(staged)):
        rel = path.relative_to(staged).as_posix()
        if rel in (README, MANIFEST):
            continue
        lines.append(f'| `{rel}` | `{sha256_file(path)}` | {files.get(rel, "")} |')
    lines += ['', 'MANIFEST.sha256 lists every file in this folder except itself, in sha256sum '
              'format; this README is listed there and does not hash the manifest.', '',
              '## Path mappings', '', '| recorded path | in this folder |', '|---|---|']
    lines.append('| `results/gates/<X>` | `<X>` |')
    for family in families:
        cfg = family['config']
        lines.append(f"| `results/tuning/{family['config_hash']}/<task>.json` | "
                     f'`tuning/<task>.json` |')
        for task in TASKS:
            lines.append(f"| results/tuning/{family['config_hash']}/{task}.json | "
                         f'tuning/{task}.json |')
        lines.append(f'| configs/{cfg.name} | configs/{cfg.name} |')
    for rel, recorded in sorted(files.items()):
        if recorded.startswith('results/gates/'):
            lines.append(f'| {recorded} | {rel} |')
    if ROWS_FILE in files:
        lines += ['', '## Dropped row columns', '',
                  'runs.sanitised.csv keeps only the allowlisted columns (`'
                  + '`, `'.join(ROW_COLUMNS)
                  + '`). These columns of results/runs.csv were dropped because they can carry a '
                  'local path, a machine identity or a timestamp: `' + '`, `'.join(DROPPED_COLUMNS)
                  + '`. Every kept cell is the identical text of its source cell (metric strings '
                  'are never passed through a float).']
    if any(RUN4_SWEEP_ID in family['sweeps'] for family in families):
        lines += ['', '## The CP4c command sequence', '',
                  'The runbook first archived results/runs.csv, results/experiments.db and '
                  'results/tuning/<config_hash>/, then ran, in this order:', '', '```']
        lines += list(CP4C_SEQUENCE)
        lines += ['```']
    lines.append('')
    return '\n'.join(lines)


# --- CLI ---------------------------------------------------------------------------------------

def evidence_handler(
    gate_jsons: Sequence[str], config_path: Optional[str], runs_csv: Optional[str],
    out_root: str, results_dir: str,
) -> int:
    """CLI handler for `qrc-thresher evidence`. Returns exit code."""
    try:
        target = export_evidence(
            [Path(p) for p in gate_jsons], out_root=Path(out_root),
            config_path=Path(config_path) if config_path else None,
            runs_csv=Path(runs_csv) if runs_csv else None, results_dir=Path(results_dir),
        )
    except EvidenceError as exc:
        print(f'evidence: refused; {exc}')
        return 1
    entries = verify_manifest(target)
    print(f'evidence written to {target.as_posix()} ({len(entries)} files + {MANIFEST})')
    for name, digest in entries.items():
        print(f'  {digest}  {name}')
    return 0


__all__ = [
    'CP4C_SEQUENCE',
    'DROPPED_COLUMNS',
    'EvidenceError',
    'FORBIDDEN_KEYS',
    'MANIFEST',
    'MEMBERS',
    'ROW_COLUMNS',
    'evidence_handler',
    'export_evidence',
    'kind_of',
    'scan_folder',
    'scan_text',
    'sha256_bytes',
    'sha256_file',
    'verify_manifest',
    'write_manifest',
]
