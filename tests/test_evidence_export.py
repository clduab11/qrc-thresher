"""`qrc-thresher evidence`: byte-exact export of gate records to docs/evidence/<short-commit>/
(CP5 item E.1; docs/DECISIONS.md D019, ruling P4).

On a synthetic results/ tree in tmp_path: copies are byte-exact and every hash check runs; the
member views are found by content (a .1 suffix included), never by a filename glob; the config
copy is byte-exact and its canonical hash checked; runs.sanitised.csv holds exactly the
allowlisted columns with every cell the identical text of its source (LF endings);
MANIFEST.sha256 covers every file except itself; README.md states the path mappings and names the
dropped columns without tripping the scan; the export refuses a mismatched family_sha256, two
views for one member, a config whose hash differs, an 'unknown' or '-dirty' commit, mixed
commits, rows of the config outside the family's sweeps, an existing target and a planted local
path or forbidden key; a refused export leaves no target folder. The staging folder and the
target both live in tmp_path.

The evidence module is imported inside each test so that before it exists every test fails on
its own instead of the whole file failing at collection.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import re
import shutil
import tempfile
from pathlib import Path

import pytest
import test_comparative_family as F
import yaml
from synthetic_rows import COLUMNS, SWEEP_ID

COMMIT = '0123456789abcdef0123456789abcdef01234567'
SHORT = COMMIT[:7]
MISREAD_FLOAT = '1.8106150706836674'  # pandas' default parser reads this one ulp off
MINIMUM_COLUMNS = {
    'run_id', 'success', 'git_commit_hash', 'config_hash', 'sweep_id', 'tuning_record_sha',
    'circuit_hash', 'task_seed', 'reservoir_seed', 'task_name', 'design', 'primary_metric_name',
    'primary_metric_value', 'measurement_model', 'n_configs', 'n_validation_evals',
    'secondary_metrics',
}
FORBIDDEN_COLUMNS = {'cli_command', 'git_branch', 'platform', 'artifact_paths',
                     'package_versions', 'config_path', 'timestamp_utc', 'failure_reason'}
ALLOWLIST = MINIMUM_COLUMNS | {'python_version', 'backend_device', 'device', 'precision',
                               'entanglement_metric', 'runtime_per_stage_seconds'}  # ruling 10


def _evidence():
    return importlib.import_module('qrc_thresher.proof.evidence')


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows, columns=COLUMNS) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: ('' if r.get(c) is None else str(r.get(c))) for c in columns})


def _read_csv(path: Path) -> list:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _build(root: Path, commit: str = COMMIT) -> dict:
    """A synthetic results/ tree whose family, views, G0.7, records, config and rows agree.

    ``commit`` is written to the family record, to G0.7's environment (before its sha is taken)
    and to every row, so a tree with 'unknown' or '<commit>-dirty' can be built for the refusals.
    """
    from qrc_thresher.gates import comparative
    from qrc_thresher.proof.run_manifest import _config_hash
    from qrc_thresher.tuning import record_sha256

    configs, gates = root / 'configs', root / 'results' / 'gates'
    configs.mkdir(parents=True)
    gates.mkdir(parents=True)
    raw = yaml.safe_load((F.REPO_ROOT / 'configs' / 'alpha_lite.yaml').read_text(encoding='utf-8'))
    raw['experiment_name'] = 'tiny_evidence'
    config_path = configs / 'tiny.yaml'
    config_path.write_text('# see https://example.org/qrc for the design notes\n'
                           + yaml.safe_dump(raw), encoding='utf-8')
    config_hash = _config_hash(config_path)

    records, shas = {}, {}
    for task in ('stm', 'parity', 'narma'):
        record = {'config_hash': config_hash, 'config_path': 'configs/tiny.yaml',
                  'sweep_id': SWEEP_ID, 'task': task, 'qrc': {}, 'esn': {}, 'rks': {}}
        record['record_sha256'] = record_sha256(record)
        records[task] = record
        shas[task] = record['record_sha256']
        path = root / 'results' / 'tuning' / config_hash / f'{task}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2), encoding='utf-8')

    g07_name = 'G0.7.tuned_qrc.20260923T120000000000Z.json'
    g07 = {'gate': 'G0.7', 'result': 'PASS', 'model': 'tuned_qrc',
           'model_details': {'tuning_config_hash': config_hash, 'sweep_id': SWEEP_ID,
                             'experiment_name': 'tiny_evidence'},
           'environment': {'git_commit_hash': commit, 'python': '3.13'},
           'figure': g07_name.replace('.json', '.forgetting_curve.png')}
    (gates / g07_name).write_text(json.dumps(g07, indent=2), encoding='utf-8')
    (gates / g07['figure']).write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 32)

    runs = F._sweep()
    runs['config_hash'] = config_hash
    runs['git_commit_hash'] = commit
    g07_tuned = {'result': 'PASS', 'json': f'results/gates/{g07_name}',
                 'sha256': _sha256(gates / g07_name),
                 'model_details': g07['model_details']}
    result = F._evaluate(runs, config_hash=config_hash, g07_tuned=g07_tuned,
                         tuning_record_sha=shas)
    result['git_commit'] = commit
    # As in run 4: written from the repository root with a relative out_dir, so that every view
    # records family_json 'results/gates/...' (TF1). An exporter built to spec refuses a tree
    # whose views record an absolute path.
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(root)
        paths = comparative.write_family_report(result, Path('results') / 'gates')
    paths = {k: (root / p) if not p.is_absolute() else p for k, p in paths.items()}
    for member in ('G1', 'G2', 'G2.5', 'G3', 'G4'):
        view = json.loads(paths[member].read_text(encoding='utf-8'))
        assert view['family_json'].startswith('results/gates/'), view['family_json']
    # One member view carries the .1 suffix _fresh adds when a stamp collides.
    g2_path = paths['G2']
    suffixed = g2_path.with_name(g2_path.name[:-len('.json')] + '.1.json')
    g2_path.rename(suffixed)
    paths['G2'] = suffixed

    rows = runs.to_dict('records')
    rows[0]['primary_metric_value'] = MISREAD_FLOAT
    for r in rows:
        r['cli_command'] = 'qrc-thresher run stm'
        r['config_path'] = str(config_path)
    _write_csv(root / 'results' / 'runs.csv', rows)
    return {'root': root, 'config_path': config_path, 'config_hash': config_hash,
            'paths': paths, 'g07': gates / g07_name, 'figure': gates / g07['figure'],
            'records': records, 'rows': rows}


@pytest.fixture(scope='module')
def base(tmp_path_factory):
    return _build(tmp_path_factory.mktemp('evidence_base'))


@pytest.fixture
def tree(base, tmp_path, monkeypatch):
    """A private copy of the synthetic tree, working directory set to it."""
    root = tmp_path / 'repo'
    shutil.copytree(base['root'], root)
    fixed = dict(base)
    fixed['root'] = root
    fixed['config_path'] = root / 'configs' / 'tiny.yaml'
    fixed['paths'] = {k: root / 'results' / 'gates' / p.name for k, p in base['paths'].items()}
    fixed['g07'] = root / 'results' / 'gates' / base['g07'].name
    fixed['figure'] = root / 'results' / 'gates' / base['figure'].name
    fixed['out_root'] = tmp_path / 'docs' / 'evidence'
    staging = tmp_path / 'staging'
    staging.mkdir()
    monkeypatch.setattr(tempfile, 'tempdir', str(staging))  # TF5: the export stages here
    fixed['staging'] = staging
    return fixed


def _no_staging_left(tree) -> None:
    assert list(tree['staging'].iterdir()) == [], 'a staging folder was left behind'


def _export(tree, gate_jsons=None, **kwargs):
    ev = _evidence()
    gate_jsons = gate_jsons if gate_jsons is not None else [tree['paths']['family'], tree['g07']]
    options = dict(out_root=tree['out_root'], config_path=tree['config_path'],
                   runs_csv=tree['root'] / 'results' / 'runs.csv',
                   results_dir=tree['root'] / 'results')
    options.update(kwargs)
    return ev.export_evidence(gate_jsons, **options)


class TestExport:
    def test_copies_are_byte_exact_and_the_folder_is_complete(self, tree) -> None:
        target = _export(tree)
        assert target == tree['out_root'] / SHORT and target.is_dir()
        _no_staging_left(tree)
        expected = {p.name for p in tree['paths'].values()} | {tree['g07'].name,
                                                                tree['figure'].name}
        expected |= {'tuning/stm.json', 'tuning/parity.json', 'tuning/narma.json',
                     'configs/tiny.yaml', 'runs.sanitised.csv', 'MANIFEST.sha256', 'README.md'}
        files = {p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()}
        assert files == expected
        for key, path in tree['paths'].items():
            assert (target / path.name).read_bytes() == path.read_bytes(), key
        assert (target / tree['g07'].name).read_bytes() == tree['g07'].read_bytes()
        assert (target / tree['figure'].name).read_bytes() == tree['figure'].read_bytes()
        assert (target / 'configs' / 'tiny.yaml').read_bytes() == tree['config_path'].read_bytes()
        for task in ('stm', 'parity', 'narma'):
            source = tree['root'] / 'results' / 'tuning' / tree['config_hash'] / f'{task}.json'
            assert (target / 'tuning' / f'{task}.json').read_bytes() == source.read_bytes()
        # The .1-suffixed G2 view was found by content and copied under its own name.
        assert (target / tree['paths']['G2'].name).exists()
        assert tree['paths']['G2'].name.endswith('.1.json')

    def test_the_manifest_verifies_and_covers_every_file_except_itself(self, tree) -> None:
        target = _export(tree)
        raw = (target / 'MANIFEST.sha256').read_bytes()
        assert b'\r' not in raw
        entries = {}
        for line in raw.decode('utf-8').splitlines():
            digest, sep, name = line.partition('  ')
            assert sep and len(digest) == 64 and '\\' not in name and not name.startswith('/')
            entries[name] = digest
        files = {p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()}
        assert set(entries) == files - {'MANIFEST.sha256'}
        assert 'README.md' in entries
        for name, digest in entries.items():
            assert _sha256(target / name) == digest, name
        assert _evidence().verify_manifest(target) == entries

    def test_the_sanitised_rows(self, tree) -> None:
        target = _export(tree)
        raw = (target / 'runs.sanitised.csv').read_bytes()
        assert b'\r' not in raw  # LF line endings
        rows = _read_csv(target / 'runs.sanitised.csv')
        columns = set(rows[0])
        assert MINIMUM_COLUMNS <= columns
        assert not columns & FORBIDDEN_COLUMNS
        assert columns == ALLOWLIST  # exactly the allowlisted columns (E.1, ruling 10)
        assert set(_evidence().ROW_COLUMNS) == columns
        source = _read_csv(tree['root'] / 'results' / 'runs.csv')
        assert len(rows) == len(source)
        by_id = {r['run_id']: r for r in source}
        for r in rows:
            src = by_id[r['run_id']]
            for column, cell in r.items():
                assert cell == src[column], (r['run_id'], column)
        assert any(r['primary_metric_value'] == MISREAD_FLOAT for r in rows)

    def test_the_readme_states_the_mappings_and_names_the_dropped_columns(self, tree) -> None:
        target = _export(tree)
        text = (target / 'README.md').read_text(encoding='utf-8')
        lines = text.splitlines()
        assert SHORT in text and SWEEP_ID in text
        # Each data file's sha256 is on the line that names it (README.md is not a data file).
        manifest = _evidence().verify_manifest(target)
        for name, digest in manifest.items():
            if name == 'README.md':
                continue
            assert any(name in line and digest in line for line in lines), name
        # The README does not hash MANIFEST.sha256, so neither file hashes the other.
        assert not re.search(r'MANIFEST\.sha256[^\n]*[0-9a-f]{64}', text)
        assert not re.search(r'[0-9a-f]{64}[^\n]*MANIFEST\.sha256', text)
        # The three path mappings, each as a mapping (recorded path -> path in this folder).
        family = tree['paths']['family'].name
        for recorded, here in (
            (f'results/gates/{family}', family),
            (f"results/tuning/{tree['config_hash']}/stm.json", 'tuning/stm.json'),
            ('configs/tiny.yaml', 'configs/tiny.yaml'),
        ):
            pattern = re.compile(r'\|\s*`?' + re.escape(recorded) + r'`?\s*\|\s*`?'
                                 + re.escape(here) + r'`?\s*\|')
            assert any(pattern.search(line) for line in lines), (recorded, here)
        for column in ('cli_command', 'git_branch', 'platform', 'artifact_paths',
                       'package_versions', 'config_path'):
            assert column in text, column
        assert 'qrc-thresher tune --config' in text  # the CP4c sequence
        assert _evidence().scan_text(text) == []

    def test_an_https_url_does_not_trip_the_drive_letter_check(self, tree) -> None:
        ev = _evidence()
        assert ev.scan_text('see https://example.org/x and http://a.b/c') == []
        assert ev.scan_text('data at C:\\work\\x') != []
        assert ev.scan_text('data at D:/work/x') != []
        assert ev.scan_text('/Users/someone/x') != [] and ev.scan_text('/home/someone/x') != []
        assert ev.scan_text('"cli_command": "x"') != []
        assert ev.scan_text('cli_command: x') != []
        assert ev.scan_text('run_id,cli_command,success') != []
        assert ev.scan_text('the cli_command column was dropped') == []
        assert ev.scan_text('"config_path": "/abs/configs/x.yaml"') != []
        assert ev.scan_text('"config_path": "configs/x.yaml"') == []
        target = _export(tree)  # the config comment holds an https URL
        assert (target / 'configs' / 'tiny.yaml').exists()


class TestRefusals:
    def _refused(self, tree, match, **kwargs):
        ev = _evidence()
        with pytest.raises(ev.EvidenceError, match=match):
            _export(tree, **kwargs)
        assert not (tree['out_root'] / SHORT).exists()
        assert not tree['out_root'].exists() or list(tree['out_root'].iterdir()) == []
        _no_staging_left(tree)

    def test_a_mismatched_family_sha256(self, tree) -> None:
        view = tree['paths']['G3']
        data = json.loads(view.read_text(encoding='utf-8'))
        data['family_sha256'] = 'f' * 64
        view.write_text(json.dumps(data, indent=2), encoding='utf-8')
        self._refused(tree, 'G3')

    def test_two_views_for_one_member(self, tree) -> None:
        view = tree['paths']['G3']
        shutil.copyfile(view, view.with_name(view.name[:-5] + '.1.json'))
        self._refused(tree, 'G3')

    def test_a_config_whose_hash_differs(self, tree) -> None:
        other = tree['root'] / 'configs' / 'other.yaml'
        raw = yaml.safe_load(tree['config_path'].read_text(encoding='utf-8'))
        raw['experiment_name'] = 'something_else'
        other.write_text(yaml.safe_dump(raw), encoding='utf-8')
        self._refused(tree, 'config_hash', config_path=other)

    @pytest.mark.parametrize('commit', ['unknown', COMMIT + '-dirty'])
    def test_an_unknown_or_dirty_commit(self, tmp_path, monkeypatch, commit) -> None:
        # A whole tree written under the bad commit (family, G0.7 and rows agree), so the
        # commit check is the only one that can fire.
        root = tmp_path / 'bad'
        built = _build(root, commit=commit)
        staging = tmp_path / 'staging'
        staging.mkdir()
        monkeypatch.setattr(tempfile, 'tempdir', str(staging))
        out_root = tmp_path / 'docs' / 'evidence'
        ev = _evidence()
        with pytest.raises(ev.EvidenceError, match='commit'):
            ev.export_evidence([built['paths']['family'], built['g07']], out_root=out_root,
                               config_path=built['config_path'],
                               runs_csv=root / 'results' / 'runs.csv',
                               results_dir=root / 'results')
        assert not out_root.exists() or list(out_root.iterdir()) == []
        assert list(staging.iterdir()) == []

    def test_mixed_commits_across_rows(self, tree) -> None:
        rows = _read_csv(tree['root'] / 'results' / 'runs.csv')
        rows[3]['git_commit_hash'] = 'fedcba9876543210fedcba9876543210fedcba98'
        _write_csv(tree['root'] / 'results' / 'runs.csv', rows, columns=list(rows[0]))
        self._refused(tree, 'commit')

    def test_rows_of_the_config_outside_the_sweeps(self, tree) -> None:
        rows = _read_csv(tree['root'] / 'results' / 'runs.csv')
        stray = dict(rows[0])
        stray.update({'run_id': 'stray-row', 'sweep_id': 'another-sweep'})
        _write_csv(tree['root'] / 'results' / 'runs.csv', rows + [stray], columns=list(rows[0]))
        self._refused(tree, 'sweep')

    def test_an_existing_target_is_refused(self, tree) -> None:
        (tree['out_root'] / SHORT).mkdir(parents=True)
        (tree['out_root'] / SHORT / 'keep.txt').write_text('mine', encoding='utf-8')
        ev = _evidence()
        with pytest.raises(ev.EvidenceError, match=SHORT):
            _export(tree)
        assert (tree['out_root'] / SHORT / 'keep.txt').read_text(encoding='utf-8') == 'mine'
        assert list((tree['out_root'] / SHORT).iterdir()) == [tree['out_root'] / SHORT / 'keep.txt']

    def test_a_planted_absolute_path(self, tree) -> None:
        text = tree['config_path'].read_text(encoding='utf-8')
        tree['config_path'].write_text('# copied from C:\\work\\configs\\tiny.yaml\n' + text,
                                       encoding='utf-8')
        self._refused(tree, 'path')

    def test_a_planted_forbidden_key(self, tree) -> None:
        # No sha binds a member view's own bytes, so only the scan can catch this.
        view = tree['paths']['G4']
        data = json.loads(view.read_text(encoding='utf-8'))
        data['cli_command'] = 'qrc-thresher gate family'
        view.write_text(json.dumps(data, indent=2), encoding='utf-8')
        self._refused(tree, 'cli_command')


class TestCli:
    def test_the_evidence_command(self, tree, monkeypatch) -> None:
        from click.testing import CliRunner

        from qrc_thresher.cli import cli

        monkeypatch.chdir(tree['root'])
        out_root = tree['out_root']
        result = CliRunner().invoke(cli, [
            'evidence', tree['paths']['family'].as_posix(), tree['g07'].as_posix(),
            '--config', 'configs/tiny.yaml', '--runs-csv', 'results/runs.csv',
            '--out-root', str(out_root),
        ])
        assert result.exit_code == 0, result.output
        assert (out_root / SHORT / 'MANIFEST.sha256').exists()
        assert SHORT in result.output
