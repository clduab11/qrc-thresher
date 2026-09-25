"""`qrc-thresher scorecard`: the plain-language scorecard generated from docs/evidence/*/ (CP5 item
E.5; docs/DECISIONS.md D019, ruling P4, D018 for the labels, CP5a rulings 7 and 14).

On synthetic evidence folders in tmp_path: the columns are pinned; every key number is in its
cell; INSUFFICIENT_EVIDENCE is shown as such and a gate without evidence shows "missing", neither
as FAIL; "baseline better" carries its two-sided p; G0 is "not published"; the notes and the B.3
footnote sit in the notes section; G2 and G2.5 files are never confused; G0.7 rows are keyed by
(model, model_details.experiment_name) with the newest stamp winning; a file that is not listed in
MANIFEST.sha256 is ignored and one whose hash does not match is refused; the unstamped legacy form
is read; each kind's commit comes from its own key; the same input gives the same bytes and no
generation timestamp.

The scorecard module is imported inside each test so that before it exists every test fails on
its own instead of the whole file failing at collection.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from pathlib import Path

import pytest

COMMIT_FAMILY = '0123456789abcdef0123456789abcdef01234567'
COMMIT_G07_TOP = 'aaaa111122223333aaaa111122223333aaaa1111'
COMMIT_G07_ENV = 'bbbb222233334444bbbb222233334444bbbb2222'
COMMIT_LEGACY = 'cccc333344445555cccc333344445555cccc3333'
CONFIG_HASH = hashlib.sha256(b'scorecard config').hexdigest()
TUNING_HASH = hashlib.sha256(b'older g07 tuning config').hexdigest()
SWEEP = '20260923T120000000000Z'
STAMP = '20260923T130000000000Z'
MEMBERS = ['G1', 'G2', 'G2.5', 'G3', 'G4']
COLUMNS = ['gate', 'plain-language claim', 'verdict', 'key numbers', 'measurement label',
           'commit', 'config hash', 'evidence file']
EXACT, CLASSICAL = 'exact (oracle upper bound)', 'classical, no measurement cost'
# Distinctive numbers that collide with nothing else in the record.
NUMBERS = {
    'mean_a': 0.9137, 'mean_b': 0.7261, 'mean_diff': 0.1876, 'ci_low': 0.1123, 'ci_high': 0.2618,
    'd_z': 1.3719, 'p_one_sided': 0.00123, 'adjusted_p': 0.00615, 'p_two_sided': 0.00246,
}
WORSE = {  # the "baseline better" member
    'mean_a': 1.7373, 'mean_b': 2.9073, 'mean_diff': -1.17, 'ci_low': -1.3801, 'ci_high': -0.8877,
    'd_z': -2.5571, 'p_one_sided': 0.99987, 'adjusted_p': 1.0, 'p_two_sided': 0.000321,
}


def _scorecard():
    return importlib.import_module('qrc_thresher.proof.scorecard')


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _g(x: float) -> str:
    return f'{x:.4g}'


def _comparison(status='OK', numbers=NUMBERS):
    if status != 'OK':
        return {'status': status, 'reason': 'an arm has no rows (arm A: 12, arm B: 0)',
                'n_pairs': 0, 'pairs': [], 'values_a': [], 'values_b': [], 'mean_a': None,
                'mean_b': None, 'mean_diff': None, 'ci_low': None, 'ci_high': None,
                'd_z': None, 'p_one_sided': None, 'p_two_sided': None, 'run_ids': []}
    return {'status': 'OK', 'reason': None, 'n_pairs': 12,
            'pairs': [[42 + i, 137 + i] for i in range(12)],
            'run_ids': [f'r{i}' for i in range(24)], **numbers}


def _member(name, result, *, task, baseline, floor=None, baseline_better=False):
    ok = result in ('PASS', 'FAIL')
    numbers = WORSE if baseline_better else NUMBERS
    classical = baseline.startswith(('esn', 'rks'))
    return {
        'gate': name, 'design': task, 'task': task, 'metric': 'accuracy', 'direction': 'greater',
        'baseline': baseline, 'sweep_id': SWEEP,
        'comparison': _comparison('OK' if ok else 'INSUFFICIENT_EVIDENCE', numbers),
        'n_pairs': 12 if ok else 0, 'raw_p': numbers['p_one_sided'] if ok else 1.0,
        'adjusted_p': numbers['adjusted_p'] if ok else 1.0, 'floor': floor,
        'baseline_better': baseline_better,
        'p_two_sided': numbers['p_two_sided'] if ok else None, 'reported': {},
        'default': _comparison('OK', {**NUMBERS, 'mean_a': 0.5432, 'mean_b': 0.4321}),
        'default_w1': {'design': 'default_w1', 'task_name': task, 'n_rows': 12, 'mean': 0.4444,
                       'std': 0.0987, 'status': 'OK', 'reason': None, 'n_rows_in': 12},
        'message': '' if ok else 'esn_narma: an arm has no rows', 'result': result,
        'measurement_labels': {'a': EXACT, 'b': CLASSICAL if classical else EXACT},
    }


def _family(results: dict, *, commit=COMMIT_FAMILY, stamp=STAMP) -> tuple:
    baselines = {'G1': 'ablation:no_entangle', 'G2': 'rks_parity', 'G2.5': 'ablation:haar',
                 'G3': 'esn', 'G4': 'esn_narma'}
    tasks = {'G1': 'parity', 'G2': 'parity', 'G2.5': 'stm', 'G3': 'stm', 'G4': 'narma'}
    members = {}
    for m in MEMBERS:
        floor = None
        if m == 'G2':
            floor = {'metric': 'accuracy', 'rule': 'mean_greater_than', 'value': 0.7,
                     'observed': NUMBERS['mean_a'], 'passed': True}
        members[m] = _member(m, results[m], task=tasks[m], baseline=baselines[m], floor=floor,
                             baseline_better=(m == 'G3' and results[m] == 'FAIL'))
    record = {
        'family': 'COMPARATIVE', 'version': 1, 'protocol_path': 'configs/gates/COMPARATIVE.v1.yaml',
        'protocol_sha256': 'a' * 64, 'alpha': 0.05, 'family_size': 5,
        'holm_family': [f'{m}.paired_t_one_sided' for m in MEMBERS], 'config_hash': CONFIG_HASH,
        'sweep_id': SWEEP, 'tuning_record_sha': {'stm': 'b' * 64, 'parity': 'c' * 64,
                                                 'narma': 'd' * 64},
        'git_commit': commit, 'measurement_model': 'exact', 'measurement_label': EXACT,
        'readout': 'z_only', 'n_rows': 336, 'members': members,
        'timestamp_utc': '2026-09-23T13:00:00+00:00',
    }
    return f'COMPARATIVE.v1.{stamp}.json', record


def _g07(model, experiment, result, *, top_level=True):
    data = {'gate': 'G0.7', 'result': result, 'model': model, 'protocol_version': 1,
            'protocol_sha256': 'e' * 64, 'n_seeds': 3, 'message': f'{result} on 3 seeds',
            'measurement_model': 'exact', 'measurement_label': EXACT,
            'clauses': {'stm': {'result': result}, 'parity': {'result': result}},
            'model_details': {'experiment_name': experiment, 'reservoir': 'windowed',
                              'kind': 'classical' if model.startswith('esn') else 'quantum',
                              'tuning_config_hash': TUNING_HASH},
            'environment': {'git_commit_hash': COMMIT_G07_ENV, 'python': '3.13.0'}}
    if top_level:  # written after D018 (CP5a ruling 7)
        data['config_hash'] = CONFIG_HASH
        data['git_commit_hash'] = COMMIT_G07_TOP
    return data


def _legacy(name, result):
    return {'gate': name, 'result': result, 'evidence': {'n_cases': 34, 'all_match': True},
            'run_ids': [], 'timestamp_utc': '2026-09-23T13:00:00+00:00',
            'git_commit_hash': COMMIT_LEGACY, 'config_hash': None,
            'config_hash_reason': f'{name} evaluates fixed cases, not a config',
            'measurement_model': 'exact', 'measurement_label': EXACT}


def _write_manifest(folder: Path, skip=()) -> None:
    lines = []
    for path in sorted(p for p in folder.rglob('*') if p.is_file()):
        rel = path.relative_to(folder).as_posix()
        if rel == 'MANIFEST.sha256' or rel in skip:
            continue
        lines.append(f'{_sha256(path)}  {rel}')
    (folder / 'MANIFEST.sha256').write_bytes(('\n'.join(lines) + '\n').encode('utf-8'))


def _folder(tmp_path, name, family=None, g07s=(), legacy=(), stamp=STAMP, skip=()) -> Path:
    """g07s: (model, experiment, result, stamp, top_level) tuples; legacy: (gate, result,
    stamp or None) tuples."""
    folder = tmp_path / 'docs' / 'evidence' / name
    folder.mkdir(parents=True)
    if family is not None:
        fname, record = family
        (folder / fname).write_text(json.dumps(record, indent=2), encoding='utf-8')
        family_sha = _sha256(folder / fname)
        for m in MEMBERS:
            view = {**record['members'][m], 'config_hash': record['config_hash'],
                    'sweep_id': SWEEP, 'git_commit': record['git_commit'],
                    'measurement_model': 'exact', 'measurement_label': EXACT,
                    'protocol_sha256': record['protocol_sha256'],
                    'family_json': f'results/gates/{fname}',
                    'family_sha256': family_sha, 'timestamp_utc': record['timestamp_utc']}
            suffix = '.1' if m == 'G2.5' else ''
            (folder / f'{m}.{stamp}{suffix}.json').write_text(json.dumps(view, indent=2),
                                                             encoding='utf-8')
    for model, experiment, result, g_stamp, top_level in g07s:
        (folder / f'G0.7.{model}.{g_stamp}.json').write_text(
            json.dumps(_g07(model, experiment, result, top_level=top_level), indent=2),
            encoding='utf-8')
    for gate, result, g_stamp in legacy:
        fname = f'{gate}.json' if g_stamp is None else f'{gate}.{g_stamp}.json'
        (folder / fname).write_text(json.dumps(_legacy(gate, result), indent=2), encoding='utf-8')
    (folder / 'README.md').write_text(f'# Evidence {name}\n', encoding='utf-8')
    _write_manifest(folder, skip=skip)
    return folder


def _table_rows(text: str) -> dict:
    rows = {}
    for line in text.splitlines():
        if line.startswith('| ') and not line.startswith('| ---') and not line.startswith('| gate'):
            cells = [c.strip() for c in line.strip('|').split(' | ')]
            rows.setdefault(cells[0], []).append(cells)
    return rows


def _notes(text: str) -> str:
    assert '## Notes' in text
    return text.split('## Notes', 1)[1]


RESULTS = {'G1': 'FAIL', 'G2': 'PASS', 'G2.5': 'FAIL', 'G3': 'FAIL', 'G4': 'INSUFFICIENT_EVIDENCE'}


class TestScorecard:
    def test_the_columns_are_pinned_and_every_gate_has_a_row(self, tmp_path) -> None:
        sc = _scorecard()
        folder = _folder(tmp_path, 'abc1234', family=_family(RESULTS),
                         g07s=[('tuned_qrc', 'comparative', 'FAIL', STAMP, True)],
                         legacy=[('G0.5', 'PASS', STAMP)])
        text = sc.build_scorecard([folder], tmp_path / 'docs')
        assert list(sc.SCORECARD_COLUMNS) == COLUMNS
        header = next(line for line in text.splitlines() if line.startswith('| gate'))
        assert header == '| ' + ' | '.join(COLUMNS) + ' |'
        rows = _table_rows(text)
        for gate in ('G0', 'G0.5', 'G0.7', 'G1', 'G2', 'G2.5', 'G3', 'G4', 'G5', 'G6', 'G7'):
            assert gate in rows, gate
        assert 'not published' in rows['G0'][0][2] and 'health' in rows['G0'][0][2]
        assert rows['G2'][0][2] == 'PASS' and rows['G1'][0][2] == 'FAIL'
        assert rows['G4'][0][2] == 'INSUFFICIENT_EVIDENCE'
        for gate in ('G5', 'G6', 'G7'):
            assert rows[gate][0][2] == 'missing' and 'FAIL' not in ' '.join(rows[gate][0]), gate

    def test_the_key_numbers_are_in_their_cell(self, tmp_path) -> None:
        sc = _scorecard()
        folder = _folder(tmp_path, 'abc1234', family=_family(RESULTS))
        rows = _table_rows(sc.build_scorecard([folder], tmp_path / 'docs'))
        g2 = rows['G2'][0]
        numbers, claim, verdict = g2[3], g2[1], g2[2]
        assert verdict == 'PASS'
        assert 'n_pairs = 12' in numbers
        for key in ('mean_a', 'mean_b', 'mean_diff', 'ci_low', 'ci_high', 'd_z', 'p_one_sided',
                    'adjusted_p'):
            assert _g(NUMBERS[key]) in numbers, key
        assert 'Holm' in numbers and 'd_z' in numbers
        assert 'floor' in numbers.lower()
        assert re.search(r'(?<![\d.])0\.7(?!\d)', numbers)  # the floor's own value, standalone
        assert 'baseline better' not in numbers
        assert g2[4] == f'a: {EXACT}; b: {CLASSICAL}'  # per arm, since the arms differ
        assert g2[5] == COMMIT_FAMILY[:7] and g2[6] == CONFIG_HASH[:8]
        assert claim and 'RKS' in claim  # D014's plain-language claim
        # The default table beside the member, marked reported-only.
        default = [r for r in rows['G2'] if 'reported' in ' '.join(r).lower()]
        assert default and _g(0.5432) in ' '.join(default[0]) and _g(0.4321) in ' '.join(default[0])
        # G1 and G2.5: the arms share the label, so one label is shown.
        assert rows['G1'][0][4] == EXACT and rows['G2.5'][0][4] == EXACT

    def test_baseline_better_carries_its_two_sided_p(self, tmp_path) -> None:
        sc = _scorecard()
        folder = _folder(tmp_path, 'abc1234', family=_family(RESULTS))
        rows = _table_rows(sc.build_scorecard([folder], tmp_path / 'docs'))
        g3 = rows['G3'][0]
        assert g3[2] == 'FAIL'
        assert 'baseline better' in g3[3] and _g(WORSE['p_two_sided']) in g3[3]
        assert _g(WORSE['mean_a']) in g3[3] and _g(WORSE['mean_b']) in g3[3]
        assert _g(WORSE['d_z']) in g3[3]

    def test_the_notes_section(self, tmp_path) -> None:
        sc = _scorecard()
        folder = _folder(tmp_path, 'abc1234', family=_family(RESULTS),
                         g07s=[('esn_linear', 'alpha_lite_phase1', 'FAIL', STAMP, True)])
        text = sc.build_scorecard([folder], tmp_path / 'docs')
        notes = _notes(text)
        assert 'baseline comparison' in notes and 'D014' in notes  # G2 is not an entanglement claim
        assert 'parity' in notes and 'G1' in notes and 'G2' in notes and 'D016' in notes
        assert 'oracle upper bound' in notes and 'D004' in notes
        assert 'finite-shot' in notes and 'pending' in notes and 'D017' in notes  # ruling 1
        assert 'before D018' in notes and 'exact' in notes  # the B.3 footnote
        assert 'G0.7 v1' in notes and 'by protocol' in notes  # ruling 14c
        rows = _table_rows(text)
        esn = [r for r in rows['G0.7'] if 'esn_linear' in ' '.join(r)]
        assert esn and esn[0][4] == CLASSICAL  # by kind, footnoted in the notes

    def test_g2_and_g25_are_never_confused(self, tmp_path) -> None:
        sc = _scorecard()
        results = {'G1': 'PASS', 'G2': 'PASS', 'G2.5': 'FAIL', 'G3': 'PASS', 'G4': 'PASS'}
        folder = _folder(tmp_path, 'abc1234', family=_family(results))
        rows = _table_rows(sc.build_scorecard([folder], tmp_path / 'docs'))
        assert rows['G2'][0][2] == 'PASS' and rows['G2.5'][0][2] == 'FAIL'
        assert f'abc1234/G2.{STAMP}.json' in rows['G2'][0][7]
        assert _sha256(folder / f'G2.{STAMP}.json')[:12] in rows['G2'][0][7]
        assert f'abc1234/G2.5.{STAMP}.1.json' in rows['G2.5'][0][7]  # found by content
        assert _sha256(folder / f'G2.5.{STAMP}.1.json')[:12] in rows['G2.5'][0][7]

    def test_g07_is_keyed_by_model_and_experiment_and_the_newest_stamp_wins(self, tmp_path):
        sc = _scorecard()
        older, newer = '20260923T100000000000Z', '20260923T110000000000Z'
        folder = _folder(tmp_path, 'abc1234', g07s=[
            ('pennylane_qrc', 'alpha_lite_phase1', 'FAIL', older, True),
            ('pennylane_qrc', 'alpha_lite_phase1', 'PASS', newer, True),
            ('pennylane_qrc', 'windowed_w2', 'FAIL', older, True),
            ('esn_linear', 'alpha_lite_phase1', 'FAIL', older, True),
        ])
        rows = _table_rows(sc.build_scorecard([folder], tmp_path / 'docs'))['G0.7']
        assert len(rows) == 3  # (model, experiment_name) keys; the older duplicate is dropped
        by_text = {' '.join(r) for r in rows}
        assert any('pennylane_qrc' in t and 'alpha_lite_phase1' in t and ' PASS ' in f' {t} '
                   and newer in t for t in by_text)
        assert any('pennylane_qrc' in t and 'windowed_w2' in t and 'FAIL' in t for t in by_text)
        assert any('esn_linear' in t and CLASSICAL in t for t in by_text)
        assert not any(older in t and 'alpha_lite_phase1' in t and 'pennylane_qrc' in t
                       for t in by_text)

    def test_each_kind_takes_its_commit_and_config_hash_from_its_own_keys(self, tmp_path):
        sc = _scorecard()
        folder = _folder(tmp_path, 'abc1234', family=_family(RESULTS), g07s=[
            ('tuned_qrc', 'comparative', 'FAIL', STAMP, True),        # top-level keys (ruling 7)
            ('pennylane_qrc', 'windowed_w2', 'FAIL', STAMP, False),   # older file: fallbacks
        ], legacy=[('G0.5', 'PASS', STAMP), ('G6', 'FAIL', None)])
        rows = _table_rows(sc.build_scorecard([folder], tmp_path / 'docs'))
        assert rows['G1'][0][5] == COMMIT_FAMILY[:7]  # the family's git_commit
        tuned = next(r for r in rows['G0.7'] if 'tuned_qrc' in ' '.join(r))
        assert tuned[5] == COMMIT_G07_TOP[:7] and tuned[6] == CONFIG_HASH[:8]
        older = next(r for r in rows['G0.7'] if 'windowed_w2' in ' '.join(r))
        assert older[5] == COMMIT_G07_ENV[:7]  # environment.git_commit_hash
        assert TUNING_HASH[:8] in older[6] and '(tuning config)' in older[6]
        assert rows['G0.5'][0][5] == COMMIT_LEGACY[:7]  # legacy git_commit_hash
        assert rows['G6'][0][2] == 'FAIL' and 'abc1234/G6.json' in rows['G6'][0][7]  # unstamped

    def test_a_file_not_in_the_manifest_is_ignored(self, tmp_path) -> None:
        sc = _scorecard()
        results = {**RESULTS, 'G3': 'FAIL'}
        folder = _folder(tmp_path, 'abc1234', family=_family(results))
        newer = '20260923T140000000000Z'
        listed = json.loads((folder / f'G3.{STAMP}.json').read_text(encoding='utf-8'))
        stray = {**listed, 'result': 'PASS'}
        (folder / f'G3.{newer}.json').write_text(json.dumps(stray, indent=2), encoding='utf-8')
        text = sc.build_scorecard([folder], tmp_path / 'docs')  # the manifest is unchanged
        rows = _table_rows(text)
        assert rows['G3'][0][2] == 'FAIL' and f'G3.{STAMP}.json' in rows['G3'][0][7]
        assert f'G3.{newer}.json' not in text

    def test_a_hash_that_does_not_match_the_manifest_is_refused(self, tmp_path) -> None:
        sc = _scorecard()
        folder = _folder(tmp_path, 'abc1234', family=_family(RESULTS))
        view = folder / f'G3.{STAMP}.json'
        data = json.loads(view.read_text(encoding='utf-8'))
        data['result'] = 'PASS '  # tampered after the manifest was written
        view.write_text(json.dumps(data, indent=2), encoding='utf-8')
        with pytest.raises(sc.EvidenceError, match=re.escape(view.name)):
            sc.build_scorecard([folder], tmp_path / 'docs')

    def test_the_same_input_gives_the_same_bytes_and_no_timestamp(self, tmp_path) -> None:
        sc = _scorecard()
        folder = _folder(tmp_path, 'abc1234', family=_family(RESULTS),
                         legacy=[('G0.5', 'PASS', STAMP)])
        first = sc.build_scorecard([folder], tmp_path / 'docs')
        second = sc.build_scorecard([folder], tmp_path / 'docs')
        assert first == second
        assert not re.search(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}', first)  # no generation stamp
        assert 'generated on' not in first.lower() and 'generated at' not in first.lower()

    def test_two_folders_and_the_missing_family(self, tmp_path) -> None:
        sc = _scorecard()
        a = _folder(tmp_path, 'aaa1111', legacy=[('G0.5', 'PASS', STAMP)])
        b = _folder(tmp_path, 'bbb2222', g07s=[('tuned_qrc', 'comparative', 'FAIL', STAMP, True)])
        rows = _table_rows(sc.build_scorecard([a, b], tmp_path / 'docs'))
        assert rows['G0.5'][0][2] == 'PASS' and rows['G0.7'][0][2] == 'FAIL'
        for m in MEMBERS:
            assert rows[m][0][2] == 'missing'


class TestCli:
    def test_the_scorecard_command(self, tmp_path, monkeypatch) -> None:
        from click.testing import CliRunner

        from qrc_thresher.cli import cli

        folder = _folder(tmp_path, 'abc1234', family=_family(RESULTS))
        monkeypatch.chdir(tmp_path)
        out = tmp_path / 'docs' / 'scorecard.md'
        result = CliRunner().invoke(cli, ['scorecard', '--evidence', str(folder), '--out',
                                          str(out)])
        assert result.exit_code == 0, result.output
        text = out.read_text(encoding='utf-8')
        assert '| ' + ' | '.join(COLUMNS) + ' |' in text
        assert 'abc1234' in text
