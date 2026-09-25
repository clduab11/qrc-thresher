"""Run 4 is the registered CP4 evidence and the CP5 code re-derives it exactly (docs/DECISIONS.md
D019, rulings R1 and R4).

These tests run on the committed docs/evidence/8ed2df4/ folder: its hashes verify, re-evaluating
its rows with the current code returns every recorded leaf of the family record (verdicts,
p-values, statistics, values and n; git_commit and timestamp_utc excepted), the five verdicts are
the recorded ones, the run-3 duplicates collapse, a differing duplicate refuses only its table,
`deployment_label` splits the rows into 28 groups, and an unrelated failed row changes nothing.

The folder's own configs/comparative.yaml copy is used, never the live configs/ file. Exactness
is required under the locked environment (uv.lock): the Haar ablation's circuit_hash depends on
the numpy and scipy build (D019). The module fails, and does not skip, when the folder is missing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parent.parent
EVIDENCE = REPO_ROOT / 'docs' / 'evidence' / '8ed2df4'
FAMILY_FILE = 'COMPARATIVE.v1.20260924T224106784228Z.json'
FAMILY_SHA256 = '3da53714d0728d7d3d4276f0c1084c59fb1a3e3daf05eeb81539943c5ef85826'
G07_FILE = 'G0.7.tuned_qrc.20260924T224100291861Z.json'
G07_SHA256 = '5164fcd97bd63e36e8135b91fc9d802223310ba423fc77187f5e07fac72f0a8a'
MEMBERS = ['G1', 'G2', 'G2.5', 'G3', 'G4']
VERDICTS = {'G1': 'FAIL', 'G2': 'PASS', 'G2.5': 'FAIL', 'G3': 'FAIL', 'G4': 'FAIL'}
EXEMPT = {('git_commit',), ('timestamp_utc',)}
N_ROWS = 336
REPLAY_PAIRS = [(42 + i, 137 + i) for i in range(7)]  # run 3's duplicated pairs


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


LOCKED = {'numpy': '2.4.6', 'scipy': '1.17.1'}  # uv.lock at 8ed2df4 (D019, CP5a ruling 3)


def _require_locked_versions() -> None:
    """Fail, never skip, unless numpy and scipy are the versions the lock pins (D019)."""
    import numpy
    import scipy

    found = {'numpy': numpy.__version__, 'scipy': scipy.__version__}
    if found != LOCKED:
        pytest.fail(f'the run-4 re-derivation is exact only under the environment locked by '
                    f'uv.lock at 8ed2df4 (numpy {LOCKED["numpy"]}, scipy {LOCKED["scipy"]}; '
                    f'D019, platform note); found numpy {found["numpy"]}, scipy '
                    f'{found["scipy"]}')


def _leaves(obj, prefix=()) -> dict:
    """Every leaf of a JSON object keyed by its path; empty containers are leaves, and every
    list also contributes its length, so that growth is a change (TF8)."""
    if isinstance(obj, dict) and obj:
        out = {}
        for k, v in obj.items():
            out.update(_leaves(v, prefix + (str(k),)))
        return out
    if isinstance(obj, list) and obj:
        out = {prefix + ('len()',): len(obj)}
        for i, v in enumerate(obj):
            out.update(_leaves(v, prefix + (f'[{i}]',)))
        return out
    return {prefix: obj}


def _normalised(result: dict) -> dict:
    """The result as the JSON writer would serialise it (allow_nan=False), so that the leaf
    types compare like for like."""
    return json.loads(json.dumps(result, allow_nan=False))


def _assert_same_leaves(expected: dict, actual: dict, *, exempt=EXEMPT, allow_changed=()) -> dict:
    """Every recorded leaf comes back exactly, with the same type; no approx. Extra keys are
    allowed. Returns the changed leaves (only those under ``allow_changed`` may be non-empty)."""
    got = _leaves(_normalised(actual))
    changed = {}
    for path, value in _leaves(expected).items():
        if path in exempt:
            continue
        assert path in got, f'recorded leaf {path} missing from the re-derivation'
        new = got[path]
        same = type(new) is type(value) and new == value
        if not same:
            changed[path] = (value, new)
    unexpected = {p: c for p, c in changed.items()
                  if not any(p[:len(a)] == a for a in allow_changed)}
    assert not unexpected, f'{len(unexpected)} recorded leaf/leaves changed: {unexpected}'
    return changed


def _read_manifest(folder: Path) -> dict:
    text = (folder / 'MANIFEST.sha256').read_bytes()
    assert b'\r' not in text  # LF line endings
    entries = {}
    for line in text.decode('utf-8').splitlines():
        digest, _, name = line.partition('  ')
        assert len(digest) == 64 and name and '\\' not in name
        entries[name] = digest
    return entries


@pytest.fixture(scope='module')
def evidence():
    """The committed folder, loaded and evaluated once for the module (about 5 s)."""
    _require_locked_versions()  # CP5a ruling 3: before any evaluation
    if not EVIDENCE.is_dir():
        pytest.fail(f'{EVIDENCE.relative_to(REPO_ROOT).as_posix()} does not exist; the run-4 '
                    'evidence export (D019, P4) has not been committed')
    from qrc_thresher.config import load_config
    from qrc_thresher.gates import comparative
    from qrc_thresher.tuning import record_sha256

    family = json.loads((EVIDENCE / FAMILY_FILE).read_text(encoding='utf-8'))
    views = {}
    for path in sorted(EVIDENCE.glob('*.json')):
        if path.name in (FAMILY_FILE, G07_FILE):
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        if 'family_sha256' in data:
            views.setdefault(data['gate'], []).append((path, data))
    g07 = json.loads((EVIDENCE / G07_FILE).read_text(encoding='utf-8'))
    records = {task: json.loads((EVIDENCE / 'tuning' / f'{task}.json').read_text(encoding='utf-8'))
               for task in ('stm', 'parity', 'narma')}
    config_path = EVIDENCE / 'configs' / 'comparative.yaml'
    cfg = load_config(config_path)
    runs = comparative.read_runs_csv(EVIDENCE / 'runs.sanitised.csv')
    designs = comparative.designs_from_records(cfg, records)
    kwargs = dict(
        g07_tuned={
            'result': g07['result'],
            'json': f'results/gates/{G07_FILE}',
            'sha256': _sha256(EVIDENCE / G07_FILE),
            'model_details': g07.get('model_details') or {},
        },
        readout=family['readout'],
        config_hash=family['config_hash'],
        sweep_id=family['sweep_id'],
        tuning_record_sha=family['tuning_record_sha'],
        measurement_model=family['measurement_model'],
    )
    protocol = comparative.load_protocol()
    result = comparative.evaluate_family(runs, protocol, designs, **kwargs)
    return {
        'family': family, 'views': views, 'g07': g07, 'records': records, 'cfg': cfg,
        'config_path': config_path, 'runs': runs, 'designs': designs, 'kwargs': kwargs,
        'protocol': protocol, 'result': result, 'record_sha256': record_sha256,
    }


def _evaluate(evidence, runs):
    from qrc_thresher.gates import comparative

    return comparative.evaluate_family(runs, evidence['protocol'], evidence['designs'],
                                       **evidence['kwargs'])


class TestHashes:
    def test_the_manifest_verifies_and_covers_every_file_except_itself(self) -> None:
        if not EVIDENCE.is_dir():
            pytest.fail('docs/evidence/8ed2df4 does not exist')
        manifest = _read_manifest(EVIDENCE)
        files = {p.relative_to(EVIDENCE).as_posix() for p in EVIDENCE.rglob('*') if p.is_file()}
        assert set(manifest) == files - {'MANIFEST.sha256'}
        for name, digest in manifest.items():
            assert _sha256(EVIDENCE / name) == digest, name

    def test_the_family_and_g07_files_are_the_registered_bytes(self, evidence) -> None:
        assert _sha256(EVIDENCE / FAMILY_FILE) == FAMILY_SHA256
        assert _sha256(EVIDENCE / G07_FILE) == G07_SHA256
        assert evidence['family']['members']['G1']['g07']['sha256'] == G07_SHA256

    def test_each_view_matches_the_family(self, evidence) -> None:
        views = evidence['views']
        assert sorted(views) == sorted(MEMBERS)
        for member in MEMBERS:
            assert len(views[member]) == 1, member
            _, view = views[member][0]
            assert view['family_sha256'] == FAMILY_SHA256
            assert view['result'] == evidence['family']['members'][member]['result']

    def test_each_tuning_record_recomputes_its_sha(self, evidence) -> None:
        family = evidence['family']
        shas = family['tuning_record_sha']
        for task, record in evidence['records'].items():
            body = dict(record)
            recorded = body.pop('record_sha256')
            assert evidence['record_sha256'](body) == recorded, task
            expected = shas[task] if isinstance(shas, dict) else shas
            assert recorded == expected, task

    def test_the_config_copy_hashes_to_the_family_config_hash(self, evidence) -> None:
        from qrc_thresher.proof.run_manifest import _config_hash

        assert _config_hash(evidence['config_path']) == evidence['family']['config_hash']
        assert evidence['family']['config_hash'].startswith('509b9d0b')


class TestReDerivation:
    def test_every_recorded_leaf_comes_back_exactly(self, evidence) -> None:
        changed = _assert_same_leaves(evidence['family'], evidence['result'])
        assert changed == {}
        assert evidence['result']['n_rows'] == evidence['family']['n_rows'] == N_ROWS

    def test_the_verdicts(self, evidence) -> None:
        for member, verdict in VERDICTS.items():
            entry = evidence['result']['members'][member]
            assert entry['result'] == verdict, (member, entry['message'])
            assert entry['n_pairs'] == 12
            assert evidence['family']['members'][member]['result'] == verdict

    def test_the_run_3_replay_collapses_the_duplicates(self, evidence) -> None:
        runs = evidence['runs']
        copies = []
        for design in ('default', 'default_w1'):
            for ts, rs in REPLAY_PAIRS:
                picked = runs[(runs['task_name'] == 'stm') & (runs['design'] == design)
                              & (runs['task_seed'] == ts) & (runs['reservoir_seed'] == rs)]
                assert len(picked) == 1, (design, ts, rs)
                copy = picked.copy()
                copy['run_id'] = f'replay-{design}-{ts}-{rs}'
                copies.append(copy)
        assert len(copies) == 14
        replay = pd.concat([runs, *copies], ignore_index=True)
        result = _evaluate(evidence, replay)
        changed = _assert_same_leaves(evidence['family'], result, allow_changed=[('n_rows',)])
        assert set(changed) == {('n_rows',)} and result['n_rows'] == N_ROWS + 14 == 350
        # The recorded run_ids lists are new CP5 keys, so they are not recorded leaves; the
        # appended ids appear in the new result only.
        for member in ('G2.5', 'G3'):
            w1 = result['members'][member]['default_w1']
            assert (w1['status'], w1['n_rows'], w1['n_rows_in']) == ('OK', 12, 19)
            named = {rid for ids in w1['run_ids'] for rid in ids}
            assert {f'replay-default_w1-{ts}-{rs}' for ts, rs in REPLAY_PAIRS} <= named

    def test_one_differing_copy_refuses_only_the_default_w1_tables(self, evidence) -> None:
        runs = evidence['runs']
        picked = runs[(runs['task_name'] == 'stm') & (runs['design'] == 'default_w1')
                      & (runs['task_seed'] == 44) & (runs['reservoir_seed'] == 139)]
        assert len(picked) == 1
        copy = picked.copy()
        copy['run_id'] = 'differing-44-139'
        copy['primary_metric_value'] = float(copy['primary_metric_value'].iloc[0]) + 1e-9
        result = _evaluate(evidence, pd.concat([runs, copy], ignore_index=True))
        allowed = [('n_rows',), ('members', 'G2.5', 'default_w1'), ('members', 'G3', 'default_w1')]
        changed = _assert_same_leaves(evidence['family'], result, allow_changed=allowed)
        assert result['n_rows'] == N_ROWS + 1 == 337
        for member in ('G2.5', 'G3'):
            w1 = result['members'][member]['default_w1']
            assert w1['status'] == 'INSUFFICIENT_EVIDENCE' and '(44, 139)' in w1['reason']
            assert w1['n_rows'] == 0 and w1['mean'] is None and w1['std'] is None
            assert result['members'][member]['result'] == VERDICTS[member]
        touched = {p[:3] for p in changed if p != ('n_rows',)}
        assert touched <= {('members', 'G2.5', 'default_w1'), ('members', 'G3', 'default_w1')}

    def test_deployments_split_run_4_into_28_groups(self, evidence) -> None:
        from qrc_thresher.gates import comparative

        runs, designs = evidence['runs'], evidence['designs']
        record_task = {r['record_sha256']: task for task, r in evidence['records'].items()}
        labelled = runs.copy()
        labelled['deployment'] = [comparative.deployment_label(r, designs, record_task)
                                  for _, r in runs.iterrows()]
        assert not labelled['deployment'].str.contains('unresolved').any()
        groups = labelled.groupby(['task_name', 'design', 'deployment', 'primary_metric_name'])
        assert groups.ngroups == 28
        by_ids = {}
        for key, g in groups:
            assert len(g) == 12
            assert len(set(zip(g['task_seed'], g['reservoir_seed']))) == 12
            by_ids[frozenset(g['run_id'].astype(str))] = key
        for member in MEMBERS:
            entry = evidence['result']['members'][member]
            for table in (entry['comparison'], entry['default']):
                for side in ('a', 'b'):
                    ids = frozenset(rid for ids in table[f'run_ids_{side}'] for rid in ids)
                    assert ids in by_ids, (member, side)
            w1 = entry['default_w1']
            ids = frozenset(rid for ids in w1['run_ids'] for rid in ids)
            assert by_ids[ids][:3] == (entry['task'], 'default_w1', 'default_w1')
        from qrc_thresher.commands.summary import summary_table

        table, notes = summary_table(runs, resolver=lambda *a, **k: (designs, record_task, None))
        assert len(table) == 28 and notes == []
        assert (table['n_rows'] == 12).all() and (table['n'] == 12).all()

    def test_an_unrelated_failed_row_changes_nothing(self, evidence, tmp_path) -> None:
        from qrc_thresher.gates import comparative

        text = (EVIDENCE / 'runs.sanitised.csv').read_text(encoding='utf-8')
        header = text.splitlines()[0].split(',')
        failed = {name: '' for name in header}
        failed.update({
            'run_id': 'unrelated-failed-row', 'success': 'False',
            'git_commit_hash': '8ed2df4d082ee6613e9a9480d5f5f41b99b23f4e',
            'config_hash': 'f' * 64, 'sweep_id': '', 'tuning_record_sha': '',
            'circuit_hash': 'f' * 64, 'task_seed': '42', 'reservoir_seed': '137',
            'task_name': 'stm', 'primary_metric_name': '', 'primary_metric_value': '',
            'measurement_model': 'exact', 'n_configs': '1', 'n_validation_evals': '0',
            'secondary_metrics': '{}', 'design': 'default',
        })
        appended = tmp_path / 'runs.csv'
        appended.write_text(text + ','.join(failed[name] for name in header) + '\n',
                            encoding='utf-8')
        runs = comparative.read_runs_csv(appended)
        assert runs['primary_metric_value'].dtype == np.float64
        assert len(runs) == N_ROWS + 1 and np.isnan(runs['primary_metric_value'].iloc[-1])
        result = _evaluate(evidence, runs)
        assert _assert_same_leaves(evidence['family'], result) == {}
        assert result['n_rows'] == N_ROWS
