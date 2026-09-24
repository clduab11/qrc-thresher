"""The baseline run path writes seed-keyed ESN rows (defect D8; docs/DECISIONS.md D009, D013).

The end-to-end test proves plumbing only. Since CP4 the comparative gates are the family of
D013: on a config without a tuning block the family is INSUFFICIENT_EVIDENCE (no tuning record),
it names the record, and it writes no fixed-name results/gates/G3.json. The tiny config keeps
5 seed pairs; its ESN rows are tuned in-line over baseline.esn_grid (design tuned, empty
sweep_id) and its QRC rows are the untuned default (design default).
"""

from __future__ import annotations

import csv
from pathlib import Path

import yaml

from qrc_thresher.proof.run_manifest import CSV_FIELDNAMES, SCHEMA_VERSION, create_manifest

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG = REPO_ROOT / 'configs' / 'alpha_lite.yaml'
TINY_GRID = {'spectral_radius': [0.9], 'input_scaling': [0.1, 1.0], 'leak_rate': [1.0]}


def _tiny_config(tmp_path: Path) -> Path:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding='utf-8'))
    raw['experiment_name'] = 'tiny_baseline_e2e'
    raw['task'].update({'length': 150, 'delay_max': 5})
    raw['reservoir'].update({'n_qubits': 2, 'depth': 1})
    raw['seeds']['n_seeds'] = 5
    raw['baseline']['esn_grid'] = TINY_GRID
    raw['training']['washout'] = 20  # one washout for every model (D012)
    path = tmp_path / 'configs' / 'tiny.yaml'
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(raw), encoding='utf-8')
    return path


def _rows(path: Path) -> list:
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def test_manifest_records_the_search_budget(tmp_path: Path) -> None:
    from qrc_thresher.proof.run_manifest import append_to_csv

    assert SCHEMA_VERSION == '1.4'
    assert {'n_configs', 'n_validation_evals'} <= set(CSV_FIELDNAMES)
    manifest = create_manifest(
        config_path=DEFAULT_CONFIG, circuit_hash='abc', task_seed=42, reservoir_seed=137,
        backend_device='numpy', runtime_per_stage_seconds={}, entanglement_metric=None,
        success=True, failure_reason=None, artifact_paths=[], n_configs=60,
        n_validation_evals=300,
    )
    append_to_csv(manifest, csv_path=tmp_path / 'runs.csv')
    row = _rows(tmp_path / 'runs.csv')[0]
    assert (row['n_configs'], row['n_validation_evals']) == ('60', '300')


def test_baseline_rows_are_seed_keyed_and_the_family_needs_a_record(
    tmp_path: Path, monkeypatch
) -> None:
    from qrc_thresher.commands.baseline import run_baselines
    from qrc_thresher.commands.gate import gate_handler
    from qrc_thresher.config import load_config
    from qrc_thresher.engine import ParallelRunner

    cfg_path = _tiny_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    cfg = load_config(cfg_path)
    ParallelRunner(config=cfg, max_workers=1).run_seeds('stm', config_path=cfg_path)
    manifests = run_baselines(cfg, 'stm', cfg_path)

    rows = _rows(Path('results') / 'runs.csv')
    qrc = [r for r in rows if r['task_name'] == 'stm']
    esn = [r for r in rows if r['task_name'] == 'esn']
    assert len(qrc) == len(esn) == 5 == len(manifests)
    pairs = [(42 + i, 137 + i) for i in range(5)]
    assert [(int(r['task_seed']), int(r['reservoir_seed'])) for r in esn] == pairs
    assert all(r['success'] == 'True' and r['primary_metric_name'] == 'stm_memory' for r in esn)
    assert all(r['primary_metric_name'] == 'stm_memory' for r in qrc)
    assert all((r['n_configs'], r['n_validation_evals']) == ('2', '10') for r in esn)
    assert all((r['n_configs'], r['n_validation_evals']) == ('1', '0') for r in qrc)
    assert all(r['design'] == 'tuned' and r['sweep_id'] == '' for r in esn)
    assert all(r['design'] == 'default' and r['sweep_id'] == '' for r in qrc)
    assert len({r['circuit_hash'] for r in esn}) == 5  # one reservoir draw per seed pair

    exit_code = gate_handler('G3', config_path=str(cfg_path))
    assert exit_code == 2  # INSUFFICIENT_EVIDENCE: no tuning block, so no record and no family
    assert not (Path('results') / 'gates' / 'G3.json').exists()


def test_narma_baseline_rows_are_what_g4_reads(tmp_path: Path, monkeypatch) -> None:
    from qrc_thresher.commands.baseline import run_baselines
    from qrc_thresher.config import load_config

    cfg_path = _tiny_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    run_baselines(load_config(cfg_path), 'narma', cfg_path)
    rows = [r for r in _rows(Path('results') / 'runs.csv') if r['task_name'] == 'esn_narma']
    assert len(rows) == 5
    assert all(r['primary_metric_name'] == 'nrmse' and float(r['primary_metric_value']) > 0
               for r in rows)
