"""Known answer: the no-entangle ablation of the w = 2 reservoir cannot represent window-2
parity (defects D1 and D12; docs/DECISIONS.md D010).

Without the CNOT ring each qubit evolves alone, and at w = 2 qubit j re-uploads u_t (j even) or
u_{t-1} (j odd), never both. Under the z_only readout every column is then a function of one
input bit, so its interaction contrast X(1,1) - X(1,0) - X(0,1) + X(0,0) is zero. XOR has
contrast -2, so it lies outside the span of the columns plus an intercept, and the G0.7 v1
parity clause is expected to fail.

That is not guaranteed. Under an additive least-squares fit a (u_{t-1}, u_t) cell is predicted
right exactly when its training count exceeds the harmonic mean of the four cell counts, so
about 41% of seeds reach 75% held-out accuracy and pass at p = 1/201. The clause needs all three
seeds, so it passes with prior probability about 7% in the least-squares limit, or about 1-5%
under the registered ridge readout (D010). The test asserts the clause, not each seed, and that
no seed is degenerate. If it goes red, the result is reported as a finding about G0.7 v1 and
left red: no seed, n, depth, encoding scale, window, readout, alpha, threshold or test changes,
and no xfail.

The full circuit's contrast is printed and recorded, not asserted; its parity pass is an
expectation (D010), not a target. There is no ZZ variant.

Setting: configs/windowed_w2.yaml (n 4, depth 3, w 2, z_only), the config's seed pairs, and the
G0.7 v1 machinery (the gate's own seed evaluation and clause functions).
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from qrc_thresher.config import load_config

REPO_ROOT = Path(__file__).parent.parent
WINDOWED_W2 = REPO_ROOT / 'configs' / 'windowed_w2.yaml'
SEED_PAIRS = [(42, 137), (43, 138), (44, 139)]
CONTRAST_ATOL = 1e-12

# Rows 1..4 of this input see (u_{t-1}, u_t) = (0, 0), (0, 1), (1, 1), (1, 0).
PATTERN_INPUT = np.array([0.0, 0.0, 1.0, 1.0, 0.0])


def _wq():
    """Import the windowed reservoir module (absent until CP3b)."""
    return importlib.import_module('qrc_thresher.reservoirs.windowed_qrc')


def _g07():
    return importlib.import_module('qrc_thresher.gates.g07')


def _contrast(X: np.ndarray) -> np.ndarray:
    """X(1,1) - X(1,0) - X(0,1) + X(0,0), per column, with X(a, b) at (u_{t-1}, u_t) = (a, b)."""
    x00, x01, x11, x10 = X[1], X[2], X[3], X[4]
    return x11 - x10 - x01 + x00


def _windowed_w2():
    cfg = load_config(WINDOWED_W2)
    reservoir = cfg.reservoir
    assert (reservoir.n_qubits, reservoir.depth, reservoir.window, reservoir.readout) == (
        4,
        3,
        2,
        'z_only',
    )
    return cfg


def test_no_entangle_columns_have_zero_interaction_contrast(record_property) -> None:
    wq = _wq()
    cfg = _windowed_w2()
    full_circuit = {}
    for _, reservoir_seed in _g07().seed_pairs_from_config(cfg):
        ablated = wq.reservoir_from_config(cfg, reservoir_seed, ablation='no_entangle')
        contrast = _contrast(ablated.features(PATTERN_INPUT))
        assert np.max(np.abs(contrast)) <= CONTRAST_ATOL, (reservoir_seed, contrast)
        full = wq.reservoir_from_config(cfg, reservoir_seed).features(PATTERN_INPUT)
        full_circuit[reservoir_seed] = [round(float(c), 6) for c in _contrast(full)]
    # Reported, not asserted.
    for reservoir_seed, contrast in full_circuit.items():
        print(f'full-circuit contrast, reservoir seed {reservoir_seed}: {contrast}')
    record_property('full_circuit_contrast', full_circuit)


def test_the_no_entangle_ablation_fails_the_g07_parity_clause(monkeypatch) -> None:
    g07 = _g07()
    cfg = _windowed_w2()
    protocol = g07.load_protocol()

    captured = {}

    def capture(feature_map, protocol, seed_pairs, **kwargs):
        captured.update(feature_map=feature_map, seed_pairs=list(seed_pairs), **kwargs)
        return {}

    # Take the feature map exactly as `gate G0.7 --model no_entangle` builds it.
    monkeypatch.setattr(g07, 'evaluate', capture)
    g07.evaluate_config(cfg, protocol, model='no_entangle')
    assert captured['model_details']['ablation'] == 'no_entangle'
    assert captured['model_details']['window'] == 2
    pairs = [(int(t), int(r)) for t, r in captured['seed_pairs']]
    assert pairs == SEED_PAIRS

    seeds = [
        g07._evaluate_parity_seed(captured['feature_map'], protocol, t, r) for t, r in pairs
    ]
    evidence = [(s['accuracy'], s['p'], s['degenerate_reason']) for s in seeds]
    assert not any(s['degenerate'] for s in seeds), evidence
    clause = g07._clause(
        seeds,
        protocol.parity.significance_level,
        len(pairs) >= protocol.seeds.min_seeds,
        'heldout_accuracy',
    )
    assert clause['result'] == 'FAIL', evidence
