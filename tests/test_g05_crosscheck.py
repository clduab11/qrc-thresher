"""G0.5, the PennyLane-vs-Qiskit cross-check, covers every reservoir design at one tolerance
(defects D14 and D6; docs/DECISIONS.md D010).

- CROSSCHECK_TOLERANCE = 1e-6 (float64) is defined once in src/, the gate imports it, and no
  literal tolerance is passed to verify_crosscheck.
- The Qiskit side builds its circuit independently: it imports nothing from PennyLane or from
  qrc_thresher, directly, relatively or dynamically.
- G0.5 checks (n, L, seed) = (2, 2, 2026), (4, 3, 137) and (5, 4, 7), at each distinct w in
  {1, 2, n}, with both readouts, over 6 steps that include the zero-padded rows, at the
  encoding scale pi (16 cases), plus the registered scale cases G05_SCALE_CASES: scale in
  {pi/4, pi/2, 3pi/4} on (4, 3, 137) at w in {1, 2, 4} with both readouts (18 cases; D011).
  Its evidence lists every case, keyed by (n, depth, seed, window, readout, scale), with its
  max |diff|: 34 cases.
- Power, on the (4, 3, 137) triple (6 cases) to keep the suite fast:
  - on the Qiskit side, changing one angle by 1e-3 or shifting the window by one step makes
    every case fail;
  - the tolerance is 1e-6 and not looser: an error of 2e-6 in the last feature of the last row
    fails every case, and one of 5e-7 in the first feature of the first row passes; both runs
    also check that the Qiskit side was called once per case, with the case's window and
    readout, the (4, 3, 137) triple and inputs, and a (6, F) output;
  - on the PennyLane side, shifting WindowedReservoir.features by 1e-3 fails every case, which
    ties G0.5 to the production reservoir;
  - on the scale cases, a Qiskit side that ignores the requested scale (and so encodes at pi)
    fails every scale case: the scale axis is live on both sides.
"""

from __future__ import annotations

import ast
import functools
import importlib
import math
from pathlib import Path

import numpy as np

from qrc_thresher.reservoirs.pennylane_qrc import build_reservoir_params

REPO_ROOT = Path(__file__).parent.parent
SRC = REPO_ROOT / 'src' / 'qrc_thresher'
QISKIT_MODULE = SRC / 'reservoirs' / 'qiskit_crosscheck.py'
TRIPLES = [(2, 2, 2026), (4, 3, 137), (5, 4, 7)]  # (n_qubits, depth, seed)
READOUTS = ['z_only', 'z_and_zz']
SCALES = [math.pi / 4, math.pi / 2, 3 * math.pi / 4]  # the registered scale cases (D011)
EXPECTED_PI_CASES = {
    (n, depth, seed, window, readout, math.pi)
    for n, depth, seed in TRIPLES
    for window in sorted({1, 2, n})
    for readout in READOUTS
}
EXPECTED_SCALE_CASES = {
    (4, 3, 137, window, readout, scale)
    for scale in SCALES
    for window in (1, 2, 4)
    for readout in READOUTS
}
EXPECTED_CASES = EXPECTED_PI_CASES | EXPECTED_SCALE_CASES
N_STEPS = 6
POWER_TRIPLES = ((4, 3, 137),)
POWER_CASES = {(window, readout) for window in (1, 2, 4) for readout in READOUTS}
N_FEATURES = {'z_only': 4, 'z_and_zz': 10}  # F at n = 4


def _gate():
    return importlib.import_module('qrc_thresher.commands.gate')


def _qc():
    return importlib.import_module('qrc_thresher.reservoirs.qiskit_crosscheck')


def _wq():
    """Import the windowed reservoir module (absent until CP3b)."""
    return importlib.import_module('qrc_thresher.reservoirs.windowed_qrc')


@functools.lru_cache(maxsize=None)
def _g05() -> tuple:
    """Evaluate G0.5 once; the coverage tests below only read the result."""
    return _gate()._evaluate_gate_g05()


def _name(node: ast.AST):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.UnaryOp):
        node = node.operand
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float))


class TestOneTolerance:
    def test_the_tolerance_is_defined_once_and_the_gate_imports_it(self) -> None:
        definitions = []
        for path in sorted(SRC.rglob('*.py')):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                targets = []
                if isinstance(node, ast.Assign):
                    targets = node.targets
                elif isinstance(node, ast.AnnAssign):
                    targets = [node.target]
                for target in targets:
                    if _name(target) in ('CROSSCHECK_TOLERANCE', '_CROSSCHECK_TOLERANCE'):
                        definitions.append(f'{path.relative_to(SRC).as_posix()}:{node.lineno}')
        assert len(definitions) == 1, definitions
        assert definitions[0].startswith('reservoirs/qiskit_crosscheck.py:')
        assert _qc().CROSSCHECK_TOLERANCE == 1e-6
        gate_tree = ast.parse((SRC / 'commands' / 'gate.py').read_text(encoding='utf-8'))
        imports = {
            (node.module, alias.name)
            for node in ast.walk(gate_tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert ('qrc_thresher.reservoirs.qiskit_crosscheck', 'CROSSCHECK_TOLERANCE') in imports

    def test_no_literal_tolerance_is_passed_to_verify_crosscheck(self) -> None:
        offenders = []
        for path in sorted(SRC.rglob('*.py')):
            rel = path.relative_to(SRC).as_posix()
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node, ast.Call) and _name(node.func) == 'verify_crosscheck':
                    passed = list(node.args[2:]) + [
                        kw.value for kw in node.keywords if kw.arg == 'tolerance'
                    ]
                    if any(_is_literal(arg) for arg in passed):
                        offenders.append(f'{rel}:{node.lineno}')
                if isinstance(node, ast.FunctionDef) and node.name == 'verify_crosscheck':
                    defaults = node.args.defaults
                    if not defaults or _name(defaults[-1]) != 'CROSSCHECK_TOLERANCE':
                        offenders.append(f'{rel}:{node.lineno} (default)')
        assert offenders == [], offenders

    def test_the_qiskit_side_imports_nothing_from_pennylane_or_the_harness(self) -> None:
        forbidden = []
        for node in ast.walk(ast.parse(QISKIT_MODULE.read_text(encoding='utf-8'))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    forbidden.append(f'line {node.lineno}: relative import')
                modules = [node.module or '']
            elif isinstance(node, ast.Call) and _name(node.func) in ('__import__', 'import_module'):
                forbidden.append(f'line {node.lineno}: dynamic import')
                continue
            else:
                continue
            forbidden += [
                f'line {node.lineno}: {m}'
                for m in modules
                if m.split('.')[0] in ('pennylane', 'qrc_thresher', 'importlib')
            ]
        assert forbidden == []


class TestCoverage:
    def test_g05_passes_on_every_case(self) -> None:
        result, evidence, _ = _g05()
        assert result == 'PASS', evidence
        assert evidence['tolerance'] == 1e-6

    def test_the_evidence_lists_every_case_with_its_max_diff(self) -> None:
        _, evidence, _ = _g05()
        cases = evidence['cases']
        listed = [
            (c['n_qubits'], c['depth'], c['seed'], c['window'], c['readout'], c['encoding_scale'])
            for c in cases
        ]
        assert len(listed) == len(EXPECTED_CASES) == 34
        assert set(listed) == EXPECTED_CASES
        assert len(EXPECTED_PI_CASES) == 16 and len(EXPECTED_SCALE_CASES) == 18
        for case in cases:
            assert case['n_steps'] == N_STEPS
            assert case['n_zero_padded_rows'] == case['window'] - 1
            assert np.isfinite(case['max_abs_diff'])
            assert case['max_abs_diff'] <= 1e-6
            assert case['passed'] is True

    def test_the_registered_scale_cases_are_separate_from_the_triples(self) -> None:
        gate = _gate()
        assert gate.G05_TRIPLES == ((2, 2, 2026), (4, 3, 137), (5, 4, 7))
        assert set(gate.G05_SCALE_CASES) == EXPECTED_SCALE_CASES
        assert len(gate.G05_SCALE_CASES) == 18


def _perturb_one_feature(delta: float, index: int, calls: list):
    """Qiskit-side wrapper: add ``delta`` to out.flat[index] of every case, and record each call."""

    def wrap(real):
        def perturbed(u, thetas, phis, n_qubits, depth, window, readout, encoding_scale=math.pi):
            out = np.array(
                real(u, thetas, phis, n_qubits, depth, window, readout,
                     encoding_scale=encoding_scale),
                dtype=np.float64,
                copy=True,
            )
            calls.append(
                {
                    'n_qubits': n_qubits,
                    'depth': depth,
                    'window': window,
                    'readout': readout,
                    'encoding_scale': encoding_scale,
                    'shape': out.shape,
                    'u': np.array(u, dtype=np.float64, copy=True),
                }
            )
            out.flat[index] += delta
            return out

        return perturbed

    return wrap


def _assert_calls_cover_the_power_cases(calls: list) -> None:
    """One Qiskit call per (window, readout) case, on the registered triple and inputs."""
    assert sorted((c['window'], c['readout']) for c in calls) == sorted(POWER_CASES)
    rng = np.random.default_rng(137)
    build_reservoir_params(
        n_qubits=4, depth=3, readout='z_only', backend='default.qubit', rng=rng
    )
    inputs = rng.uniform(-1.0, 1.0, size=N_STEPS)  # the next 6 draws after the angles
    for call in calls:
        assert (call['n_qubits'], call['depth']) == (4, 3), call
        assert call['encoding_scale'] == math.pi, call
        assert call['shape'] == (N_STEPS, N_FEATURES[call['readout']]), call
        assert np.array_equal(call['u'], inputs), call


class TestPower:
    @staticmethod
    def _run(monkeypatch, qiskit_wrap=None, pennylane_wrap=None, scale_cases=()) -> tuple:
        """G0.5 on the power triple at pi (6 cases); the scale cases only when asked for."""
        gate, qc = _gate(), _qc()
        if qiskit_wrap is not None:
            monkeypatch.setattr(qc, 'qiskit_features', qiskit_wrap(qc.qiskit_features))
        if pennylane_wrap is not None:
            wq = _wq()
            wrapped = pennylane_wrap(wq.WindowedReservoir.features)
            monkeypatch.setattr(wq.WindowedReservoir, 'features', wrapped)
        monkeypatch.setattr(gate, 'G05_TRIPLES', POWER_TRIPLES if not scale_cases else ())
        monkeypatch.setattr(gate, 'G05_SCALE_CASES', tuple(scale_cases))
        return gate._evaluate_gate_g05()

    @staticmethod
    def _cases(evidence) -> list:
        assert 'error' not in evidence, evidence.get('error')
        cases = evidence['cases']
        assert len(cases) == len(POWER_CASES) == 6
        assert {(c['window'], c['readout']) for c in cases} == POWER_CASES
        assert all((c['n_qubits'], c['depth'], c['seed']) == (4, 3, 137) for c in cases)
        assert all(np.isfinite(c['max_abs_diff']) for c in cases)
        return cases

    def _assert_every_case_failed(self, result, evidence) -> list:
        cases = self._cases(evidence)
        assert result == 'FAIL'
        assert all(c['passed'] is False and c['max_abs_diff'] > 1e-6 for c in cases), cases
        return cases

    def test_one_angle_off_by_1e_3_fails(self, monkeypatch) -> None:
        def wrap(real):
            def one_angle_off(u, thetas, phis, n_qubits, depth, window, readout,
                              encoding_scale=math.pi):
                thetas = np.array(thetas, dtype=np.float64, copy=True)
                thetas[0, 0] += 1e-3
                return real(u, thetas, phis, n_qubits, depth, window, readout,
                            encoding_scale=encoding_scale)

            return one_angle_off

        self._assert_every_case_failed(*self._run(monkeypatch, qiskit_wrap=wrap)[:2])

    def test_a_window_shifted_by_one_step_fails(self, monkeypatch) -> None:
        def wrap(real):
            def shifted(u, thetas, phis, n_qubits, depth, window, readout,
                        encoding_scale=math.pi):
                u = np.asarray(u, dtype=np.float64)
                late = np.concatenate([[0.0], u[:-1]])  # every qubit sees one step further back
                return real(late, thetas, phis, n_qubits, depth, window, readout,
                            encoding_scale=encoding_scale)

            return shifted

        self._assert_every_case_failed(*self._run(monkeypatch, qiskit_wrap=wrap)[:2])

    def test_an_error_of_2e_6_in_the_last_feature_fails(self, monkeypatch) -> None:
        calls: list = []
        wrap = _perturb_one_feature(2e-6, -1, calls)  # last feature of the last row
        result, evidence, _ = self._run(monkeypatch, qiskit_wrap=wrap)
        cases = self._assert_every_case_failed(result, evidence)
        assert all(abs(c['max_abs_diff'] - 2e-6) < 1e-9 for c in cases), cases
        _assert_calls_cover_the_power_cases(calls)

    def test_an_error_of_5e_7_in_the_first_feature_passes(self, monkeypatch) -> None:
        calls: list = []
        wrap = _perturb_one_feature(5e-7, 0, calls)  # first feature of the first row
        result, evidence, _ = self._run(monkeypatch, qiskit_wrap=wrap)
        cases = self._cases(evidence)
        assert result == 'PASS', cases
        assert all(c['passed'] is True for c in cases)
        assert all(abs(c['max_abs_diff'] - 5e-7) < 1e-9 for c in cases), cases
        _assert_calls_cover_the_power_cases(calls)

    def test_the_pennylane_side_is_the_production_reservoir(self, monkeypatch) -> None:
        def wrap(real):
            def shifted(self, u, *args, **kwargs):
                return np.asarray(real(self, u, *args, **kwargs), dtype=np.float64) + 1e-3

            return shifted

        self._assert_every_case_failed(*self._run(monkeypatch, pennylane_wrap=wrap)[:2])

    def test_a_qiskit_side_that_ignores_the_scale_fails_every_scale_case(self, monkeypatch) -> None:
        calls: list = []

        def wrap(real):
            def at_pi(u, thetas, phis, n_qubits, depth, window, readout, encoding_scale=math.pi):
                calls.append(encoding_scale)
                return real(u, thetas, phis, n_qubits, depth, window, readout)  # pi, whatever

            return at_pi

        result, evidence, _ = self._run(monkeypatch, qiskit_wrap=wrap,
                                        scale_cases=_gate().G05_SCALE_CASES)
        assert 'error' not in evidence, evidence.get('error')
        cases = evidence['cases']
        assert len(cases) == 18
        assert {(c['window'], c['readout'], c['encoding_scale']) for c in cases} == {
            (w, r, s) for w in (1, 2, 4) for r in READOUTS for s in SCALES
        }
        assert sorted(calls) == sorted(c['encoding_scale'] for c in cases)
        assert result == 'FAIL'
        assert all(c['passed'] is False and c['max_abs_diff'] > 1e-6 for c in cases), cases
        # The same 18 cases pass when the scale reaches the Qiskit side.
        monkeypatch.undo()
        result, evidence, _ = self._run(monkeypatch, scale_cases=_gate().G05_SCALE_CASES)
        assert result == 'PASS', evidence
        assert all(c['passed'] for c in evidence['cases']) and len(evidence['cases']) == 18
