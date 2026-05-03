# QRC-THRESHER MASTER AUGMENTATION PLAN 2026
## Multi-Agent Synthesis Report | System Analyst × Technical Architect × Security Auditor × Quality Engineer

**Repository:** `qrc-thresher` v0.1.0 (Phase 1 Alpha-Lite)  
**Date:** 2026-05-03  
**Mission:** Falsification-first quantum reservoir computing benchmark with rigorous provenance, statistical integrity, and 2026-grade engineering practices

---

## EXECUTIVE SUMMARY

The qrc-thresher codebase represents an exemplary **scientific prototype**: mathematically precise, determinism-first, with a well-documented methodology (850+ line BUILD_SPEC). However, it exhibits classic Phase-1 technical debt that blocks Phase 1.5 gate completion and threatens long-term reproducibility.

**Overall Risk Assessment:** MEDIUM-HIGH
- **6 Critical (P0)** findings that break scientific integrity or block Phase 1
- **12 High (P1)** findings impairing reproducibility, maintainability, or gate completion
- **8 Medium (P2)** best practice deviations with moderate risk
- **4 Low (P3)** improvements with low impact

**Strategic Direction:** Implement in two parallel tracks:
- **Track A (Stabilization):** Fix P0 blockers blocking gate evaluation (4 weeks)
- **Track B (Modernization):** Modular architecture with plugin framework and DAG orchestration (6 weeks)

Both tracks converge at **Phase 1.5 milestone** (complete G0–G5 gates, enable manuscript-track sweeps).

---

## 1. CURRENT ARCHITECTURE DIAGNOSIS (SYSTEM ANALYST)

### 1.1 Architecture Overview

```
YAML Config → Pydantic → CLI (God module) → Sequential pipeline:
  task_gen → reservoir_build → feature_extraction → readout_train → eval → manifest
```

**Strengths:**
- Clean module boundaries: tasks/reservoirs/baselines/metrics/proof are internally decoupled
- Immutable dataclasses throughout (`frozen=True, slots=True`)
- Explicit RNG seeding (`task_seed`, `reservoir_seed`, ablation offsets)
- Manifest v1.1 captures comprehensive provenance (git commit with -dirty flag, package versions, platform, per-stage timings)
- Finite-value guards in all metric functions (`scoring.py` raises on NaN/inf)

**Critical Bottlenecks (P0–P1):**

| # | Bottleneck | Location | Impact | Severity |
|---|-----------|----------|--------|----------|
| B1 | **Serial execution** — no parallelization across seeds | CLI runs single (seed,RR) per invocation | 15 seeds × 3 RRs = hours wasted | P1 |
| B2 | **Gate evaluators stubbed** (G0.5, G3, G4, G5) | `cli.py:430–506` | Cannot evaluate gates; CI gates disabled (`continue-on-error`) | **P0** |
| B3 | **Hardcoded task routing** — no plugin registry | `cli.py: if task=='stm'...` | New tasks require core edit; violates Open/Closed | P1 |
| B4 | **Ablation metric wrong** — parity ablation logs MC not accuracy | `cli.py:396` | G2 gate cannot consume parity ablation results | **P1** |
| B5 | **NARMA-10 gated but not integrated** | `cli.py:137–140` gate check only | Cannot run NARMA without manual config edit | P1 |
| B6 | **ESN grid search serial loop** | `baselines/esn.py:95–123` | 300+ model fits serially; 30–60s overhead per ESN | P2 |
| B7 | **No artifact storage** — figures not written | CLI plot stub; `artifact_paths=[]` | Cannot reconstruct figures for paper | P2 |
| B8 | **Manifest lacks gate verdict linkage** | Schema v1.1 no `gates_passed` column | Cannot trace which gates passed for a run | P1 |
| B9 | **Config immutability policy not mechanized** | Policy only; no runtime guard | User could edit config after runs, breaking provenance | P2 |
| B10 | **Qiskit cross-check unused** | `qiskit_crosscheck.py:72` never called | G0.5 gate cannot run; backend consistency unvalidated | P1 |

**Technical Debt Register (Prioritized):**

| ID | Debt Item | Severity | Effort | Fix |
|----|-----------|----------|--------|-----|
| TD-001 | God CLI module (745 lines, imports all) | **P0** | 8h | Extract to `commands/` package |
| TD-002 | Gates G0.5/G3/G4/G5 stubbed | **P0** | 16h | Implement evaluators; test with synthetic data |
| TD-003 | Holm-Bonferroni correction missing | **P0** | 6h | Implement `stats.holm_bonferroni()`; integrate |
| TD-004 | Effect size thresholds hardcoded (not in config) | **P0** | 4h | Add `gates:` block to config schema |
| TD-005 | **No lockfile** — dependency drift risk | **P0** | 4h | Generate `uv.lock`; CI enforce install from lock |
| TD-006 | Parity ablation logs MC not accuracy | P1 | 2h | Fix to use task's primary metric |
| TD-007 | NARMA integration incomplete | P1 | 4h | Complete run/ablation paths; gate check |
| TD-008 | Qiskit cross-check unused | P1 | 8h | Integrate into G0.5 gate; log residuals |
| TD-009 | Config immutability not mechanized | P2 | 4h | Add lockfile hash to manifest; reject modified config |
| TD-010 | Manifest lacks gate linkage | P1 | 4h | Add `gates_passed` JSON column |
| TD-011 | CI health check `continue-on-error` | P1 | 1h | Remove; fix underlying failures |
| TD-012 | Single Python version in CI (3.11 only) | P2 | 2h | Add 3.12, 3.13 to matrix |
| TD-013 | Plot command no-op | P3 | 6h | Wire up `viz/plots.py` |
| TD-014 | GRU stub `NotImplementedError` | P2 | 12h | Implement (Phase 2) |
| TD-015 | No parallel execution | P1 | 12h | Process pool with progress bar |
| TD-016 | Type hints gaps | P2 | 6h | Run `mypy --strict` on key modules |
| TD-017 | No timeout/retry logic | P2 | 8h | Add `--timeout` and retry wrapper |
| TD-018 | Results DB absent (flat CSV only) | P2 | 8h | SQLite with indices |
| TD-019 | Test coverage gaps on gates | P1 | 12h | Gate evaluator unit tests |
| TD-020 | No performance profiling | P3 | 8h | Benchmark suite |

---

## 2. SECURITY & REPRODUCIBILITY AUDIT (SECURITY AUDITOR)

### Critical Findings (P0)

#### SEC-001: **No Dependency Lockfile — Reproducibility Broken** (REP-001)
**Severity:** Critical | **Category:** Reproducibility / Supply Chain  
**Location:** `pyproject.toml` (version ranges), no `uv.lock`  
**Description:** Dependencies use version ranges (`pennylane>=0.44,<0.45`). Six months from now, `pip install` may resolve to different patch versions, altering numerical results. Manifest captures package versions at run time (good), but install-time drift cannot reproduce exact environment.  
**Impact:** Reviewer cannot reproduce published results due to dependency drift.  
**Fix (Phase 1.5 Week 1):**
```bash
uv pip compile pyproject.toml -o uv.lock  # or pip-compile
git add uv.lock
```
CI install: `pip install -r uv.lock`. Pre-commit hook to detect unpinned changes.  
**Target Milestone:** Phase 1.5 (before manuscript-track sweeps)  
**BUILD_SPEC Violation:** §6.4, ASSUMED-DEFAULT 6.A ("uv is preferred" but not enforced)

---

#### SEC-002: ** Holm-Bonferroni Multiple Comparison Correction Not Implemented** (REP-002)
**Severity:** Critical | **Category:** Statistical Integrity  
**Location:** Specified in `docs/METHODOLOGY.md:559`; absent in `src/metrics/stats.py`  
**Description:** Pre-registered methodology requires Holm-Bonferroni correction across all primary task comparisons. Current `stats.py` has `paired_test` and `wilcoxon_test` but no `holm_bonferroni`. Gate evaluators G3/G4 therefore cannot apply correction.  
**Impact:** Family-wise error rate inflated; positive claims may be false positives.  
**Fix (Phase 1.5 Week 2):**
```python
def holm_bonferroni(p_values: List[float]) -> List[float]:
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    adjusted = np.minimum(1.0, np.maximum.accumulate(n - np.arange(n)) * sorted_p)
    corrected = np.zeros_like(adjusted)
    corrected[sorted_indices] = adjusted
    return corrected.tolist()
```
Integrate into G3/G4 gates.  
**Target:** G3/G4 implementation  
**BUILD_SPEC Violation:** §13.5, §14.1

---

#### SEC-003: **Pre-registered Effect Size Thresholds Not Enforced** (REP-003)
**Severity:** Critical | **Category:** Statistical Integrity  
**Location:** Config has no `gates:` block; gate code uses hardcoded numbers (`qrc_mean > 1.0`, `margin_pct >= 0.20`)  
**Description:** Thresholds specified only in BUILD_SPEC §13.4; not in `configs/alpha_lite.yaml`. Gate criteria can be silently changed after seeing data (p-hacking).  
**Impact:** Scientific claim criteria not pre-registered; invalidates falsification-first discipline.  
**Fix (Phase 1.5 Week 2):**
```yaml
# configs/alpha_lite.yaml
gates:
  G1_stm_mc_threshold: 1.0
  G1_margin_pct: 0.20
  G2_accuracy_threshold: 0.70
  G2_random_features_max: 0.60
  G25_effect_se: 1.0
```
Config schema extends with optional `gates: GateThresholds` block.  
**Target:** G1/G2 implementation

---

#### SEC-004: **Gate Verdict Files Overwritable Without Audit Trail** (REP-004)
**Severity:** High | **Location:** `cli.py:686` writes `results/gates/<name>.json`  
**Description:** Re-running gate overwrites previous verdict. No timestamp, no signature, no history.  
**Impact:** User can re-run gate after modifying `runs.csv`; no record of gate flip.  
**Fix (Phase 1.5 Week 3):**
```python
ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
filepath = gates_dir / f'{name}.{ts}.json'
(gates_dir / f'{name}.latest.json').symlink_to(filepath.name)  # Windows: copy
```
Add `verdict_history` table to SQLite DB; each evaluation recorded with commit hash.  
**Target:** Gate framework implementation

---

#### SEC-005: **Concurrent CSV Writes Not Protected** (REP-005)
**Severity:** Medium | **Location:** `run_manifest.py:286` opens `runs.csv` append mode no lock  
**Description:** With `--parallel N`, multiple processes could interleave writes → corrupted CSV row.  
**Impact:** Lost runs, malformed CSV, gate evaluation failure.  
**Fix:** Phase 1.5 parallel engine uses SQLite WAL (preferred) or file lock (`filelock.FileLock`). Short-term: per-worker temp files + atomic merge.  
**Target:** Parallel execution implementation

### High Findings (P1)

- **SEC-006:** CI health check `continue-on-error: true` — builds pass despite broken env
- **SEC-007:** ReservoirPy version range too wide (`>=0.3.11`) — pin exact version
- **SEC-008:** Circuit hash uses string concatenation — use JSON canonical serialization
- **SEC-009:** Config immutability policy not mechanized — add lockfile hash to manifest
- **SEC-010:** Manifest lacks code hash — only git commit, not file-level snapshot (Phase 2)

### Medium Findings (P2)

- SEC-011: No timeout on long-running feature extraction
- SEC-012: No retry on transient BLAS/LAPACK errors
- SEC-013: ESN grid search exhaustive; no early-stopping
- SEC-014: Logging not structured JSON (only basicConfig)
- SEC-015: `results/` gitignored; no backup strategy specified

---

## 3. QUALITY ENGINEER ASSESSMENT — TEST & RELIABILITY GAPS

### Test Coverage Map (Estimated)

| Module | Lines | Test File | Est. Cov | Critical Gaps |
|--------|-------|-----------|----------|---------------|
| `config.py` | 101 | indirect | 70% | No config overlay tests |
| `tasks/stm.py` | 61 | `test_tasks.py` (10 tests) | 90% | Good |
| `tasks/temporal_parity.py` | 56 | `test_tasks.py` (6 tests) | 90% | Good |
| `tasks/narma10.py` | 67 | `test_tasks.py` (6 tests) | 85% | Recurrence edge cases |
| `reservoirs/pennylane_qrc.py` | 176 | `test_smoke_qrc.py` (11 tests) | 60% | Missing: z_and_zz dimension, circuit structure validation, backend switching |
| `reservoirs/ablations.py` | 160 | **None** | **30%** | Only exercised via CLI; no unit tests per ablation |
| `baselines/esn.py` | 193 | **None!** | **0%** | **Critical**: no tests for grid search, train_predict |
| `baselines/random_features.py` | 106 | **None** | **0%** | **Critical**: no unit tests |
| `metrics/scoring.py` | 126 | `test_metrics.py` (17 tests) | 95% | Excellent |
| `metrics/stats.py` | 126 | `test_metrics.py` (7 tests) | 70% | Missing: BCa bootstrap, power analysis |
| `proof/run_manifest.py` | 343 | `test_proof_layer.py` (6 tests) | 40% | Tests only CSV structure; not field population logic |
| `cli.py` | 745 | **None** (integration only) | **10%** | **Critical**: No tests for command handlers, gate logic, error paths |
| `viz/plots.py` | 119 | None | 0% | Low priority (Phase 2) |

**Overall Coverage:** ~45% estimated. **Critical gaps:** ESN baseline tests, CLI command tests, gate evaluator tests, ablation tests.

### Reliability Weaknesses (Top 10 Failure Modes)

| # | Failure Mode | Current Behavior | Severity | Mitigation |
|---|--------------|------------------|----------|------------|
| F1 | `runs.csv` corruption from concurrent writes | No locking; interleaved writes | **High** | SQLite WAL or file lock; per-worker temp + merge |
| F2 | Transient BLAS/LAPACK `LinAlgError` | Uncaught → run fails | Medium | Retry wrapper (3 attempts, exponential backoff) |
| F3 | Out-of-memory for large T | No pre-check | Medium | Validate `T*F*8 < 2GB`; refuse or stream |
| F4 | ESN grid search all folds fail | Raises `RuntimeError` with "no valid" | Medium | Log each fold's exception; provide diagnostics |
| F5 | Manifest partial write (disk full) | OS error propagates | Medium | Atomic write via tempfile + rename (already used for compute tracker); apply to CSV append |
| F6 | Gate evaluation with insufficient seeds | Returns INSUFFICIENT_EVIDENCE; CI continues | Medium | `--require-seeds N` flag; fail gate if not met |
| F7 | Config validation error | Pydantic `ValidationError` multi-line | Low | Pretty-print errors with `rich`; suggest fixes |
| F8 | Qiskit/PennyLane mismatch → cross-check fail | Tolerance 1e-6 hardcoded | Medium | Make tolerance configurable; log actual max diff |
| F9 | Health check false negative (optional dep missing) | All 16 packages required | Low | Split required vs optional; optional warn not fail |
| F10 | No progress feedback on long sweeps | Silent until completion | Low | Add `--verbose` progress bar (`tqdm`) |

### Gate Testing Strategy (Critical P1)

**Problem:** Gate evaluators untested; stubs return `INSUFFICIENT_EVIDENCE`.

**Solution:** Synthetic `runs.csv` fixtures.

```python
# tests/fixtures/gates.py
def make_runs_df(rows: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[
        'task_name', 'success', 'primary_metric_name',
        'primary_metric_value', 'run_id'
    ])

def test_g1_pass_minimum():
    df = make_runs_df([
        {'task_name': 'stm', 'primary_metric_name': 'mc', 'value': 1.2, ...},
        ...  # 5 rows
        {'task_name': 'ablation:no_entangle', 'primary_metric_name': 'mc', 'value': 0.9, ...},
    ])
    result = gates.g1.evaluate(df, config_with_thresholds)
    assert result.verdict == 'PASS'
```

Create fixtures for: G1 PASS/FAIL/INSUFFICIENT, G2 PASS/FAIL/INSUFFICIENT, G2.5 PASS/FAIL, G3 with Holm-Bonferroni, G5 cross-check residuals.

**Integration Test:** Full `runs.csv` with 20 rows (5 seeds × 4 conditions); run `qrc-thresher gate G1` → assert exit code 0 and `gates/G1.json` verdict PASS.

---

## 4. TECHNICAL ARCHITECT VISION — MODULAR AUGMENTATION

### Target Architecture (Phase 2 Complete)

```
                    ┌─────────────────────────────────────────┐
                    │          CLI Layer (click)               │
                    │  health │ run │ ablation │ gate │ query │
                    └───────────────┬─────────────────┬────────┘
                                    │                 │
                    ┌───────────────▼─────────┐  ┌─────▼───────┐
                    │  Orchestrator v2 (DAG)  │  │ Config v2   │
                    │  • Caching (CAS)        │  │ • Overlays  │
                    │  • Parallel exec        │  │ • Sweeps    │
                    │  • Resumability         │  │ • Diff tool │
                    └───────────────┬─────────┘  └─────────────┘
                                    │
                    ┌───────────────▼───────────────────────────┐
                    │      Plugin Registry (entry-points)        │
                    │  TaskRegistry  ReservoirRegistry  ...      │
                    └───────────────┬────────────────────────────┘
                                    │
        ┌───────────────────────────┴──────────────┐
        │   Plugin Implementations (discoverable)    │
        │   tasks/    reservoirs/   baselines/      │
        │   gates/    viz/         proof/           │
        └───────────────────────────────────────────┘
```

### 1. DAG Orchestrator (Replaces Sequential Pipeline)

```python
class PipelineOrchestrator:
    def __init__(self, config, max_workers=4):
        self.dag = self._build_dag()  # nodes: task, reservoir, features, train, eval, manifest
        self.cache = ContentAddressableCache()
        self.executor = ProcessPoolExecutor(max_workers=max_workers)
    
    def execute(self, seeds: List[int]) -> List[RunManifest]:
        # Topological execution with caching; parallel independent branches
        ...
```
**Benefits:** Concurrent seeds (speedup ×n_workers); cached intermediate results; resumable; DAG visualization.

### 2. Plugin Registry via Entry Points

```toml
[project.entry-points."qrc_thresher.tasks"]
stm = "qrc_thresher.tasks.stm:generate_stm"
parity = "qrc_thresher.tasks.temporal_parity:generate_parity"

[project.entry-points."qrc_thresher.reservoirs"]
pennylane = "qrc_thresher.reservoirs.pennylane_qrc:extract_features"
```
Registry auto-discovers; third-party plugins installable via pip without core fork.

### 3. Configuration v2: Layered Overlays + Parametric Sweeps

```yaml
# configs/alpha_lite_base.yaml (frozen)
experiment_name: alpha_lite_v1
...

# configs/experiments/n_qubits_sweep.yaml (overlay)
_overlay: true
inherits: alpha_lite_base.yaml
sweeps:
  reservoir.n_qubits: [4, 6, 8, 10]
  seeds.n_seeds: 5
```
CLI: `qrc-thresher run stm --config base.yaml --overlay sweep.yaml`

### 4. Results Database (SQLite) — Replace Flat CSV

```python
class ExperimentDB:
    SCHEMA = """
    CREATE TABLE runs (
        run_id TEXT PRIMARY KEY,
        gates_passed JSON,
        config JSON,
        manifest JSON,
        ...
    );
    CREATE INDEX idx_task_success ON runs(task_name, success);
    """
```
Dual-write: DB primary, CSV export for backward compatibility.

### 5. Gate Evaluator Framework (Modular, Testable)

```python
class GateEvaluator(Protocol):
    name: str
    spec: Dict[str, Any]
    def evaluate(self, df: pd.DataFrame, db: ExperimentDB) -> GateResult: ...

# Entry-point registration
[project.entry-points."qrc_thresher.gates"]
G0 = "qrc_thresher.gates.g0:HealthGate"
G1 = "qrc_thresher.gates.g1:STMGate"
...
```

### 6. Content-Addressable Cache (CAS)

```
cache/
├── features/sha256(u+circuit_hash).npz
├── models/sha256(X+y+alphas).pkl
└── manifests/run_<uuid>.json
```
Cache key = `hash(serialized(inputs) + code_hash)`; invalidated on git commit change.

---

## 5. ENHANCEMENT ROADMAP (PHASED)

### PHASE 1.5 (8 weeks, ~3 person-weeks) — **Stabilization & Gate Completion**

**Theme:** Enable manuscript-track experiments with full gate evaluation

| # | Feature | Effort | Dependencies | Deliverable |
|---|---------|--------|--------------|-------------|
| 1 | **Lockfile + CI enforcement** | 4h | None | `uv.lock` committed; CI installs from lock |
| 2 | **Parallel execution engine** | 12h | None | `--workers N` flag; concurrent seed runs; atomic CSV merge |
| 3 | **Gate evaluators G0.5, G3, G4, G5** | 16h | Complete artifact logging | All gates executable; CI passes without `continue-on-error` |
| 4 | **Artifact lineage logging** | 8h | Gate framework | `results/artifacts/features/<run_id>.npz`, `models/<run_id>.pkl` |
| 5 | **Ablation metric fix** | 2h | None | Parity ablation logs `accuracy` not `mc` |
| 6 | **Config overlay system** | 8h | Config v2 schema | `load_config(path, overlays=[])` API |
| 7 | **NARMA-10 integration** | 4h | Gate G3 pass check | `qrc-thresher run narma` works after G3 pass |
| 8 | **Holm-Bonferroni correction** | 6h | None | `stats.holm_bonferroni()`; integrated into G3/G4 |
| 9 | **Effect size thresholds in config** | 4h | Config schema extend | `gates:` block; gate evaluators read from config |
| 10 | **Results SQLite DB** | 8h | Parallel engine | `ExperimentDB()` with `query()`; CSV export maintained |
| 11 | **CI matrix expansion** | 2h | None | Python 3.12, 3.13 added; all green |
| 12 | **Gate evaluator unit tests** | 12h | Gate implementations | Synthetic `runs.csv` fixtures; 100% gate logic coverage |

**Total Phase 1.5:** ~80 person-hours ≈ **3 weeks** (single engineer)

**Milestone:** All gates G0–G5 pass on manuscript-track config (`n_seeds ≥ 5`). CI green without error suppression.

---

### PHASE 2.0 (+2–4 months, ~6 person-weeks) — **Extensibility & Advanced Analysis**

**Theme:** Plugin architecture, advanced statistics, hardware readiness

| # | Feature | Effort | Dependencies | Deliverable |
|---|---------|--------|--------------|-------------|
| 1 | **Plugin SDK & entry-points** | 1w | Phase 1.5 complete | `qrc_thresher.plugins` ABCs; example external plugin |
| 2 | **Stateful reservoir support** | 2w | DAG cache redesign | Carry quantum state across timesteps; cache intermediate states |
| 3 | **Advanced gate suite G6–G7** | 1.5w | Hardware backend stubs | G6: Phase-2 readiness checklist; G7: device calibration validator |
| 4 | **Noise model extensions** | 1w | Qiskit Aer integration | Shot-noise sweep; T1/T2 decay; readout error profiles |
| 5 | **BCa bootstrap & power analysis** | 1w | scipy.stats bootstrap | `stats.bca_ci()`, `stats.power_analysis()` |
| 6 | **Performance profiling suite** | 0.5w | None | `qrc-thresher perf` command; benchmark results in `results/benchmarks/` |
| 7 | **Advanced visualization** | 1w | matplotlib enhancement | MC per-delay heatmap; gate decision trees; correlation matrices |
| 8 | **Sphinx API documentation** | 1w | Sphinx, sphinx-apidoc | Docs on GitHub Pages; auto-generated from docstrings |
| 9 | **Structured JSON logging** | 0.5w | structlog | Machine-parseable logs; `jq`-friendly |
| 10| **OpenTelemetry traces** | 1w | opentelemetry-api | Optional trace export to Jaeger |

**Total Phase 2.0:** ~6 person-weeks ≈ **1.5 months** (1 engineer) or **1 month** (2-engineer team)

**Milestone:** Phase 2 ready for hardware execution (IBM Quantum/AWS Braket). Plugin SDK published.

---

### FUTURE (Phase 3+, optional)

| Feature | Effort | Rationale |
|---------|--------|-----------|
| Web dashboard (FastAPI + SQLite) | 3–4w | Browse runs, compare gates, export reports |
| Distributed compute (Dask/Celery + Redis) | 4–6w | Scale to 1000s of seeds; cost-tracking |
| Artifact server (S3/MinIO) | 2w | Offload figures, feature matrices; `manifest.artifact_url` |
| Notebook automation (Papermill) | 1w | One-click manuscript generation |
| Live hardware backends (IBM/Braket/Azure) | 8w+ | NISQ execution with calibration ingestion |

---

## 6. IMPLEMENTATION ROADMAP (ACTIONABLE)

### Week 1–2: Foundation Fixes (P0)

- [ ] Generate `uv.lock`; commit; CI install from lock
- [ ] Remove `continue-on-error` from CI health/G0; fix underlying import issues
- [ ] Fix package install: pin `reservoirpy==0.3.11` (not `>=0.3.11`)
- [ ] Fix parity ablation metric: use `classification_accuracy` when task is parity
- [ ] Add `gates:` threshold block to `configs/alpha_lite.yaml` and extend `GateConfig` in `config.py`
- [ ] Implement `holms_bonferroni()` in `stats.py`; add unit tests
- [ ] Write gate evaluator unit tests (G1, G2 synthetic fixtures) **before** implementing gates

### Week 3–4: Gate Completion

- [ ] Implement **G0.5**: PennyLane vs Qiskit cross-check using existing `verify_crosscheck()`
- [ ] Implement **G3**: QRC vs best ESN (parameter-matched) with Holm-Bonferroni correction
- [ ] Implement **G4**: NARMA-10 NRMSE vs ESN; integrate NARMA run/ablation paths
- [ ] Implement **G5**: Full-circuit cross-check across gated cells
- [ ] Implement gate verdict timestamping + `.latest.json` symlink
- [ ] Verify all gates pass on synthetic data (known fixtures)

### Week 5–6: Parallelization & DB

- [ ] Implement `ParallelRunner` with `ProcessPoolExecutor` and progress bar (`tqdm`)
- [ ] Add `--workers N` CLI flag; default 1 (backward-compatible)
- [ ] Implement SQLite `ExperimentDB` with WAL mode; dual-write to CSV
- [ ] Add atomic CSV append with file lock as fallback
- [ ] Add `qrc-thresher query "SELECT ..."` command

### Week 7–8: Modernization Prep

- [ ] Refactor CLI: extract command handlers to `commands/` package (thin dispatcher remains)
- [ ] Introduce `TaskRegistry` and `ReservoirRegistry` (initially manual registration; entry-points in Phase 2)
- [ ] Implement config overlay system (`load_config(path, overlays=[])`)
- [ ] Implement `qrc-thresher graph --dot` DAG visualization
- [ ] End-to-end test: full sweep with 5 seeds, 3 RRs, parallel=4, verify manifest + gate pass

**Phase 1.5 Complete:** All gates operational, parallel execution, reproducible lockfile, CI green without error suppression.

---

## 7. SUCCESS METRICS

### Phase 1.5 Completion Criteria

1. ✅ **All 5 health checks** (`qrc-thresher health`) PASS on clean checkout
2. ✅ **All gates G0–G5** evaluate autonomously on manuscript-track data (`n_seeds ≥ 5`)
3. ✅ **Parallel execution** achieves ≥3× speedup on 4 workers vs serial
4. ✅ **Dependency lockfile** ensures bit-identical environment recreation (`uv sync --frozen`)
5. ✅ **Gate evaluator tests** cover 100% of decision logic with synthetic fixtures
6. ✅ **CI pipeline** runs full gate suite; no `continue-on-error`
7. ✅ **Manifest immutability** enforced: post-gate config changes detected and rejected
8. ✅ **Multiple comparison correction** applied in G3/G4; p-values corrected <0.05 threshold

### Phase 2.0 Completion Criteria

1. ✅ **Plugin SDK published** — external plugin example passes CI
2. ✅ **Stateful reservoirs** benchmarked on STM with carry-depth study
3. ✅ **Hardware backends** connected (IBM Quantum, AWS Braket) with calibration data ingest
4. ✅ **Advanced gates** G6/G7 implemented and documented
5. ✅ **Results DB** query API used by dashboard prototype
6. ✅ **Documentation** auto-generated and hosted on GitHub Pages
7. ✅ **Performance baseline** established; no regression >10% on Alpha-Lite config

---

## 8. RISK REGISTER & MITIGATION

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Phase 1 runs become irreproducible after refactor | Medium | **Critical** | Dual-run period: old + new code for 3 seeds; compare manifests; gates based on original until all pass |
| Parallelization introduces race conditions | Medium | High | SQLite WAL mode + file lock; extensive concurrency tests |
| Plugin system introduces circular imports | Medium | High | Lazy loading in registry: `importlib.import_module()` on first `get()` |
| Cache poisoning (malicious precomputed X) | Low | High | Cache keys include code hash; only trust if git commit matches; `--no-cache` flag |
| Database corruption during concurrent writes | Medium | Medium | WAL mode; one writer at a time; backup runs.csv before migration |
| Config overlay merge bugs | Medium | Medium | Property tests: random overlay combos == manual deep merge |
| Performance regression from DAG overhead | Low | Low | Benchmark DAG executor vs serial; overhead <5% target; cache hit >80% |
| **Dependency drift** (observed already) | **High** | **High** | **Lockfile enforcement; CI fails on `pip check`** |
| Python 3.14+ breaking API | Medium | Medium | CI matrix 3.11–3.13; track EOL; pin 3.11 for Phase 1 |

---

## 9. MIGRATION PATH (ZERO DISRUPTION)

**Strategy:** Flag-driven dual runtime with gradual cutover.

**Week 1 — Compatibility Layer:**
```python
# qrc_thresher/v1_compat.py
def run_legacy(task, config):
    import subprocess
    subprocess.run([sys.executable, "-m", "qrc_thresher.cli", "run", task, "--config", config])
```
Old CLI unchanged.

**Week 2–3 — Core Extraction:**
- Move `run_cmd` internals to `qrc_thresher.core.run_experiment()` (pure, testable)
- Old CLI calls `core.run_experiment()`
- New orchestrator also calls it
- Validate: `python -c "from qrc_thresher.core import run_experiment"` works

**Week 4 — Deploy New Orchestrator:**
- Add `qrc-thresher2` entry point (parallel CLI) alongside `qrc-thresher`
- Default `qrc-thresher` continues legacy; docs recommend `qrc-thresher2` for new sweeps
- Both write to same `results/runs.csv` (concurrent writes protected)

**Week 5–6 — Feature Parity:**
- Port commands (health, run, ablation, gate, plot, summary) to new engine
- Deprecation warnings on old commands: "use `qrc-thresher2 run` for parallel execution"
- Benchmark: new engine 2× faster on 4 seeds, 5× on 16 seeds

**Week 7 — Switch Default:**
- Rename: `qrc-thresher2` → `qrc-thresher`; old becomes `qrc-thresher-legacy`
- Deprecation period: 90 days before removal
- Update CI to use new engine; health check passes without error suppression

**Week 10 — Remove Legacy:**
- Delete monolithic `cli.py`; all logic in `commands/` + `orchestrator.py`
- Remove `v1_compat` shim
- Release **v1.0.0** (first stable)

**Rollback:** Tag `legacy/cli` before each step; `git checkout` to revert.

---

## 10. CONTACT & OWNERSHIP

**Phase 1.5 Lead:** 1 engineer (3 weeks full-time)  
**Phase 2.0 Lead:** 1–2 engineers (6 weeks)  
**Review Cadence:** Weekly architecture review; gate decisions documented in `docs/DECISIONS.md`

**Key Decisions Required (Week 1):**
1. **Lockfile tool:** `uv` (preferred) vs `pip-compile` (fallback) — decide and standardize
2. **Parallel backend:** `concurrent.futures.ProcessPoolExecutor` (simplest) vs `joblib` (richer) vs `dask` (overkill)
3. **Cache backend:** Filesystem CAS vs SQLite BLOB vs Redis (future) — start with filesystem for Phase 1.5
4. **Gate pre-registration:** Config schema location — add to `GateConfig` or separate `gates.yaml`?
5. **CI cost budget:** Nightly sweeps allowed? Set monthly compute cap ($50 per BUILD_SPEC §3.C5)

---

**APPENDIX:** For complete technical specifications, see:
- `docs/BUILD_SPEC.md` (canonical engineering spec)
- `docs/METHODOLOGY.md` (statistical methodology)
- `docs/DECISIONS.md` (ADR log, append-only)

*This master plan synthesizes four expert perspectives into a coherent, risk-managed augmentation path. Implement Phase 1.5 in priority order; gate completion is the critical path to publication.*
