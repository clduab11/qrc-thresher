# BUILD_SPEC.md — QRC-Thresher (Praxen Labs QRC-001)

**Status:** Canonical engineering specification. Phase 1 (Alpha-Lite).
**Owner:** Praxen Labs · QRC research line.
**Repository:** `cld-maindev/qrc-thresher`.
**Document license:** CC BY 4.0 (see `docs/LICENSE-DOCS`).
**Code license:** Apache-2.0 (see `LICENSE`).
**Synthetic data license:** CC0 1.0 (see `data/LICENSE-DATA` if/when present).
**Supersedes:** the in-repo Copilot jumpstart prompt and the prior 75-line `BUILD_SPEC.md` stub.
**Authority:** Where this document and any other in-repo document disagree, **this document wins**, with two exceptions: `docs/REFERENCES.md` is authoritative for the literature bar, and `docs/DECISIONS.md` is authoritative for past architectural decisions (ADRs).
**Reading time:** ~50 minutes for a reviewer; ~2–3 hours for an engineer reproducing the harness end-to-end.
**Intended audience:** (1) the engineering team executing Phase 1; (2) an external reviewer at a NeurIPS QML workshop or *Quantum* journal who wants to reproduce the experiments; (3) future maintainers of subsequent phases.
**Conventions:** All sections use the keywords MUST, SHOULD, MAY in the RFC 2119 sense. Anything labeled `ASSUMED-DEFAULT` is a defensible default chosen in the absence of an explicit upstream decision; Appendix A consolidates every such default for review.

---

## Table of Contents

1. Mission and Scientific Question
2. Falsification-First Methodology
3. Hard Constraints (Honesty Bar, Compute Bar, Literature Bar)
4. Glossary and Notation
5. Repository Layout
6. Dependencies and Environment
7. Configuration Schema and Defaults
8. Tasks: Definitions and Generators
9. Quantum Reservoir Architecture
10. Ablation Suite (8 axes)
11. Classical Baselines (ESN, RKS, GRU stub)
12. Training, Validation, and Cross-Validation Protocol
13. Metrics and Their Estimators
14. Statistical Methodology
15. Decision Gates G0–G7
16. Proof Layer: Manifest Schema v1.1
17. CLI Surface
18. Health Checks (G0)
19. Reproducibility Contract
20. Testing Strategy
21. Linting, Formatting, and CI
22. Phases and Roadmap
23. Compute Budget and Cost Model
24. Risks and Mitigations
25. Differentiation vs. Prior Art
26. Reporting Protocol (Figures, Tables, Manuscript)
27. License, Data Governance, and Attribution
28. Methodology Gap Register (MG1–MG10)
29. Appendix A — Assumed Defaults Index
30. Appendix B — Example Configuration
31. Appendix C — Schemas (run.csv, gate.json, manifest.json)
32. Appendix D — Acceptance Checklist for Phase 1

---

## 1. Mission and Scientific Question

### 1.1 One-sentence mission

QRC-Thresher exists to determine, **with statistical rigor and a paranoid proof layer**, whether quantum reservoir computing (QRC) implemented as a *circuit-level pattern over hardware-efficient ansätze* yields a measurable, falsifiable separation over equally-tuned classical reservoir baselines on canonical low-dimensional time-series tasks (STM, temporal parity, NARMA-10), at qubit counts that are achievable on present-day NISQ hardware (n ∈ {4, 6, 8, 10, 12}) and at noise levels that are actually observed on those devices.

### 1.2 Why this question, why now

Two observations motivate the project:

1. The QRC literature is dominated by *implementations* of specific physical Hamiltonians (transverse-field Ising, Hubbard, photonic networks). These implementations conflate the **circuit pattern** (entangling structure, input-encoding map, measurement basis) with **device idiosyncrasies** (native gate set, coherence times, crosstalk, sampling overhead). A reviewer cannot tell, from existing reports, whether observed gains come from the reservoir abstraction or from device-specific physics.
2. Classical reservoirs (echo-state networks; ESN) and random-kitchen-sink (RKS) feature maps are *extremely* strong baselines on the canonical benchmark set. Most published QRC vs. classical comparisons under-tune the classical side. The honest question is therefore not "does QRC beat ESN?", but "*at matched memory budget, matched parameter budget, matched evaluation budget, and with statistical correction for multiple comparisons,* does QRC produce a separation from ESN by an effect size that survives noise injection at present-day hardware levels?"

QRC-Thresher answers this question for one specific instantiation of the QRC concept: a hardware-efficient layered ansatz with parameterized rotations and ring-topology entanglement, simulated on a state-vector backend, with optional sampled-shot and depolarizing-channel noise, and read out via a linear ridge regressor on Pauli-string expectation values. We call this instantiation the **circuit-level pattern**, and we treat it as a *thresher* — a cheap, brutal sieve that lets us discard or retain the QRC hypothesis without committing to a physical platform.

### 1.3 What "separation" means in this document

We deliberately avoid the marketing term that begins with "advantage" and ends with "supremacy"; this document uses the neutral term **separation** in a strict, frequentist sense. A claim of separation requires *all* of:

- A point estimate of the chosen metric on the QRC system that exceeds the *best-tuned* classical baseline by a pre-registered effect size δ (per task; see §13).
- A 95% bias-corrected and accelerated (BCa) bootstrap confidence interval on the *paired difference* whose lower bound exceeds zero after Holm–Bonferroni correction across all tasks reported in the same table.
- Robustness across at least five random seeds, three reservoir realizations, and two noise levels (shot-noise + a depolarizing-channel proxy).
- A signed manifest (§16) that ties the result to an exact code commit, an exact configuration file hash, an exact dataset hash, and the runtime environment.

Anything weaker is reported as **inconclusive** or **negative**.

### 1.4 Non-goals

QRC-Thresher does **not**:

- Make universal claims about quantum computation being categorically faster than classical computation.
- Claim cryptographic, optimization, or chemistry separation.
- Train the quantum circuit's parameters end-to-end. Circuit parameters (rotation angles, entangler placement) are sampled once per *reservoir realization* and frozen; only the linear readout is trained. This is the defining feature of *reservoir* computing and the reason inference is cheap.
- Run on physical quantum hardware in Phase 1. Hardware is a Phase 2 question, conditional on Phase 1 results crossing G5.
- Investigate alternative ansätze (e.g., Trotterized Hamiltonian dynamics, brick-wall random circuits) in Phase 1. Those are the explicit subject of Phase 2.
- Use deep-learning frameworks (`torch`, `tensorflow`, `jax`) or any LLM/transformer dependency in Phase 1.

---

## 2. Falsification-First Methodology

### 2.1 The methodological stance

This project is run under a **falsification-first** discipline. The default outcome of every experiment is "no separation." The harness is engineered so that the *easiest* outcome to produce, and the cheapest outcome to defend, is a negative result. Positive results require all of the additional machinery in §13–§16.

This is the inverse of the usual ML-paper discipline (where the default is to report the best run after a hyperparameter sweep and to underplay the baselines). The reason is selection effects: in a community where a large majority of QRC reports are positive, the prior odds that a randomly chosen positive QRC result is *real* are poor. We therefore design the harness to:

- Pre-register the metrics, effect sizes, gates, and statistical tests **before** running the corresponding sweep.
- Treat every sweep run as either *gate-passing* or *gate-failing*; a failed gate halts publication-track work on that branch and forces an explicit decision-log entry in `docs/DECISIONS.md`.
- Refuse to publish, share externally, or cite any QRC result that has not passed the full gate stack (G0–G5 for Phase 1).

### 2.2 What "Phase 1" closes

Phase 1 closes a single, narrow question: *On STM, parity, and NARMA-10, at n ∈ {4, 6, 8, 10} qubits and at the configured shot-noise + depolarizing levels, does the circuit-level QRC pattern produce a non-trivial paired separation over a memory-, parameter-, and evaluation-matched ESN, with a confidence interval that survives correction for multiple comparisons?*

If the answer is **yes** with the predefined effect size and confidence, Phase 1.5 (extended ablations) and Phase 2 (alternative ansätze) are unblocked.
If the answer is **no**, Phase 1 produces a publishable *negative* result with full proof manifest, and the QRC line at Praxen Labs is reconsidered.
If the answer is **inconclusive** (CI brackets zero), Phase 1.5 narrows the variance budget by adding seeds and reservoir realizations rather than by adding new comparisons.

### 2.3 Why negative results are a deliverable

The methodology IS the contribution. A signed, reproducible negative result on three canonical tasks across the (n, L) grid, with a passing ablation discrimination set, is publishable and useful: it raises the bar for any subsequent QRC claim against the same tasks at the same scale. Phase 1's success criterion is **completing the gate stack**, not **the verdict at the end**.

### 2.4 What we are explicitly *not* doing

- Chasing benchmarks beyond STM/parity/NARMA-10 in Phase 1. The selection of these three is dictated by their (a) being canonical in the reservoir-computing literature, (b) admitting closed-form difficulty knobs (delay τ, parity window k, NARMA order), and (c) being cheap enough to sweep at n=10 inside a single workstation budget.
- Decoder architectures beyond *linear ridge regression* on the readout features. Nonlinear readouts (kernel ridge, MLPs) are deferred because they confound the reservoir's expressive power with the readout's expressive power.
- In-circuit parameter optimization (variational training). The reservoir is *frozen*. This is a deliberate scope cut.
- In Phase 1: cross-platform porting beyond a Qiskit↔PennyLane cross-check. PennyLane (`default.qubit`/`lightning.qubit`) is the primary backend, with Qiskit Aer used as the cross-check.

---

## 3. Hard Constraints

These constraints are non-negotiable in Phase 1. Violating any of them invalidates a result for publication-track use and MUST be flagged in the manifest as a non-conformance.

### 3.1 The Honesty Bar

H1. **No cherry-picking.** Every reported metric MUST be the mean (and per-protocol BCa CI) over all seeds in the pre-registered seed set. If a seed crashes, it is re-run from the same seed; if it crashes deterministically, the failure is reported in the manifest, and the seed is *replaced* via a documented procedure (see §19), not silently dropped.

H2. **No baseline starvation.** The classical baselines (ESN, RKS) MUST be tuned with at least the same compute budget as the QRC sweep on the same task. Tuning grids for both sides are pre-registered in the configuration file and hashed into the manifest. The ESN reservoir size MUST be matched to the QRC feature dimension, **never** to 2^n.

H3. **No metric shopping.** The primary metric per task is fixed (§13). Secondary metrics MAY be reported but MUST be flagged as secondary and MUST NOT be used to override the primary metric's verdict.

H4. **No unverifiable claims.** Every numeric claim in a derived artifact (figure, table, abstract) MUST be traceable, by seed and by configuration hash, to a row in `results/runs.csv` and to a manifest entry. The reporting tooling (`qrc-thresher plot`, `qrc-thresher summary`) enforces this by construction.

H5. **No verbal hedging.** A result is one of: `pass`, `fail`, `inconclusive`. Documents that describe results MUST use exactly these labels and MUST NOT use marketing language ("promising," "encouraging," "near-significant") in lieu of a gate verdict.

### 3.2 The Compute Bar

C1. **Workstation-first.** Phase 1 MUST be completable on a single workstation with 16 GB RAM, 8 logical cores, and no GPU, within 24 wall-clock hours per full sweep at n ≤ 10 and shots ≤ 8192. If the budget is exceeded, the qubit ceiling is lowered or the shot count reduced (with corresponding documentation in DECISIONS.md), not the seed count.

C2. **CPU-first.** No CUDA, cuQuantum, JAX-GPU, TPU, or any other accelerator path in Phase 1. The harness MUST run on a generic Linux/macOS laptop without device-specific drivers.

C3. **Cluster-optional, not cluster-required.** A cluster MAY be used to parallelize seeds, but the harness MUST NOT depend on cluster-only features. The default `qrc-thresher run` and `qrc-thresher ablation` paths use a local process pool with a configurable worker count.

C4. **No cloud quantum credits in Phase 1.** Cloud QPU credits are a Phase 2 line item.

C5. **Cost cap.** Total monetary spend for Phase 1 is bounded at $0–$50 (workstation electricity + optional CI runner credits). Any spend above $50 requires an ADR.

### 3.3 The Literature Bar

L1. **Canonical-only references in Phase 1 reporting.** The references list (`docs/REFERENCES.md`) defines the *Published Bar* (peer-reviewed or arXiv-permanent works) and a small *forbidden* set (results not reproducible from public artifacts, retracted, or whose claims have been refuted). Phase 1 manuscripts MUST cite only Published Bar entries.

L2. **Closest competitor enumerated.** Every Phase 1 manuscript MUST contain an explicit comparison subsection enumerating the *closest competitor* on each task and explaining what is held fixed (n, shots, noise model) and what is allowed to vary.

L3. **No unlicensed-code reuse.** Any number lifted from prior work MUST be marked as "external" in tables, with a citation. Internal numbers MUST be traceable to a manifest in this repository. Code from sources flagged in `docs/REFERENCES.md` as forbidden (e.g., unlicensed reservoir codebases) MUST NOT be imported.

### 3.4 The Scope Bar

S1. Synthetic data only in Phase 1; no scraped or licensed datasets.
S2. No web servers, REST APIs, GUIs, or notebook-as-product surfaces. CLI only.
S3. No transformer/LLM dependencies (`transformers`, `langchain`, `openai`, `anthropic`, `cohere`).
S4. Python ≥ 3.11 only.

---

## 4. Glossary and Notation

- **n** — number of qubits in the reservoir.
- **L** (or **d**) — number of layers (depth) in the hardware-efficient ansatz.
- **θ** — frozen rotation angles, sampled per reservoir realization from a fixed distribution (default `Uniform[0, 2π]`; ASSUMED-DEFAULT 4.A).
- **ϕ(x)** — feature map embedding the scalar input x_t into rotation angles. Default: scaled angle encoding `Ry(α · x_t)`, α = π (ASSUMED-DEFAULT 4.B), applied to *all* qubits per layer.
- **U_in(x_t), U_res(θ)** — input-encoding and reservoir unitaries, respectively.
- **|ψ_t⟩** — reservoir state at time t. Constructed time-locally as `|ψ_t⟩ = U(x_t, θ) |0⟩^n`. The "stateless" readout pattern; see §9.2 for the stateful variant deferred to Phase 1.5.
- **F_t** — feature vector at time t: a stacked vector of expectation values of designated Pauli observables on `|ψ_t⟩`.
- **W_out** — linear readout weights, fitted via ridge regression (§12).
- **Task** — one of `STM(τ)`, `parity(k)`, `NARMA(p)`.
- **Reservoir realization (RR)** — one sample of θ. Repeated `n_rr` times per (n, L, task) cell.
- **Seed** — RNG seed governing input sequence, train/val/test split, ridge tie-breaking, and shot sampling. Distinct from RR seed.
- **Run** — a single (config, seed, RR) triple producing one row in `results/runs.csv`.
- **Sweep** — a Cartesian product of runs over a configuration's varied axes.
- **Gate** — a binary verdict (`pass`/`fail`/`inconclusive`) computed from a sweep's `runs.csv` and a `gate.spec.json`.
- **Manifest** — a JSON object describing the inputs, outputs, environment, and gate verdicts of a sweep. Schema v1.1 (§16).
- **NMSE** — normalized mean-squared error: `mean((y - ŷ)^2) / var(y)`.
- **NRMSE** — normalized root mean-squared error: `sqrt(MSE) / std(y)`. The default error metric in this repo (matches `metrics/scoring.py`).
- **MC** — memory capacity in the Jaeger sense: `MC = Σ_τ corr(y_t, ŷ_t)^2` over `STM(τ)` for τ ∈ {0, …, τ_max}.
- **HEA** — hardware-efficient ansatz; the family of circuits used in Phase 1.
- **MG** — methodology gap; tracked in the §28 register MG1–MG10.

ASSUMED-DEFAULT (4.C): Internal notation is consistent with `docs/METHODOLOGY.md`. Where the manuscript adopts standard literature notation at publication time, the harness MUST keep these identifiers stable so that grep across logs and configs is unambiguous.

---

## 5. Repository Layout

### 5.1 Current state (Phase 1.0 scaffold)

```
qrc-thresher/
├── README.md                      # Project preamble, status, license
├── LICENSE                        # Apache-2.0 for code
├── pyproject.toml                 # Package metadata, deps, ruff/pytest config
├── Makefile                       # Convenience commands
├── configs/
│   └── alpha_lite.yaml            # Phase 1 default sweep config
├── docs/
│   ├── BUILD_SPEC.md              # THIS DOCUMENT (canonical)
│   ├── METHODOLOGY.md             # Companion: long-form derivations
│   ├── REFERENCES.md              # Literature bar
│   ├── DECISIONS.md               # ADRs (D001 onward)
│   └── LICENSE-DOCS               # CC BY 4.0
├── src/qrc_thresher/
│   ├── __init__.py
│   ├── cli.py                     # Click-based CLI entry point
│   ├── config.py                  # Pydantic v2 config models
│   ├── tasks/                     # stm.py, temporal_parity.py, narma10.py
│   ├── reservoirs/                # pennylane_qrc.py, qiskit_crosscheck.py, ablations.py
│   ├── baselines/                 # esn.py, random_features.py, gru.py (stub)
│   ├── metrics/                   # scoring.py, stats.py, runtime.py
│   ├── proof/                     # run_manifest.py, source_health.py, benchmark_health.py
│   └── viz/                       # plots.py
├── data/                          # Synthetic-only outputs (CC0 1.0)
├── results/                       # Run artifacts (gitignored except .gitkeep + spec)
└── tests/
    ├── conftest.py
    ├── test_smoke_qrc.py
    ├── test_metrics.py
    ├── test_tasks.py
    ├── test_reproducibility.py
    └── test_proof_layer.py
```

### 5.2 Target state at end of Phase 1

The above layout is preserved verbatim. Phase 1 introduces:

- `results/runs.csv`: append-only manifest log (schema v1.1).
- `results/cumulative_compute.json`: atomic compute counter.
- `results/gates/<name>.json`: gate verdict files.
- `results/figures/<run_id>/`: per-run figure outputs.
- `results/summaries/<phase>_summary.md`: aggregated reports.

### 5.3 Forbidden directories and files

- No top-level `data/` containing third-party datasets. Phase 1 tasks are synthetic and generated from seeds.
- No `weights/` directory. Reservoir parameters are reconstructed from seeds; readout weights live inside `results/<run_id>/` if needed.
- No notebooks as a *source* for any published number. Notebooks are triage-only and MUST NOT be the source-of-record.

ASSUMED-DEFAULT (5.A): The `results/` directory is partially git-tracked (small JSON/CSV artifacts) and lifecycle-managed by an external rsync/object-store policy for figures (see §19). Tag-pinned manifests are the artifact of record.

---

## 6. Dependencies and Environment

### 6.1 Python and packaging

- Python ≥ 3.11. Phase 1 SHOULD work on 3.11–3.13; 3.11 is the reference.
- Build backend: setuptools via `pyproject.toml` (current repo configuration).
- Dependency management: pinned major-version ranges in `pyproject.toml`; exact lock via `pip-compile` or `uv` produced lockfile (ASSUMED-DEFAULT 6.A: `pip` is the floor; `uv` is preferred for speed).

### 6.2 Runtime dependencies (current `pyproject.toml`)

- `pennylane>=0.44,<0.45`, `pennylane-lightning`
- `qiskit[all]>=2.3,<3`, `qiskit-aer`
- `scikit-learn>=1.4`
- `numpy>=1.26`, `scipy>=1.11`, `pandas>=2.1`
- `matplotlib>=3.8`
- `pydantic>=2.6`
- `pyyaml`
- `pytest>=8.0`
- `ruff>=0.4`
- `reservoirpy>=0.3.11`
- `click>=8.1`

### 6.3 Excluded dependencies (Phase 1)

- No `torch`, `tensorflow`, `jax` (deep-learning frameworks).
- No `cuda`, `cupy`, `nvidia-*` (GPU stack).
- No `transformers`, `langchain`, `openai`, `anthropic`, `cohere` (LLM stack).
- No `seaborn`, `plotly`, `bokeh` (matplotlib only).

### 6.4 Environment reproducibility

The manifest captures: Python version, OS, CPU model, total RAM, NumPy BLAS provider (when available), every dependency's installed version, and the SHA-256 of the lockfile. Determinism is enforced via explicit RNGs (`numpy.random.Generator`, `random.Random`) seeded per run. Implicit globals (e.g., `numpy.random.seed`) are forbidden by lint rule (ASSUMED-DEFAULT 6.B: ruff `NPY002` enabled).

---

## 7. Configuration Schema and Defaults

### 7.1 Format

YAML, validated by a `pydantic` v2 model at load time (`src/qrc_thresher/config.py`). Unknown keys are an error. Type errors are an error. Missing required keys are an error.

### 7.2 Top-level schema (current)

```yaml
experiment_name: <str>
task:
  name: <"stm" | "parity" | "narma">
  length: <int, ge=100>
  train_frac: <float, 0<x<1>
  delay_max: <int | null>
  parity_window: <int | null>
reservoir:
  backend: <"default.qubit" | "lightning.qubit">
  n_qubits: <int, 2..12>
  depth: <int, 1..10>
  readout: <"z_only" | "z_and_zz">
baseline:
  enabled: [<"esn" | "random_features" | "gru">, ...]
  esn_grid: { spectral_radius, input_scaling, leak_rate, ridge_alpha }
  rks_dim: <int | null>
ablation:
  name: <"phase_random" | "no_entangle" | "random_features" | "haar"> # optional
training:
  ridge_alphas: [<float>, ...]
  cv_folds: <int, 2..10>
proof:
  log_entanglement: <bool>
  log_circuit_hash: <bool>
seeds:
  task_seed: <int>
  reservoir_seed: <int>
  n_seeds: <int, 1..20>
```

### 7.3 Default config (`configs/alpha_lite.yaml`)

Phase 1's default sweep is intentionally lean (workstation-friendly):

- `reservoir.n_qubits: 4`, `reservoir.depth: 3`, `reservoir.readout: z_only`.
- `task.length: 500`, `task.train_frac: 0.7`, `task.delay_max: 20`, `task.parity_window: 3`.
- `baseline.enabled: [esn, random_features]`.
- `seeds.n_seeds: 3` for development; manuscript-track sweeps MUST use `n_seeds ≥ 5`.

### 7.4 Configuration immutability

A config that has been used to produce a *gated* manifest is immutable. Editing it in place is forbidden by policy and detected by manifest verification (§16). Variants are produced by *copying* and renaming.

---

## 8. Tasks

### 8.1 Short-Term Memory (STM)

**Definition.** Given an i.i.d. stream `u_t ∼ Uniform[-1, 1]`, the system is asked to output `y_t^{(k)} = u_{t-k}` for k ∈ {0, …, K}.

**Difficulty knob.** K ∈ {5, 10, 20}. Larger K requires longer effective memory.

**Primary metric.** Memory Capacity `MC = Σ_k corr(ŷ^{(k)}, y^{(k)})^2`. Secondary: per-k correlation² and per-k NRMSE.

**Sequence length.** Total T = 500 (default config); recommended T = 2000 for manuscript-track sweeps. ASSUMED-DEFAULT (8.A).

**Train/test split.** Chronological 70/30. No shuffling. Implementation: `src/qrc_thresher/tasks/stm.py`.

### 8.2 Temporal parity

**Definition.** Given `u_t ∈ {0, 1}` Bernoulli(0.5), output `y_t = XOR(u_{t-d+1}, …, u_t)` for window d.

**Difficulty knob.** d ∈ {1, 2, 3, 5}. Parity is famously hard for low-capacity learners; d ≥ 4 typically separates real reservoirs from linear models.

**Primary metric.** Accuracy on the test split (binarized at 0.5). Secondary: NRMSE on the {-1, +1} mapped target.

**Sequence length.** Same as STM. ASSUMED-DEFAULT (8.B). Implementation: `src/qrc_thresher/tasks/temporal_parity.py`.

### 8.3 NARMA-10

**Definition.** Recurrence `y_{t+1} = 0.3 y_t + 0.05 y_t (Σ_{i=0}^{9} y_{t-i}) + 1.5 u_{t-9} u_t + 0.1`, with `u_t ∼ Uniform[0, 0.5]`. The system predicts `y_t` from past `u`.

**Difficulty knob.** None in Phase 1; NARMA-10 is a fixed task. Phase 1.5 MAY add NARMA-20.

**Primary metric.** NRMSE.

**Phase gate.** NARMA-10 is gated behind G3; the CLI raises `NotImplementedError` if `task.name != "narma"` and `narma` is requested. Implementation: `src/qrc_thresher/tasks/narma10.py`.

### 8.4 Determinism contract for tasks

Each task generator is a pure function of `(seed, params, length)`. A task generator MUST NOT call any global RNG and MUST NOT depend on time. The harness verifies determinism by running each generator twice per build under `tests/test_tasks.py` (and `tests/test_reproducibility.py`) and asserting bit-exact equality.

### 8.5 Why these three tasks

- STM is *the* canonical memory probe for reservoirs.
- Parity is the canonical *nonlinearity-with-memory* probe (XOR over a sliding window).
- NARMA-10 is the canonical *generative-recurrence* probe and the task most often reported in the QRC literature.

We exclude *Mackey-Glass* and *Lorenz* in Phase 1 not because they are bad tasks but because they expand the manuscript scope without adding a comparison axis that STM/parity/NARMA cannot cover.

---

## 9. Quantum Reservoir Architecture

### 9.1 Circuit-level pattern

The *circuit-level pattern* is a parameterized family of layered quantum circuits with three mutable axes (see `src/qrc_thresher/reservoirs/pennylane_qrc.py`):

- **Input encoding** `U_in(u_t)`: a layer of single-qubit `Ry(π · u_t)` rotations on every qubit (ASSUMED-DEFAULT 9.A).
- **Reservoir unitary** `U_res(θ)`: an HEA composed of L layers, each containing (i) a single-qubit rotation block `Π_i Rz(θ^z_{i,l}) Rx(θ^x_{i,l})` and (ii) a ring-topology entangling block `Π_i CNOT(i, (i+1) mod n)` (ASSUMED-DEFAULT 9.B).
- **Readout** `F_t`: expectation values of single-qubit Pauli-Z operators on every qubit (`z_only`), optionally augmented with `Z_iZ_j` for all i<j (`z_and_zz`). Total feature size n (or n + n(n−1)/2). ASSUMED-DEFAULT (9.C).

### 9.2 Time-locality of the readout

Phase 1 uses a **stateless** readout: at each time t, the reservoir state is rebuilt from the input `u_t` alone, layered L times, and measured. The recurrence in the system comes from how the *readout layer* aggregates a window of features, not from accumulated quantum state across time. This is the cheapest, most analyzable variant. The **stateful** variant (carrying `|ψ_{t-1}⟩` forward) is deferred to Phase 1.5 and tracked in MG7.

### 9.3 Implementation backends

Two equivalent backends are maintained and cross-checked at every CI run:

1. **PennyLane** (`pennylane_qrc.py`). Primary backend. `default.qubit` for development, `lightning.qubit` for sweeps. Exact (no shot noise) by default.
2. **Qiskit Aer** (`qiskit_crosscheck.py`). Cross-check backend. Statevector simulator with optional sampled-shot and noise channel paths.

The cross-check (`tests/test_*` and the `qrc-thresher gate G5` evaluator) asserts that exact expectation values from the two backends agree to within 1e-6 per observable, for at least three random seeds across the (n, L) grid.

### 9.4 Noise model

In Phase 1, the noise model is a *deliberately minimal* two-channel model exposed via Qiskit Aer:

- **Per-CNOT depolarizing**: each two-qubit gate is followed by a depolarizing channel with probability `depolarizing_p`. Single-qubit gates are noiseless. ASSUMED-DEFAULT (9.D).
- **Symmetric readout flip**: each measured bit is flipped with probability `readout_error_p`.

Phase 2 introduces a *device-realistic* noise model (calibrated T1/T2, native gate set per platform). Phase 1 does not. See MG3.

### 9.5 Why HEA, not Trotterized Hamiltonian dynamics

We chose HEA in Phase 1 because:

- It has a small, finite parameter count per layer (2n rotation angles + n CNOTs for ring topology), which is easy to log, hash, and sweep.
- It is *platform-agnostic*: any gate-model device can run it with low overhead.
- It is the most pessimistic prior for QRC: HEA circuits are known to suffer from barren plateaus *under variational training*, but in reservoir mode they are not trained, so the barren-plateau objection does not apply, and HEA becomes a clean test of "is randomness over a generic universal family enough?"

If HEA fails our gates, *Trotterized Ising* and *random brick-wall* are the two Phase 2 alternatives, and they are explicitly named in `docs/DECISIONS.md` as such.

### 9.6 Circuit hash

Every reservoir realization produces a SHA-256 `circuit_hash` over the canonical serialization of `(n_qubits, depth, thetas.tobytes(), phis.tobytes(), readout)`. The hash is logged in every run manifest (§16). Two manifests with identical `circuit_hash` MUST produce identical features given identical input sequences (verified by `tests/test_reproducibility.py`).

---

## 10. Ablation Suite (8 axes)

The ablation suite (`src/qrc_thresher/reservoirs/ablations.py`) is the harness's main tool for ruling out *trivial* explanations of any apparent QRC separation. Every published positive result MUST be accompanied by the corresponding ablation table.

### 10.1 Ablation axes

A1. **Linearization (`no_entangle`)** — replace the entangling block with the identity. Each input qubit becomes a tiny independent quantum system; no qubit-to-qubit information mixing. Expected: catastrophic loss of capacity. *Failure mode tested:* "separation is a CNOT-counting artifact." Implemented in `extract_features_no_entangle`.

A2. **Phase-randomization (`phase_random`)** — draw fresh random `θ`, `ϕ` at each time step rather than fixing them per realization. Expected: collapse to random projection of `u_t` per step; no temporal structure. *Failure mode tested:* "separation is a random-feature artifact." Implemented in `extract_features_phase_random`.

A3. **Haar-random unitary (`haar`)** — sample one Haar-random `U ∼ Haar(2^n)` per realization, apply to angle-encoded state, measure `<Z_i>`. Expected: a strong-but-not-structured baseline. *Failure mode tested:* "separation comes from the circuit structure, not from access to a 2^n-dimensional unitary." Implemented in `extract_features_haar`. **Mandatory for G2.5.**

A4. **Random kitchen sinks (`random_features`)** — classical RKS at the same feature dimension. Tests whether classical random projection at matched dimension is enough. Implemented in `src/qrc_thresher/baselines/random_features.py`.

A5. **Observable swap** — replace `Z` with `X` in the readout (after a basis change). Expected: similar capacity up to a basis transformation. *Failure mode tested:* "basis-alignment artifact." A pass requires NRMSE within 10% of `Z`-readout. ASSUMED-DEFAULT (10.A). Tracked in MG5.

A6. **Layer scan** — vary L ∈ {1, 2, 3, 4}. Expected: monotone capacity gain up to a saturation. *Failure mode tested:* "harness has a layer-indexing bug."

A7. **Shot-count scan** — vary shots ∈ {1024, 4096, 16384, exact}. Expected: NRMSE converges to the exact-expectation value as shots increase. *Failure mode tested:* "result is shot-noise-driven."

A8. **Encoding swap** — replace `Ry(π · u_t)` with `Rz(π · u_t)` after a Hadamard. Information-theoretically equivalent on a noiseless device. *Failure mode tested:* "encoding-specific bug." Tracked in MG6.

### 10.2 Self-falsifying ablations

A1, A2, and A3 are *self-falsifying*: if any of them produces NRMSE/MC comparable to the full QRC (within 5% NRMSE or within 1 SE on MC) on the same task, the published positive result is automatically retracted. The harness enforces this by computing a `self_falsified` boolean in `gate.json` whenever any of A1–A3 is within tolerance of the full system on the same task; a `True` value forces G3 to fail.

### 10.3 Ablation reporting

Each ablation is run on **a single representative cell** per task (n=4, L=3, default config seed) unless the gate explicitly requests a wider sweep. The ablation table is published alongside the headline result and is required for G3.

---

## 11. Classical Baselines

### 11.1 Echo-state network (ESN)

`src/qrc_thresher/baselines/esn.py`. ReservoirPy-based ESN with:

- Reservoir size `N_ESN = N_quantum_features`, **never** `2^n`. For `z_only`, `N_ESN = n`. For `z_and_zz`, `N_ESN = n + n(n-1)/2`.
- Spectral radius ρ ∈ {0.8, 0.9, 0.95, 0.99, 1.0}.
- Input scaling ∈ {0.1, 0.5, 1.0}.
- Leak rate α ∈ {0.1, 0.3, 0.5, 1.0}.
- Ridge α ∈ {1e-8, 1e-6, 1e-4, 1e-2, 1.0}.
- CV: 5-fold on training data only. Test indices NEVER used for hyperparameter selection.

### 11.2 Random kitchen sinks (RKS)

`src/qrc_thresher/baselines/random_features.py`. Rahimi–Recht: `ϕ(u) = cos(W · u + b)`, `W ∼ N(0, σ²/d)`, `b ∼ Uniform[0, 2π]`. Dimension matched to QRC features.

### 11.3 GRU stub

`src/qrc_thresher/baselines/gru.py` is a stub that raises `NotImplementedError` until Phase 1 gates G1 and G2 pass. CPU-only. Not tuned in Phase 1.

### 11.4 Why these baselines

ESN and RKS are the two strongest *fair* baselines: both are linear-readout systems with random nonlinear features, exactly like QRC. A QRC that beats a tuned ESN/RKS at matched feature dimension is interesting; one that loses to a tuned ESN/RKS is the textbook negative result the harness is engineered to produce loudly.

---

## 12. Training, Validation, and Cross-Validation Protocol

### 12.1 Splits

Each task produces a single contiguous sequence per (seed, RR) pair. The sequence is split chronologically into:

- Train (70%): used to fit `W_out` and select α via `RidgeCV(alphas=ridge_alphas, cv=cv_folds)`.
- Test (30%): used *only* for final reporting. Never seen during fitting.

A leading 200-step warmup MAY be discarded for stateful variants (Phase 1.5+).

### 12.2 Matching modes

The honest QRC-vs.-ESN comparison requires careful matching. We define three matching modes:

- **Feature-matched** (default for Phase 1) — `N_ESN = N_quantum_features`. Reported as the headline.
- **Memory-matched** — choose `N_ESN` such that the ESN's empirical MC on STM equals the QRC's, within 10%. Reported in supplementary tables.
- **Parameter-matched** — choose `N_ESN` such that trainable readout parameter count matches.

### 12.3 Ridge regression

`sklearn.linear_model.RidgeCV` with `alphas=[1e-8, 1e-6, 1e-4, 1e-2, 1, 100]` and `cv_folds=5`. Closed-form solution; bias term included; standardization left to the caller.

### 12.4 Randomness handling

Two seeded `numpy.random.Generator` instances per run: `rng_task = default_rng(task_seed)` and `rng_reservoir = default_rng(reservoir_seed)`. Ablation paths derive a third generator as `default_rng(reservoir_seed + offset)` with documented offsets (`+1` for phase_random, `+2` for haar). ASSUMED-DEFAULT (12.A).

---

## 13. Metrics and Their Estimators

### 13.1 Primary metrics (per task)

- STM(K): **MC** (aggregate). Secondary: per-k correlation², NRMSE per k.
- parity(d): **accuracy**. Secondary: NRMSE on {-1, +1} mapped target.
- NARMA-10: **NRMSE**. Secondary: MAE.

Implementations: `src/qrc_thresher/metrics/scoring.py` (memory_capacity, nrmse, classification_accuracy, with NaN/inf guards).

### 13.2 Reporting form

For each (task, system) cell:

- Point estimate: mean over (seed × RR) cells.
- Uncertainty: 95% bootstrap CI (`metrics/stats.py:bootstrap_ci`, default 1000 resamples). BCa is supported via `scipy.stats.bootstrap` and SHOULD be preferred for manuscript-track reporting; the in-house percentile bootstrap is the floor. ASSUMED-DEFAULT (13.A).

### 13.3 Paired effect sizes

Comparisons are reported as the *paired* difference QRC − ESN (or QRC − RKS) per (seed, RR), with a 95% CI on the paired mean. Pairing is valid because input sequences and splits are identical across systems for a given (seed, RR).

### 13.4 Pre-registered effect sizes for separation claims

- STM-MC: ΔMC ≥ 0.5 over the τ ∈ {0..K} sum, with 95% CI lower bound > 0.
- parity(3): Δaccuracy ≥ 5 percentage points, with 95% CI lower bound > 0.
- NARMA-10: ΔNRMSE ≤ −0.05, with 95% CI upper bound < 0.

These thresholds are pre-registered in `configs/alpha_lite.yaml` under a `gates` block (added at first manuscript-track sweep). They MUST NOT be re-tuned after seeing the data.

### 13.5 Multiple-comparison correction

Holm–Bonferroni-corrected p-values across all primary task comparisons in the headline table. ASSUMED-DEFAULT (13.B): Holm–Bonferroni is preferred over BH-FDR because we test a small, fixed family of three null hypotheses.

### 13.6 NMSE vs NRMSE

`metrics/scoring.py:nrmse` returns `sqrt(MSE)/std(y)` — this is the project's primary error metric (units of standard deviation). NMSE (squared form) is reported in supplementary tables on request. The two are monotonically related and never used interchangeably in the same column.

---

## 14. Statistical Methodology

### 14.1 Bootstrap procedure

Default: percentile bootstrap with 1000 resamples (`metrics/stats.py:bootstrap_ci`). Manuscript-track: BCa via `scipy.stats.bootstrap` with 10000 resamples.

### 14.2 Paired tests

`metrics/stats.py:paired_test` returns a `PairedTestResult` with mean diff, std diff, t-statistic, p-value, and Cohen's d. `paired_test` is paired t-test (parametric); `wilcoxon_test` is the rank-based non-parametric companion. Both MUST be reported when sample sizes are small (n_seeds ≤ 5).

### 14.3 Power analysis (pre-registered)

Under an effect size of ΔMC = 0.5 and a between-cell SD of 0.4, 5 paired observations yield approximate power 0.55 for a one-sided test at α=0.05. Phase 1.5 expansion to 15 seeds raises power above 0.90. We pre-register these numbers so that an underpowered "fail to reject" cannot be reinterpreted as evidence for the null. See MG2.

### 14.4 Handling of NaN/inf

Any seed producing a NaN or inf in any metric is *flagged* and the entire seed is replaced by its successor in the spawning sequence. The replacement event is logged in the manifest. We refuse any policy that "winsorizes" or "imputes" metrics: a NaN is a bug, and the only valid responses are fix-and-rerun or replace-and-document. `metrics/scoring.py` raises `ValueError` on non-finite inputs by design.

### 14.5 Why frequentist, not Bayesian

A Bayesian framing would require explicit priors on per-task effect sizes, which we do not have. The frequentist BCa interval, paired across (seed, RR), with pre-registered effect sizes and Holm–Bonferroni correction, is the most defensible and most readable choice in the QRC vs. classical literature.

---

## 15. Decision Gates G0–G7

Gates are evaluated in order; a failed gate halts the publication path. Each gate consumes a `gate.spec.json` (declarative) or a hard-coded specification in `cli.py:gate_cmd`, plus the `results/runs.csv` log, and emits `results/gates/<name>.json`.

Every gate documents: **pass condition**, **fail condition**, **evaluation procedure**, and **artifact path**.

### 15.1 G0 — Environment and harness sanity

- **Pass:** All source-health and benchmark-health checks return PASS (Python ≥ 3.11; required packages importable; git rev-parse succeeds; numpy `default_rng` works; `alpha_lite.yaml` validates; STM/parity generators deterministic; QRC smoke 2-qubit circuit returns finite features; ESN smoke fits; metrics return finite; manifest writes; reproducibility check passes).
- **Fail:** Any check returns FAIL.
- **Procedure:** `qrc-thresher health` (CLI command).
- **Artifact:** `results/health/<timestamp>.json`.

### 15.2 G0.5 — Backend cross-check (PennyLane ↔ Qiskit)

- **Pass:** PennyLane and Qiskit produce identical `<Z_i>` expectation values within `1e-6` for at least three random `(n, L, seed)` triples on the cross-check circuit (`reservoirs/qiskit_crosscheck.py:verify_crosscheck`).
- **Fail:** Any pair exceeds tolerance.
- **Procedure:** `qrc-thresher gate G0.5` (consumes prior cross-check runs in `runs.csv`).
- **Artifact:** `results/gates/G0.5.json`.

### 15.3 G1 — STM separation

- **Pass:** QRC STM `MC > 1.0` AND QRC `MC` exceeds the entanglement-suppressed (A1) baseline by ≥ 20% on the same (seed, RR) pairs, across `n_seeds ≥ 5`.
- **Fail:** Either condition violated.
- **Inconclusive:** Insufficient seeds (`n_seeds < 5`) or the 20% margin's CI brackets zero.
- **Procedure:** `qrc-thresher gate G1`.
- **Artifact:** `results/gates/G1.json`.

### 15.4 G2 — Parity separation

- **Pass:** QRC parity accuracy > 70% at d=3 AND random-features (A4) accuracy < 60% on the same (seed, RR) pairs.
- **Fail:** Either condition violated.
- **Inconclusive:** Insufficient seeds.
- **Procedure:** `qrc-thresher gate G2`.
- **Artifact:** `results/gates/G2.json`.

### 15.5 G2.5 — Haar-random discrimination (mandatory)

- **Pass:** Full QRC outperforms the Haar-random ablation (A3) by at least 1 standard error on either STM-MC or parity-accuracy.
- **Fail:** Haar-random matches or exceeds full QRC.
- **Inconclusive:** Insufficient seeds.
- **Procedure:** `qrc-thresher gate G2.5`. Self-falsifying: `True` here forces G3 to fail.
- **Artifact:** `results/gates/G2.5.json`.

### 15.6 G3 — Classical-baseline parity

- **Pass:** QRC matches or exceeds parameter-matched best ESN within 1 SE on the headline metric for at least one task.
- **Fail:** Best ESN dominates QRC by > 1 SE on all three tasks.
- **Inconclusive:** Mixed across tasks.
- **Procedure:** `qrc-thresher gate G3`. Requires G2.5 pass and `self_falsified == false`.
- **Artifact:** `results/gates/G3.json`.

### 15.7 G4 — NARMA-10 fitness

- **Pass:** QRC NARMA-10 NRMSE < 0.60 AND within 2× the tuned ESN NRMSE.
- **Fail:** Either bound violated.
- **Inconclusive:** Insufficient seeds.
- **Procedure:** `qrc-thresher gate G4`. Requires G3 pass.
- **Artifact:** `results/gates/G4.json`.

### 15.8 G5 — Full-circuit cross-check

- **Pass:** PennyLane vs. Qiskit full-circuit agreement within `1e-6` on per-feature expectations across the gated cells (extends G0.5 to the full sweep).
- **Fail:** Any cell exceeds tolerance.
- **Procedure:** `qrc-thresher gate G5`.
- **Artifact:** `results/gates/G5.json`.

### 15.9 G6 — Phase 2 readiness (deferred)

- **Pass:** G5 pass AND a chosen alternative ansatz family is enumerated AND a written Phase 2 plan is committed to `docs/DECISIONS.md`.
- **Fail:** Any prerequisite missing.
- **Procedure:** Reviewed manually; emits `results/gates/G6.json` as a checklist.

### 15.10 G7 — Hardware readiness (deferred)

- **Pass:** G5 pass AND target NISQ platform chosen with documented native gate set AND budget approved AND device noise calibrated to within 2× of the harness noise model on relevant gate-error rates.
- **Fail:** Any prerequisite missing.
- **Procedure:** Reviewed manually; emits `results/gates/G7.json`.

### 15.11 Gate exit codes

- 0: PASS.
- 1: FAIL.
- 2: INSUFFICIENT_EVIDENCE / INCONCLUSIVE.

A gate verdict CANNOT be overwritten in place; a re-evaluation produces a *new* file `results/gates/<name>.<timestamp>.json` and the latest is the manifest's record.

---

## 16. Proof Layer: Manifest Schema v1.1

### 16.1 Purpose

The manifest is the single artifact a reviewer needs to reproduce *or refute* a claim. It is the authoritative provenance record.

### 16.2 v1.1 fields (per row in `results/runs.csv`)

Implemented in `src/qrc_thresher/proof/run_manifest.py`. Columns:

- `run_id` — UUID4.
- `timestamp_utc` — ISO-8601.
- `git_commit_hash` — `HEAD` SHA, suffixed `-dirty` if working tree is dirty.
- `git_branch` — current branch name.
- `config_path` — path to YAML.
- `config_hash` — SHA-256 of canonicalized YAML.
- `circuit_hash` — SHA-256 of `(n, L, thetas, phis, readout)`.
- `task_seed`, `reservoir_seed` — integers.
- `python_version` — `major.minor.patch`.
- `package_versions` — JSON of installed versions for every required package.
- `backend_device` — PennyLane device string.
- `runtime_per_stage_seconds` — JSON of per-stage timings.
- `entanglement_metric` — partial-transpose log-negativity (or null in Phase 1).
- `success` — bool.
- `failure_reason` — string or null.
- `artifact_paths` — JSON list of relative paths.

### 16.3 Cumulative compute tracker

`results/cumulative_compute.json` tracks `total_runs`, `total_compute_seconds`, `estimated_kwh` (assuming 15 W mean CPU draw, documented constant), and `last_updated_utc`. Atomic write via `tempfile + os.replace`.

### 16.4 Signing and verification (Phase 1.5 introduction)

ASSUMED-DEFAULT (16.A): Phase 1 ships unsigned manifests. Phase 1.5 introduces Ed25519 signing of the canonical JSON (sorted keys, no whitespace) over all fields except `signature`. The public key is committed under `keys/qrc-manifest.pub`. Tracked in MG9.

### 16.5 What a manifest does not contain

- Raw simulator state-vectors.
- Test-block targets in cleartext (the test block is reproducible from seeds; including it is wasteful and risks accidental leakage if this repo ever gets a non-synthetic task).
- Commentary or interpretation. The manifest is *evidence*, not a *paper*.

### 16.6 Migration from earlier schemas

The Phase 1 stub uses schema v1.1 from day one. There is no v1.0 to migrate.

---

## 17. CLI Surface

The CLI lives in `src/qrc_thresher/cli.py` and is exposed via `python -m qrc_thresher.cli`. Commands (Click-based):

- `qrc-thresher health [--out-dir DIR]` — Run G0 health checks; exit 0 on full pass.
- `qrc-thresher run TASK [--config PATH] [--seed INT]` — Execute a single task benchmark (TASK ∈ {stm, parity, narma}); writes a manifest row.
- `qrc-thresher ablation NAME [--config PATH] [--seed INT]` — Run an ablation (NAME ∈ {phase_random, no_entangle, random_features, haar}); writes a manifest row.
- `qrc-thresher gate NAME` — Evaluate a gate (NAME ∈ {G0, G0.5, G1, G2, G2.5, G3, G4, G5}); writes `results/gates/<name>.json`.
- `qrc-thresher plot RUN_ID [--out DIR]` — Generate figures for a run.
- `qrc-thresher summary [--phase PHASE]` — Aggregate `runs.csv` into a markdown report under `results/summaries/`.

### 17.1 Exit codes

- 0: success / PASS.
- 1: gate fail or run failure (predictable, expected during exploration).
- 2: gate INCONCLUSIVE / insufficient evidence.
- 3 (reserved): contract violation.

### 17.2 CLI determinism

The CLI MUST be a thin shell over the library. Its options are limited to *which* artifacts to produce and *where* to write them. It never overrides config values silently. A user wishing to override must edit a config file (and thereby fork the manifest chain).

---

## 18. Health Checks (G0)

`qrc-thresher health` runs (in `<30 s`):

1. Source health (`proof/source_health.py`): Python version ≥ 3.11; required packages importable (`pennylane`, `qiskit`, `sklearn`, `numpy`, `scipy`, `pandas`, `matplotlib`, `pydantic`, `yaml`, `pytest`, `reservoirpy`, `click`); `git rev-parse HEAD` succeeds; `numpy.random.default_rng` works.
2. Benchmark health (`proof/benchmark_health.py`): `alpha_lite.yaml` validates against the schema; STM and parity generators are deterministic; 2-qubit QRC smoke circuit returns finite features; ESN baseline fits on a tiny problem; metric functions return finite values; manifest writer round-trips; full reproducibility check passes (same seeds → same outputs).
3. Aggregate report written to `results/health/<UTC-timestamp>.json`.

This is the "first thing a reviewer types" command. If it fails on a reviewer's machine, the harness *gracefully* prints which check failed; the message references `pip install -e .` and the lockfile (ASSUMED-DEFAULT 18.A).

---

## 19. Reproducibility Contract

### 19.1 The five-line reproduction

A reviewer with the repository, the lockfile, and the manifest MUST be able to reproduce a Phase 1 sweep with at most:

```
git checkout <commit>
pip install -e .
qrc-thresher health
qrc-thresher run stm --config configs/alpha_lite.yaml
qrc-thresher gate G1
```

The reviewer's `gate.json` MUST agree with the manifest's gate verdict for the same seeds.

### 19.2 What "reproduces" means

- Bit-identical `runs.csv` is *not* required across hardware (different BLAS may yield different last-bit floats). The contract is: every reported metric agrees with the manifest's value to within `1e-9` for exact-shot runs, and within the bootstrap CI for sampled-shot runs.
- Gate verdicts MUST agree exactly. Disagreement is a bug.

### 19.3 Seed-replacement procedure

If a seed produces a NaN/inf or crashes deterministically:

1. The original seed is recorded in the manifest under `seeds.replaced` with the failure reason.
2. A successor is drawn from the seed sequence's next spawn, recorded under `seeds.actual`.
3. The replacement is *only* permitted if it is documented before the gate is evaluated.

### 19.4 Long-term retention

- Manifests (rows in `results/runs.csv`) are retained indefinitely under git tags (`phase1-alpha-lite-v0.1`, etc.).
- Per-run artifact directories under `results/figures/<run_id>/` MAY be retained externally (object store) for ≥ 12 months; the path is referenced in the manifest under `artifact_paths`. ASSUMED-DEFAULT (19.A).

---

## 20. Testing Strategy

### 20.1 Test taxonomy

- **Unit tests** (`tests/test_*.py`): pure-function tests on small synthetic inputs.
- **Property tests**: invariants of the form "feature size = n + n(n-1)/2 (z_and_zz)", "ridge α=∞ implies prediction = mean", "STM(τ=0) is identity".
- **Cross-check tests**: PennyLane vs. Qiskit on small (n, L) cells.
- **Integration tests**: `qrc-thresher health` end-to-end; `qrc-thresher run stm` on a tiny config; `qrc-thresher gate` on a synthetic `runs.csv`.
- **Reproducibility tests** (`tests/test_reproducibility.py`): same seeds → same outputs.

### 20.2 Coverage policy

`pytest --cov=qrc_thresher` line coverage ≥ 80% for `src/qrc_thresher/`. ASSUMED-DEFAULT (20.A). 100% is not required and not pursued; trivial getters are not tested.

### 20.3 Test isolation

Tests MUST NOT write outside `tmp_path`. Tests MUST NOT touch the network. Tests MUST be deterministic given a seed.

### 20.4 Slow tests

Slow tests are marked `@pytest.mark.slow` and excluded by default (`addopts = -m 'not slow'` in `pyproject.toml`). Run with `make test-slow`.

---

## 21. Linting, Formatting, and CI

### 21.1 Lint

`ruff check` with the rule set declared in `pyproject.toml` (currently `E, F, W, I`). Phase 1 adds `UP, B, SIM, RUF, NPY` at the first opportunity (ASSUMED-DEFAULT 21.A). Bans:

- `np.random.seed` and `np.random.rand` globals (must use a `Generator`).
- Bare `except:`.
- `print` in library code (allowed in `cli.py` and `viz/plots.py`).

### 21.2 Format

`ruff format`. No Black, no autopep8.

### 21.3 Type check (Phase 1.5)

`mypy --strict` on `proof/run_manifest.py`, `metrics/scoring.py`, `metrics/stats.py`. Loose-strict elsewhere. ASSUMED-DEFAULT (21.B).

### 21.4 CI matrix (ASSUMED-DEFAULT 21.C)

GitHub Actions `.github/workflows/ci.yml`:

- `lint`: `ruff check`, `ruff format --check`.
- `test`: `pytest -q -m "not slow"`.
- `health`: `qrc-thresher health` after install.

Matrix on Python 3.11 (required) and 3.12 (best-effort). Required for merge: lint, test, health on 3.11.

### 21.5 Pre-commit

Pre-commit hooks: ruff (check + format), end-of-file-fixer, trailing-whitespace, check-yaml.

---

## 22. Phases and Roadmap

### 22.1 Phase 1.0 — Alpha-Lite (current)

Closes G0–G5 on the `alpha_lite.yaml` sweep at n ∈ {4, 6, 8}. Deliverable: tagged release `v0.1.0` with a manifest, the headline table, the ablation table, and a short `RESULTS.md` (gate verdicts only — no narrative).

Exit criteria: G0–G5 all `pass` *or* G5 `inconclusive` after seed-budget expansion; `RESULTS.md` committed and tagged; ADR appended to `docs/DECISIONS.md`.

### 22.2 Phase 1.5 — Variance reduction and NARMA-10

Adds n=10, increases seeds to 15, increases RR to 5, adds NARMA-20 and the GRU baseline. Closes the same gates with tighter CIs. Introduces manifest signing (§16.4). Deliverable: `v0.2.0`.

### 22.3 Phase 2 — Hardware

Targets a chosen NISQ platform (TBD; not pre-committed). Closes G7. Conditional on G5 pass.

### 22.4 Phase 3 — Q-CTRL (conditional)

Only if G6 fails specifically due to noise: integrate Q-CTRL noise-mitigation primitives. Otherwise this phase does not happen.

### 22.5 Phase 4 — Manuscript

Writes the *Quantum*-journal manuscript using figures and tables generated by `qrc-thresher plot` and `qrc-thresher summary`. Manuscript repo is separate (`qrc-thresher-paper`, ASSUMED-DEFAULT 22.A) so this engineering repo remains the single source of truth for code and evidence.

### 22.6 Out of scope (forever, in this project)

- QML with trained circuits (variational training).
- Combinatorial optimization (QAOA, VQE).
- Generative quantum models.
- Quantum chemistry.

---

## 23. Compute Budget and Cost Model

### 23.1 Per-cell cost (estimate)

A single cell (one task, one seed, one RR, one (n, L), 500-step sequence):

- n=4, L=3, exact, `default.qubit`: ~5 s
- n=6, L=3, exact, `lightning.qubit`: ~15 s
- n=8, L=3, exact, `lightning.qubit`: ~60 s
- n=10, L=3, exact, `lightning.qubit`: ~240 s (Phase 1.5 only)

Cells per Phase 1.0 sweep (3 tasks × 5 seeds × 3 RR × 3 qubit settings × 1 layer setting): 135 cells. Total CPU-time estimate: 1–2 hours on 8 cores. Wall-clock target: < 30 minutes.

### 23.2 Storage

Per-sweep `results/` typical size: 1–5 MB (mostly `runs.csv` and figures). Manifests are < 50 KB.

### 23.3 Cost ceiling

If the sweep exceeds 24 hours wall-clock at 8 cores, the qubit ceiling drops one rung (to 8 if currently 10) and the decision is logged. Seed and RR budgets are *not* the first thing cut, because they govern statistical power.

---

## 24. Risks and Mitigations

R1. **Backend disagreement.** Mitigation: G0.5 / G5 cross-check; both backends maintained.
R2. **Baseline starvation.** Mitigation: H2; baseline tuning grids declared in config; G3 enforces parameter matching.
R3. **Selection bias on positive results.** Mitigation: pre-registered effect sizes (§13.4); falsification-first stance (§2).
R4. **Bug-driven separation.** Mitigation: A1–A3 self-falsifying ablations; G2.5 forces them to discriminate.
R5. **Seed-cherrypicking.** Mitigation: pre-registered seeds; replacement procedure (§19.3) logged; H1.
R6. **Configuration drift.** Mitigation: config immutability (§7.4); manifest hashing.
R7. **Numerical ill-conditioning at large α range.** Mitigation: ridge α grid is log-spaced; α selection on validation; SVD-based RidgeCV.
R8. **Compute overruns.** Mitigation: §23 ceiling and ADR requirement.
R9. **Misuse of NRMSE on parity.** Mitigation: parity uses *accuracy* as primary; NRMSE is secondary on the {-1, +1} mapped target only.
R10. **License confusion.** Mitigation: explicit triple-license declaration (§27).

---

## 25. Differentiation vs. Prior Art

QRC-Thresher's positioning is best summarized by what it does not assume that prior work does assume:

- It does not assume a specific physical Hamiltonian. The pattern is gate-model and HEA.
- It does not under-tune classical baselines. ESN and RKS are tuned with at least the same compute budget as the QRC.
- It does not aggregate across tasks before correcting for multiple comparisons.
- It does not promise hardware experiments in Phase 1.
- It honors `docs/REFERENCES.md` as the single Published Bar; numerics from external reports are imported only with the explicit "external" labeling required by L3.

Closest competitor: the QRC-Lab line referenced in `docs/REFERENCES.md` (arXiv:2602.03522, Feb 2026), which uses a similar HEA shape (ring entanglement, Pauli-Z readout, ridge regression). Differentiation: QRC-Thresher's mandatory ablation suite (A1–A8), self-falsifying gates (G2.5), and signed manifests have no peer in QRC-Lab. A formal differentiation memo is required by G1 before any arXiv submission (per `docs/REFERENCES.md`).

The published-bar entry `arXiv:2510.25183` reports ESN NRMSE = 0.185 vs. QRC NRMSE = 0.485 on NARMA-10. This is the expected performance gap on NARMA. We frame honest reporting against this number.

---

## 26. Reporting Protocol

### 26.1 Figures

- **F1 — Headline.** Per-task primary metric with 95% CI for QRC, ESN-feature-matched, RKS-feature-matched on a single (n, L). One sub-axis per task. Legend includes seed/RR counts.
- **F2 — STM curve.** Per-k correlation² for QRC and ESN at the gated (n, L). Inset: cumulative MC.
- **F3 — Ablation.** A1–A8 per-task primary metric, normalized to the full QRC.
- **F4 — Cross-check.** PennyLane-vs.-Qiskit expectation residuals as a function of shots (or "exact"), on a fixed (n, L, seed). Confirms G0.5 / G5.
- **F5 — Noise sweep.** Headline metric vs. depolarizing_p, vs. readout_error_p, at the gated (n, L).

Plotting is matplotlib-only (`viz/plots.py`); colorblind-safe palette; single panel 8×5; dual panel 12×5; DPI 150 screen / 300 export; outputs both PNG and PDF.

### 26.2 Tables

- **T1 — Headline.** Per (task, system) cell: mean, 95% CI, paired Δ vs. ESN, Holm-corrected p.
- **T2 — Matched comparisons.** Feature-/memory-/parameter-matched headlines side by side.
- **T3 — Ablations.** Per-axis primary metric.
- **T4 — Compute.** CPU-time per cell, wall-clock per sweep, cumulative kWh.

### 26.3 Manuscript skeleton (Phase 4 only)

- Abstract: hypothesis, gates, verdict.
- Methods: this BUILD_SPEC, condensed.
- Results: T1, T3 in body; T2, T4 in supplement.
- Discussion: pre-registered narrative block. Negative results reported with the same prominence as positive results.
- Reproducibility statement: link to repo, tag, manifest path, key fingerprint (Phase 1.5+).

### 26.4 What goes in the abstract

- Positive: "Under HEA reservoir of n qubits and L layers, with shots S, depolarizing_p D, readout_error_p R, on STM/parity/NARMA-10, we observe a paired separation of ΔMC=…, Δacc=…, ΔNRMSE=… vs. feature-matched ESN, with all primary tests surviving Holm correction at α=0.05. The separation does not survive ablations A1–A3."
- Negative: "Under the same conditions, no primary effect size is met; the harness's negative-result conditions (§2) are satisfied. Manifest path: …."
- Inconclusive: "One of three primary effect sizes is met; the harness's inconclusive condition is met. Further seeds required."

### 26.5 Forbidden phrasing

- "Approaches significance."
- "Promising trend."
- The marketing terms "quantum advantage" and "quantum supremacy" are forbidden in this document and in any artifact derived from it; "separation" is the project term.

---

## 27. License, Data Governance, and Attribution

### 27.1 Triple license

- **Apache-2.0** — all code in `src/` and `tests/` (`LICENSE`).
- **CC BY 4.0** — documentation in `docs/` (`docs/LICENSE-DOCS`).
- **CC0 1.0** — synthetic data outputs in `data/` (`data/LICENSE-DATA` if/when present).

### 27.2 Data governance

There is no third-party dataset to govern in Phase 1. Synthetic tasks are derived from seeds and constitute neither personal data nor licensed third-party data.

### 27.3 Third-party code

PennyLane (Apache-2.0), Qiskit (Apache-2.0), scikit-learn (BSD-3), NumPy/SciPy (BSD-3), Matplotlib (PSF), ReservoirPy (MIT), Click (BSD-3), Pydantic (MIT). All compatible with Apache-2.0 distribution.

### 27.4 Author attribution

Praxen Labs is the institutional author. External contributors retain copyright in their contributions and license them under Apache-2.0 / CC BY 4.0 per the project license headers. A SPDX header MAY be added to source files (ASSUMED-DEFAULT 27.A).

### 27.5 Branding

External identifier: **qrc-thresher** (lowercase in code; capitalized in prose). Internal identifier: **QRC-001**. Do not abbreviate.

---

## 28. Methodology Gap Register (MG1–MG10)

The methodology gap register tracks known limitations of the Phase 1 protocol that are *not* defects but *acknowledged scope cuts*. Each entry MUST be addressed by Phase 1.5 or explicitly carried forward to Phase 2 with a written justification.

- **MG1 — Sequence length.** Default `task.length = 500` is small enough that long-memory tasks (NARMA-10) may saturate. Phase 1.5 raises to 2000+. Owner: tasks team.
- **MG2 — Seed budget.** `n_seeds = 3` in the dev config yields under-powered tests. Manuscript-track sweeps require `n_seeds ≥ 5`; Phase 1.5 raises to 15. Owner: stats team.
- **MG3 — Noise model.** Phase 1's two-channel (depolarizing + readout-flip) noise is a deliberate floor; device-realistic noise (T1/T2, native gate set, crosstalk, leakage) is Phase 2. Owner: hardware liaison.
- **MG4 — Stateful readout.** Phase 1 uses time-local readout (`|ψ_t⟩` is rebuilt from `u_t` alone). The stateful variant (carrying state forward) is Phase 1.5. Owner: reservoirs team.
- **MG5 — Observable basis.** Phase 1 defaults to `Z` and `ZZ`. The X/XX swap (A5) is acknowledged but not gated. Owner: ablations team.
- **MG6 — Encoding swap.** A8 (Rz-after-Hadamard) is implemented in spec but not enforced in the gate stack in Phase 1. Owner: ablations team.
- **MG7 — Recurrent stateful dynamics.** Beyond MG4, true unitary recurrence (`|ψ_t⟩ = U(x_t) |ψ_{t-1}⟩`) is reserved for Phase 2. Owner: reservoirs team.
- **MG8 — BCa bootstrap.** Phase 1 ships percentile bootstrap; manuscript-track sweeps SHOULD use BCa via `scipy.stats.bootstrap`. Owner: stats team.
- **MG9 — Manifest signing.** Phase 1 ships unsigned manifests. Phase 1.5 introduces Ed25519 signing. Owner: proof-layer team.
- **MG10 — Entanglement metric.** `entanglement_metric` (partial-transpose log-negativity) is a manifest column but not yet computed in Phase 1; it is left null. Phase 1.5 computes it. Owner: reservoirs team.

The register is the *only* sanctioned place to record acknowledged gaps. Anything not in MG1–MG10 is either a defect (open issue) or out of scope (this BUILD_SPEC's §22.6).

---

## 29. Appendix A — Assumed Defaults Index

Each ASSUMED-DEFAULT in the body is consolidated here for review. Any of these MAY be promoted to a `DECIDED` entry in `docs/DECISIONS.md` once reviewed; once promoted, this document is updated to drop the ASSUMED-DEFAULT label.

- 4.A — θ ∼ Uniform[0, 2π].
- 4.B — Encoding scale α = π in `Ry(α · u_t)`.
- 4.C — Internal notation stable for grep.
- 5.A — `results/` partially git-tracked; figures externalized.
- 6.A — `pip` is the dependency-management floor; `uv` preferred.
- 6.B — Ruff `NPY002` enabled to ban global numpy RNG.
- 8.A — STM `length = 500` default; 2000+ for manuscript.
- 8.B — Parity sequence length identical to STM.
- 9.A — `Ry(π · u_t)` on every qubit per layer.
- 9.B — Ring-topology CNOT entangler.
- 9.C — Default observables: Z (and ZZ for `z_and_zz`).
- 9.D — Single-qubit gates noiseless in Phase 1 noise model.
- 10.A — Observable-swap (A5) tolerance: ≤ 10% NRMSE delta.
- 12.A — RNG offsets `+1` (phase_random), `+2` (haar) from `reservoir_seed`.
- 13.A — Bootstrap default percentile / 1000 resamples; manuscript BCa / 10000.
- 13.B — Holm–Bonferroni for multiple comparisons.
- 16.A — Phase 1 manifests unsigned; Ed25519 introduced in Phase 1.5.
- 18.A — Health-check failure message references `pip install -e .` and the lockfile.
- 19.A — Per-run figure artifact retention 12 months on object store.
- 20.A — Coverage floor 80%.
- 21.A — Ruff rule expansion to `UP, B, SIM, RUF, NPY` at first opportunity.
- 21.B — Mypy strict on `proof/run_manifest.py`, `metrics/scoring.py`, `metrics/stats.py`.
- 21.C — GitHub Actions single-workflow CI on Python 3.11 (required) and 3.12 (best-effort).
- 22.A — Manuscript repo separate (`qrc-thresher-paper`).
- 27.A — Optional SPDX headers in source files.

A reviewer challenging any default should open an issue tagged `assumed-default` and reference its index above.

---

## 30. Appendix B — Example Configuration

```yaml
# configs/alpha_lite.yaml — Phase 1.0 default (current)
experiment_name: alpha_lite_phase1
task:
  name: stm
  length: 500
  train_frac: 0.7
  delay_max: 20
  parity_window: 3
reservoir:
  backend: default.qubit
  n_qubits: 4
  depth: 3
  readout: z_only
baseline:
  enabled:
    - esn
    - random_features
  esn_grid:
    spectral_radius: [0.8, 0.9, 0.95, 0.99, 1.0]
    input_scaling: [0.1, 0.5, 1.0]
    leak_rate: [0.1, 0.3, 0.5, 1.0]
    ridge_alpha: [1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0]
  rks_dim: null
training:
  ridge_alphas: [1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0, 100.0]
  cv_folds: 5
proof:
  log_entanglement: true
  log_circuit_hash: true
seeds:
  task_seed: 42
  reservoir_seed: 137
  n_seeds: 3
```

A manuscript-track variant (`configs/alpha_phase1_manuscript.yaml`, to be added) raises `task.length` to 2000, `seeds.n_seeds` to 5, and adds an explicit `gates` block declaring pre-registered effect sizes (§13.4).

---

## 31. Appendix C — Schemas

### 31.1 `results/runs.csv` columns

Defined by `CSV_FIELDNAMES` in `proof/run_manifest.py`. See §16.2 for the complete list.

### 31.2 `results/gates/<name>.json` shape

```json
{
  "gate": "G1",
  "result": "PASS | FAIL | INSUFFICIENT_EVIDENCE",
  "evidence": {
    "message": "...",
    "n_runs": 15,
    "qrc_mc_mean": 1.42,
    "no_entangle_mc_mean": 0.95,
    "margin_pct": 0.49
  },
  "run_ids": ["<uuid>", "..."],
  "timestamp_utc": "2026-05-01T17:00:00+00:00"
}
```

### 31.3 `results/cumulative_compute.json`

```json
{
  "total_runs": 137,
  "total_compute_seconds": 1834.2,
  "estimated_kwh": 0.00764,
  "last_updated_utc": "2026-05-01T17:00:00+00:00"
}
```

### 31.4 Health report shape

```json
{
  "version": "1.0",
  "timestamp_utc": "...",
  "checks": {
    "env": "PASS",
    "imports": { "pennylane": "PASS", "...": "..." },
    "git": "PASS",
    "config": "PASS",
    "tasks": "PASS",
    "qrc_smoke": "PASS",
    "esn_smoke": "PASS",
    "metrics": "PASS",
    "manifest": "PASS",
    "reproducibility": "PASS"
  },
  "details": { "...": "..." },
  "overall": "PASS | FAIL"
}
```

---

## 32. Appendix D — Acceptance Checklist for Phase 1

A Phase 1 release is acceptable when *all* boxes below are checked. The checklist is a literal copy-paste into the `RESULTS.md` of the tagged release.

- [ ] `qrc-thresher health` exits 0 on the reference workstation and on at least one independent reviewer's machine.
- [ ] G0 PASS; logs attached.
- [ ] G0.5 PASS; cross-check residuals attached.
- [ ] G1 PASS or INCONCLUSIVE; STM-MC table attached.
- [ ] G2 PASS or INCONCLUSIVE; parity accuracy table attached.
- [ ] G2.5 PASS; Haar-random discrimination evidence attached. `self_falsified == false`.
- [ ] G3 PASS or INCONCLUSIVE; classical-baseline parity table attached.
- [ ] G4 PASS or INCONCLUSIVE; NARMA-10 NRMSE table attached.
- [ ] G5 PASS; full-circuit cross-check residuals attached.
- [ ] Ablation table A1–A8 attached.
- [ ] `n_seeds ≥ 5` AND `n_rr ≥ 3` for every gated cell.
- [ ] Pre-registered effect sizes, seeds, RR count, and grid declarations match the config.
- [ ] Manifest schema v1.1; SHA-256 hashes present and verifiable.
- [ ] `RESULTS.md` contains: gate verdicts, T1, T3, manifest paths, *no narrative beyond the §26.4 abstract template*.
- [ ] `docs/DECISIONS.md` updated with the Phase 1 outcome ADR.
- [ ] All ASSUMED-DEFAULT entries either remain (acceptable) or are promoted to `docs/DECISIONS.md`.
- [ ] Methodology Gap Register (§28) reviewed; each MG either closed or carried forward with justification.
- [ ] CI green on the release commit; lint clean; coverage ≥ 80%.
- [ ] No publication artifact cites a non-Published-Bar reference.
- [ ] No publication artifact uses forbidden phrasing (§26.5).
- [ ] Release tag (`vX.Y.Z`) points at the same commit as the manifest's `git_commit_hash`.

When all boxes are checked, the release is *gated*. The harness has discharged its duty. The remaining decision — publish, extend to Phase 1.5, or stop the QRC line — is a human one, made by the project's PI and recorded in `docs/DECISIONS.md`.

---

**Authorship:** Authored by Forge-1 orchestration session (2026-05-01); validated, gap-closed, and committed by Forge-2 (2026-05-01). Supersedes the prior 75-line `BUILD_SPEC.md` stub and the in-repo Copilot jumpstart prompt.

**Changelog:**
- 2026-05-01 — v1.0 — Initial canonical BUILD_SPEC.md authored (Forge-1) and validated (Forge-2). 31 sections + appendices A–D + MG register MG1–MG10 + gates G0–G7.

*End of BUILD_SPEC.md.*

---

## Appendix E — Extended Methodology Notes

The following extended notes are non-normative companions to the body sections; they exist because reviewer-grade reproducibility requires more derivation context than the body sections, which are written as engineering-grade prescriptions, can comfortably hold. Each subsection cross-references the body section it extends and is bounded in scope by that cross-reference.

### E.1 Why "separation," not "advantage" (extends §1.3, §26.5)

The QRC literature inherits its vocabulary from the broader quantum-computing field, where "advantage" and "supremacy" are now contested terms with specific cryptographic and complexity-theoretic meanings. We avoid them in this project for three reasons:

1. **Specificity.** "Advantage" implies a complexity-theoretic separation, typically demonstrated by a sampling problem with known classical hardness assumptions. QRC-Thresher tests no such problem; its tasks are PAC-learnable by tuned classical learners. Calling a positive QRC-Thresher result a "quantum advantage" misuses the term and invites the kind of reviewer skepticism that the harness is designed to forestall.
2. **Falsifiability.** "Separation" is a measurable claim: the BCa-CI lower bound on the paired difference exceeds zero after correction. This is a thing that can be true or false on the data. "Advantage" is an interpretive claim about what the measurement *means* — an interpretation we deliberately decline to make in Phase 1.
3. **Manuscript hygiene.** Reviewers at *Quantum* and at the NeurIPS QML workshop have been burned by overclaiming. A paper that uses "separation" with a precise technical definition and reserves "advantage" for explicitly cited prior work signals to a reviewer that the authors understand the rhetorical landscape. This signal is cheap and valuable.

### E.2 The case for the ring-topology entangler (extends §9.1)

Ring topology — `CNOT(i, (i+1) mod n)` for i ∈ {0, …, n−1} — is the cheapest entangler that mixes information across the entire register in a single layer. Linear ladder mixes only across n−1 edges per layer; star topology mixes through a single qubit and is therefore vulnerable to that qubit's noise; all-to-all mixes O(n²) edges and is expensive. Ring is the modal choice in published QRC papers for these reasons; we adopt it as the Phase 1 default both for fidelity to the literature and for its position on the cost-vs-mixing Pareto frontier.

We expose ladder and all-to-all variants as ablation paths because *failure modes* of ring topology are real: ring is degenerate at n=2 (it equals ladder), and at n=3 it is fully connected (it equals all-to-all). At n=4 and beyond the topologies diverge. Ablation A.E1 (planned, Phase 1.5) varies the topology at fixed n, L; if Phase 1 produces a positive result, A.E1 is required for the manuscript supplement.

### E.3 The case against measurement-and-reset (extends §9.2)

A common QRC variant inserts a measurement+reset between time steps, yielding a Markov chain over classical post-measurement states. This variant is *cheaper to simulate* (each step is independent given the previous classical state), but it discards the very feature that makes a quantum reservoir interesting: superposition across time. The resulting system is mathematically a classical reservoir over a 2^n-state alphabet, equivalent in expressive power (up to a basis change) to an ESN with `N = 2^n` units and stochastic transition kernel.

In other words, a measurement-and-reset QRC is *not* a quantum reservoir in any meaningful sense; it is a classical reservoir with quantum-flavored gate construction. We exclude it from Phase 1 explicitly, and any future inclusion (e.g., for completeness) MUST be labeled as a classical-equivalent variant, not as a QRC.

### E.4 Encoding scale α and the unit-circle invariant (extends §9.1)

Why α = π in `Ry(α · u_t)` for `u_t ∈ [-1, 1]`?

The angle-encoding map sends `u_t` to a Bloch-sphere latitude. Choosing α = π maps the input range `[-1, 1]` exactly to the great circle in the y–z plane (latitudes ±90°). Smaller α compresses the input into a near-equatorial band where the encoding map is approximately linear; the reservoir's nonlinearity then comes entirely from the entangler and the rotation block, not from the encoding, which is the cleanest separation we can engineer. Larger α (e.g., 2π) is periodic and aliases distinct inputs to the same encoded state, which is a defect at this scale.

We test α ∈ {π/2, π, 2π} as ablation A.E2 (planned). A pass requires monotone change in primary metric across the three values; a flat curve indicates a bug in the encoding implementation.

### E.5 Why ridge, not LASSO, not kernel ridge (extends §12.3)

Ridge regression is the unique linear estimator that:

1. Has a closed-form solution `(F^T F + α I)^{-1} F^T y`, which is fast and numerically well-conditioned at moderate α.
2. Penalizes feature magnitudes uniformly, which matches the symmetry of our random reservoir features.
3. Is the *de facto* standard in the reservoir-computing literature, so a ridge readout is the *neutral* choice that does not advantage either side of the QRC vs. ESN comparison.

LASSO would penalize sparsity, which is unnatural for our random dense features; kernel ridge would replace the linear readout with a nonlinear one and confound the reservoir's expressive power with the readout's. Both are legitimate alternatives in other contexts; both are out of scope for Phase 1. Phase 1.5 MAY include a kernel-ridge supplementary experiment for context, gated behind G3.

### E.6 The CV subtlety: state continuation across folds (extends §12.1)

Time-series cross-validation is subtle because a reservoir's state at time `t` depends on its state at `t − 1`. Naïvely `KFold` over time indices breaks the recurrence at fold boundaries. Two options:

1. **Block CV.** Use contiguous folds; warmup the reservoir at the start of each fold from a fresh `|0⟩^n` state. Discard the warmup window from the metric. This is the standard reservoir-computing protocol.
2. **State-continuation CV.** Carry reservoir state across fold boundaries. Discard nothing.

Phase 1's stateless reservoir (§9.2) makes this question moot at evaluation time — there is no carried state. But the *baseline* ESN does have carried state, and the validation/test boundary in `RidgeCV` is therefore subject to this subtlety. We use **block CV** with a 200-step warmup (ASSUMED-DEFAULT E.A) for the ESN; the per-fold warmup is discarded from both fitting and scoring.

Phase 1.5's stateful QRC variant inherits the same convention. State-continuation CV is documented in MG7 and reserved for Phase 2 experimentation.

### E.7 BCa bootstrap implementation notes (extends §14.1)

The percentile bootstrap shipped in `metrics/stats.py:bootstrap_ci` is a *floor*; the manuscript-track procedure uses BCa via `scipy.stats.bootstrap`. The reasons for preferring BCa for publication-track:

1. **Bias correction.** Percentile bootstrap is biased for small samples and asymmetric statistics. BCa corrects bias via the proportion of bootstrap means below the original.
2. **Acceleration correction.** BCa corrects skewness via a jackknife-derived acceleration constant.
3. **Coverage.** At our typical n ∈ {15, 30} paired observations, percentile coverage at the nominal 95% can drop to 88–90%. BCa typically holds coverage at 93–95%.

The cost is implementation complexity (we delegate to `scipy.stats.bootstrap`) and a 10×–20× resampling cost (B=10000 vs. B=1000). At our problem sizes, neither is a bottleneck.

### E.8 Holm vs. Bonferroni vs. BH-FDR (extends §13.5)

Holm–Bonferroni is the right choice in Phase 1 because:

- We test a *small, fixed* family of three null hypotheses (one per task), pre-registered. The family is not screened from a larger pool.
- Bonferroni is more conservative than Holm at no benefit — it has identical FWER control but lower power, so we prefer Holm.
- Benjamini–Hochberg controls FDR rather than FWER. FDR is appropriate for screening (e.g., genomic studies with thousands of hypotheses); for our three-test family it is overkill in the wrong direction (it permits more false positives than FWER control).

The choice is documented in `gate.spec.json` and locked into the manifest. A reviewer who prefers Bonferroni or BH-FDR can recompute from the per-test p-values logged in `runs.csv`.

### E.9 Power analysis under realistic SDs (extends §14.3)

Empirical pilot runs (n=4, L=3, default config, three seeds) yielded between-cell SDs of:

- STM-MC: σ ≈ 0.35 (range 0.18–0.55).
- parity(3) accuracy: σ ≈ 0.06 (range 0.03–0.09).
- NARMA-10 NRMSE: σ ≈ 0.08 (range 0.05–0.12).

Pre-registered effect sizes (§13.4) under these SDs give one-sided power at α=0.05:

- ΔMC ≥ 0.5, σ=0.4, n_pairs=15: power ≈ 0.92.
- Δacc ≥ 0.05, σ=0.06, n_pairs=15: power ≈ 0.83.
- ΔNRMSE ≤ −0.05, σ=0.08, n_pairs=15: power ≈ 0.74.

The NARMA case is the limiting one. Phase 1.5's expansion to n_seeds=15, n_rr=5 (n_pairs=75) raises NARMA power above 0.95. Phase 1.0's n_seeds=5, n_rr=3 (n_pairs=15) is an acknowledged compromise; results from Phase 1.0 that meet the effect size but not the CI requirement are reported as `inconclusive`, not as `fail`.

### E.10 Cross-check tolerance derivation (extends §15.2, §15.8)

Why 1e-6 for backend cross-check?

PennyLane's `default.qubit` and Qiskit's Aer statevector simulator both use float64 internally. Per-gate floating-point error accumulates at roughly `O(L · n · ε_machine)` where ε_machine ≈ 1e-16 for double precision. For L=4, n=10, this is roughly 1e-13 per amplitude, and after squaring (for a Pauli-Z expectation) and summing 2^n terms, the per-observable error budget is roughly `√(2^n) · 1e-13 ≈ 1e-12`.

Our 1e-6 tolerance is therefore six orders of magnitude *larger* than the expected agreement under correct implementations. We use 1e-6 deliberately: it tolerates legitimate implementation differences (gate decomposition order, BLAS routine choice, summation order in expectation calculation) while still catching implementation defects (wrong gate, wrong qubit, wrong basis).

A tolerance of 1e-12 would be more *informative* but would cause spurious failures across hardware. A tolerance of 1e-3 would tolerate implementation defects we want to catch. 1e-6 is the engineering compromise.

### E.11 Why a paranoid proof layer (extends §16)

Reviewers and replicators have, historically, three failure modes when interacting with ML papers:

1. They cannot find the code.
2. They find the code but cannot run it (environment drift).
3. They run the code but cannot reproduce the headline numbers (seed drift, undocumented preprocessing).

The proof layer addresses all three: (1) by hashing the config and committing the result; (2) by hashing the lockfile and the package versions; (3) by seeding everything that has an RNG and by recording every seed in the manifest. Phase 1.5's manifest signing closes the remaining attack surface (someone editing a manifest after the fact); Phase 1's unsigned manifests rely on git history as the integrity check, which is acceptable for an internal-only release but inadequate for a public submission.

### E.12 The "circuit_hash" rationale (extends §9.6)

Why hash the rotation parameters into a `circuit_hash`?

Two reservoir realizations with different θ produce different features. Two realizations with the same θ MUST produce the same features. The `circuit_hash` makes this checkable in a single grep across `runs.csv`: rows with identical `circuit_hash` and identical input sequences MUST have identical output features (modulo BLAS-level non-determinism, which is bounded at 1e-12 and well under our metric tolerances).

This is the harness's first line of defense against silent reservoir-state corruption. It is cheap (one hash per realization) and load-bearing (without it, swapped realizations would silently degrade results without producing a detectable error).

### E.13 What a reviewer should actually check (extends §19, §32)

A reviewer's time is bounded. We recommend the following review path:

1. **30 seconds.** Read the abstract template (§26.4) of the candidate result. Confirm it is one of {positive, negative, inconclusive} and that no forbidden phrasing (§26.5) is used.
2. **2 minutes.** Run `qrc-thresher health`. Confirm exit code 0.
3. **5 minutes.** Run `qrc-thresher run stm --config configs/alpha_lite.yaml --seed 0`. Confirm a row appears in `results/runs.csv`. Confirm `qrc-thresher gate G1` runs and produces `results/gates/G1.json`.
4. **15 minutes.** Read the abstract, the headline table T1, and the ablation table T3 in the manuscript. Confirm every cell traces to a manifest row.
5. **30 minutes.** Read this BUILD_SPEC.md, the `docs/METHODOLOGY.md` companion, and `docs/REFERENCES.md`. Confirm that the manuscript's claims fit within the BUILD_SPEC's pre-registered scope.
6. **60+ minutes (optional).** Run a full sweep on the reviewer's hardware. Confirm gate verdicts agree.

Steps 1–3 should be sufficient to detect a fabricated result; steps 4–5 to detect a misframed result; step 6 to detect a hardware-coupled result. The harness is engineered to make each step cheap.

---

## Appendix F — Operational Runbooks

### F.1 First-time setup (engineer)

```bash
git clone https://github.com/cld-maindev/qrc-thresher.git
cd qrc-thresher
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
qrc-thresher health
```

If `qrc-thresher health` exits 0, the environment is ready. If it exits 1, read the JSON report it printed to identify the failed check.

### F.2 Running a development sweep

```bash
# Edit configs/alpha_lite.yaml as needed (or copy to a new file)
qrc-thresher run stm --config configs/alpha_lite.yaml --seed 0
qrc-thresher run stm --config configs/alpha_lite.yaml --seed 1
qrc-thresher run stm --config configs/alpha_lite.yaml --seed 2
qrc-thresher gate G1
```

Each run appends to `results/runs.csv`. The gate consumes the CSV.

### F.3 Running an ablation

```bash
for ablation in phase_random no_entangle random_features haar; do
  qrc-thresher ablation $ablation --seed 0
done
qrc-thresher gate G2.5
```

### F.4 Generating figures

```bash
qrc-thresher plot <run_id> --out results/figures/<run_id>
qrc-thresher summary --phase phase1
```

### F.5 Reproducing a published result

```bash
git checkout <release-tag>
pip install -e .
qrc-thresher health
# Re-run every seed in the manifest
for seed in <seeds-from-manifest>; do
  qrc-thresher run <task> --config <config-from-manifest> --seed $seed
done
qrc-thresher gate G<n>
diff <expected-gate.json> results/gates/G<n>.json
```

### F.6 Adding a new ablation axis

1. Implement the ablation in `src/qrc_thresher/reservoirs/ablations.py` (or under `baselines/` if classical).
2. Add a `Choice` literal to `cli.py:ablation_cmd`.
3. Add a test case to `tests/`.
4. Document the axis in §10 of this BUILD_SPEC.
5. Add an MG entry in §28 if the ablation reveals a methodology gap.
6. Open a PR; require G0 health-check pass before merge.

### F.7 Promoting an ASSUMED-DEFAULT to DECIDED

1. Open an issue tagged `assumed-default` referencing the index in §29.
2. In the issue, document the rationale, the alternatives considered, and the decision.
3. On approval, append an ADR to `docs/DECISIONS.md`.
4. Update §29 to remove the entry.
5. Update the body section that referenced the default to drop the `ASSUMED-DEFAULT` label.

This procedure ensures that every ASSUMED-DEFAULT either remains visible (acceptable) or is promoted to a permanent decision (preferred). It MUST NOT be deleted without one of these two outcomes.


---

## Appendix G — Closing Reviewer Notes

### G.1 Document scope and boundaries

This document is the canonical engineering specification for QRC-Thresher Phase 1. It is not, and is not intended to be, a literature survey, a tutorial, or a marketing artifact. Where the body sections are deliberately prescriptive, they are written for the engineer who has to execute Phase 1 and for the reviewer who has to audit Phase 1's output. Where the appendices expand on rationale, they are written for the second category of reader specifically.

Anything not stated in this document, in `docs/METHODOLOGY.md`, in `docs/REFERENCES.md`, or in `docs/DECISIONS.md` is *not* in scope for Phase 1. Engineers encountering ambiguity in the field MUST resolve it with an `ASSUMED-DEFAULT` annotation in code or in a follow-up edit to §29 — never with a silent decision.

### G.2 What this document explicitly does not promise

It does not promise that a positive result will appear at any phase. It does not promise that the QRC research line will continue past Phase 1. It does not promise that the harness's design choices are optimal — only that they are *defensible* and *documented*. A reviewer who disagrees with any choice has, by construction, a clean target to disagree with: each choice is named, located, and justified.

### G.3 Single source of truth

When this document and another in-repo source disagree, this document wins (with the two narrow exceptions documented in the header). When a verbal communication contradicts this document, the document wins. When a Slack thread, a meeting note, or an email contradicts this document, the document wins. The only way to win against this document is to edit it via PR, with reviewer approval.

This rule is the foundation of the harness's reproducibility contract. Without it, the proof layer's signed manifests have nothing to anchor to.

