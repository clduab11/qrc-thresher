# Decision Log

This file is append-only. Never rewrite history.

---

## 2026-05-01: D001 — Initial Phase 1 scaffold

**Decision**: Build complete Phase 1 scaffold per the build spec brief.

**Rationale**: Establish reproducible benchmark harness with all required components before
running any experiments.

**Components included**:
- Task generators: STM, temporal parity (NARMA-10 gated behind G3)
- Quantum reservoir: PennyLane QRC (default.qubit + lightning.qubit)
- Ablations: phase-randomized, entanglement-suppressed, Haar-random
- Classical baselines: ESN (reservoirpy), random kitchen sinks, GRU stub
- Metrics: MC, NRMSE, accuracy, bootstrap CIs, paired t-test
- Proof layer: schema v1.1 run manifests, health checks
- CLI: health, run, ablation, gate, plot, summary commands

**Constraints applied**:
- CPU-only (no CUDA/GPU paths)
- No LLM/transformer references
- Synthetic data only
- All randomness via seeded `numpy.random.Generator`

**Status**: Implemented. 68 tests pass.

---

## 2026-09-22: D002 — Direction: a functioning QRC workbench, with the gates as its quality bar

**Decision**: qrc-thresher is developed into a functioning quantum reservoir computing (QRC)
workbench for the quantum-AI community. It should provide:
- reservoirs that genuinely remember;
- the measurement costs a real device would pay;
- fair classical baselines, built in;
- diagnostics that show what the quantum parts contribute;
- visuals a non-specialist can read.

The pre-registered gates are the quality bar that makes the workbench's results trustworthy,
whether they are positive or negative. The gates are not the product.

**Rationale**: A September 2026 audit of the code found that the default reservoir has no
memory (every feature row is a function of u_t alone) and that some gate evaluators cannot
reach the verdicts their documentation describes. A harness whose default reservoir cannot
remember cannot test memory claims. Building a working workbench, with the gates kept as its
acceptance criteria, addresses both findings.

**Decided by**: PI, 2026-09-22.

---

## 2026-09-22: D003 — Reservoir design order: windowed input first, carried state later

**Decision**: The first reservoir with memory is design (a), a windowed-input reservoir. With
window w, qubit j re-uploads u_{t−(j mod w)} at every layer. At w = 1 this is the current
circuit, which stays in the workbench as the memoryless negative control. The carried-state
designs come later, each on the PI's call:
- (c) a Hamiltonian reservoir (a Trotterized Ising model, or a Rydberg-like model);
- (b) a gate circuit that resets and re-encodes only the input qubit.

Designs (b) and (c) will share one engine: a density matrix carried across time steps, reset
and re-encoding of the input qubit, and a per-step layer. The layer is a Trotterized Ising
step for (c) and a hardware-efficient gate layer for (b).

**Rationale**: Design (a) gives the workbench a reservoir with a known, testable memory horizon
(a memory cliff at delay k = w). Its simulation stays stateless and cheap, and today's circuit
remains reproducible as w = 1. Carried-state designs cost more to simulate (a dense
density-matrix step scales as about 8^n for n qubits), and they need the measurement accounting
of D004 before they can be compared fairly.

**Relation to BUILD_SPEC**: §9.2 describes a windowed *readout* that was never implemented;
design (a) supplies the window through the input instead. Design (b) resets only the input
qubit and carries the rest of the register, so it is not the whole-register
measurement-and-reset variant that Appendix E.3 excludes. The spec text will be reconciled in
a later entry.

**Decided by**: PI, 2026-09-22.

---

## 2026-09-22: D004 — Measurement model: exact expectation values are an oracle upper bound

**Decision**:
- Exact expectation values are used for development. Every report labels them
  "exact (oracle upper bound)", and they are never used for a headline claim.
- The measurement model is a configuration field (`measurement.model`) and a run-manifest
  field (`measurement_model`) from now on. The only accepted value is `exact` until the
  finite-shot path exists. Adding the field moves the manifest schema to version 1.2.
- Finite-shot measurement lands before any comparative sweep. For windowed designs, shots are
  drawn i.i.d. at each time step. For carried-state designs, the reservoir is rerun from the
  washout for each measurement, and the number of circuit executions is logged.
- Monitored or weak measurement is out of scope for v1 and is stated as a limitation.

**Rationale**: A device estimates each expectation value from a finite number of shots, and
measuring a carried-state reservoir disturbs its state. Results computed from exact expectation
values therefore bound from above what hardware could achieve. Recording the measurement model
in the configuration and in every manifest keeps each number traceable to the measurement
assumption it was computed under.

**Decided by**: PI, 2026-09-22.

---

## 2026-09-22: D005 — G0.7 memory sanity gate (option A)

**Decision**: Add gate G0.7, a memory sanity check that a reservoir or baseline must pass before
any memory or comparative claim is made about it. Of the options considered, option A is
adopted: a short-term-memory (STM) permutation test plus a window-2 parity clause. The protocol
and every threshold are pre-registered in `configs/gates/G0.7.v1.yaml`, committed on its own
before any evaluation that uses it. In summary:
- Data: STM with T = 500, train fraction 0.7 and K = 20. The first 50 rows are a washout,
  dropped from training for every model.
- Statistic: S = Σ_{k=1..20} r²_k, where r²_k is the held-out squared Pearson correlation
  between the prediction and u_{t−k}. The k = 0 value is reported separately and never counts
  as memory.
- Null: 200 permutations of the target rows, with train and test rows permuted independently
  and the readout refit each time under the harness protocol. The per-seed p-value is
  p = (1 + #{null ≥ S}) / 201.
- Parity clause: window-2 temporal parity. Held-out accuracy is tested against its own
  200-permutation shuffled-label null.
- Decision: G0.7 passes only if both clauses pass, and a clause passes only if every seed has
  p ≤ 0.05, with at least 3 seeds.
- Reporting: per-delay r²_k with the null's 95th percentile for every seed, a forgetting-curve
  figure, and the measurement-model label.

A changed protocol is added as a new version file (for example `G0.7.v2.yaml` for the planned
information-processing-capacity variant, option C). Version 1 is never edited.

**Rationale**: G1's memory-capacity threshold counts k = 0 and ignores the finite-sample bias of
each r²_k, so a memoryless feature map can pass it. A gate that counts only k ≥ 1 and compares
against an explicit null separates memory from instantaneous reconstruction and from
finite-sample bias. The parity clause adds a nonlinear memory task that linear memory alone
cannot solve.

**Decided by**: PI, 2026-09-22.

---

## 2026-09-22: D006 — GPU use permitted, with CPU float64 as the reference

**Decision**: GPU computation is permitted alongside the CPU (the reference workstation has one
NVIDIA RTX 4060 Ti with 8 GB), under these rules:
- CPU float64 remains the reference for every gate and for CI.
- GPU dependencies live in an optional extra, never in the default dependency set that CI
  installs.
- Any float32 or GPU path needs a tolerance, pre-registered in `configs/` and validated against
  the CPU float64 reference on a subset, before its results are used.
- GPU reductions can be nondeterministic, so the reproducibility contract for GPU paths is
  tolerance-based. Deterministic modes are enabled where the library offers them.
- Manifests record the device and the numerical precision of every run.

**Supersedes**: the "CPU-only (no CUDA/GPU paths)" constraint of D001.

**Rationale**: Consumer GPUs of this class run float64 at a small fraction of their float32
rate, so the GPU pays off for float32 batching (many seeds, permutations, configurations or
trajectories at once), not for single small circuits. Keeping CPU float64 as the reference keeps
every gate verdict independent of the GPU path.

**Decided by**: PI, 2026-09-22.

---

## 2026-09-22: D007 — Division of labor between the builder and the referee

**Decision**: Two AI coding agents work on the repository under the PI's direction, in separate
roles:
- The **builder** runs in a sandbox. It writes and edits files in `src/`, `tests/`, `configs/`
  and `docs/`, and runs development tests in the project's locked environment. It does not run
  git or uv, does not edit `pyproject.toml`, and does not delete files; it lists any such change
  for the referee.
- The **referee** runs on the host. It owns the branch and git, dependency changes (uv, and the
  future GPU extra), `pyproject.toml` and deletions. It reviews every change before running it,
  commits in the order the builder proposes, and runs the authoritative tests and gates in a
  separate environment built from `uv.lock`.
- The PI makes every scientific and scope decision.

**Consequences**: A commit means the change was reviewed. Pre-registered thresholds are
committed, and so timestamped, by the referee before any gate code that tests them exists.

**Rationale**: The sandbox deliberately withholds git, deletions and the files that later run
code on the host. Splitting the roles along that boundary also keeps review independent of
authorship.

**Decided by**: PI, 2026-09-22.

---

## 2026-09-22: D008 — G0.7 v1: protocol details fixed at registration

**Decision**: The G0.7 v1 pre-registration (D005) also fixes the following details.
- Readout: the harness protocol as it is implemented today. One `RidgeCV` fit covers all
  targets of a clause jointly: the 21 delays k = 0..20 share a single ridge penalty. The fit
  uses the experiment config's `training.ridge_alphas`, 5 contiguous (unshuffled) folds and an
  intercept. The gate refuses to run if the config's alphas or fold count differ from the
  registered ones.
- The parity clause uses the STM clause's data settings: T = 500, train fraction 0.7, and a
  50-row washout dropped from training. Labels are {0, 1}, and a prediction counts as 1 when
  the ridge output is at least 0.5 (the harness's classification rule).
- Permutations are drawn from numpy `default_rng([20260922, clause, task_seed, reservoir_seed])`,
  with clause 0 for STM and 1 for parity. Each permutation draws the train order, then the test
  order.
- Degeneracy: a seed is flagged degenerate, and fails its clause without a score, if every
  training feature column has a standard deviation of at most 1e-12, if any held-out prediction
  column does, or if any observed or null statistic is not finite.
- Seeds: the experiment config's seed pairs (task_seed + i, reservoir_seed + i) for
  i = 0..n_seeds−1. Fewer than 3 seeds gives INSUFFICIENT_EVIDENCE.
- Output: each evaluation writes a new timestamped JSON file under `results/gates/`, never
  overwriting an earlier one (BUILD_SPEC §15.11), and a forgetting-curve figure.

**Rationale**: These details determine the p-values, so they are fixed before any evaluation.
The shared ridge penalty matches the readout every current run uses. A per-delay penalty would
be a separate decision and a new gate version.

**Decided by**: PI, on review at checkpoint CP1a (details proposed by the builder).

---
