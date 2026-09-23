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

## 2026-09-23: D009 — ESN baseline: dense wiring, one draw per seed, tuning protocol, run path

**Decision**: The ESN baseline is rebuilt in numpy (`baselines/esn.py`) to fix defects D9, D5
and D8.
- Model: the leaky-integrator update x_t = (1 − a) x_{t−1} + a tanh(ρ Ŵ x_{t−1} + s W_in u_t),
  from x_{−1} = 0, with no bias. N_ESN equals the QRC feature count F (n for `z_only`,
  n + n(n−1)/2 for `z_and_zz`), never 2^n. Its states are read exactly, at no measurement cost.
- Wiring (D9): input connectivity 1.0, so every unit sees the input, and a dense recurrent
  matrix, self-connections included. A draw is refused if its spectral radius is non-finite or
  at most 1e-8, or if any scaled weight is non-finite or larger than 1e3 in magnitude.
- One draw per seed (D5): `default_rng(reservoir_seed)` draws Ŵ with i.i.d. N(0, 1) entries,
  normalised to spectral radius 1, then W_in with i.i.d. Uniform(−1, 1) entries. The
  hyperparameters (ρ, s, a) rescale that one draw, so a search compares hyperparameters, not
  draws. A run fails unless the deployed ESN's weight hash equals the validated ESN's.
- Readout: the harness readout every model uses (RidgeCV over `training.ridge_alphas`, with
  `training.cv_folds` contiguous inner folds and an intercept). The ridge penalty is no longer an
  ESN grid entry: `baseline.esn_grid` lists spectral_radius, input_scaling and leak_rate only
  (5 × 3 × 4 = 60 configurations in `alpha_lite.yaml`).
- Tuning (D5): the reservoir runs once over the whole input sequence (inputs only). The first
  `baseline.esn_washout` = 50 rows are dropped from selection, fitting and scoring. The
  remaining training rows [50, train_end) are split into `cv_folds` contiguous validation
  blocks. Each block is predicted by the readout fitted on the other blocks and scored with the
  task's primary metric (STM: MC over the task's delays; NARMA-10: −NRMSE; parity: accuracy).
  The highest mean score wins, and ties go to the first configuration in grid order. Test rows
  are never used. The selected ESN's readout is then refitted on [50, train_end).
- Degeneracy (D9): a correlation-based score (MC) of a column whose standard deviation is at
  most 1e-12 (`metrics.scoring.DEGENERATE_STD`) raises `DegeneratePredictionError` instead of
  scoring 0. In a search, a configuration with degenerate predictions or non-finite states is
  flagged and never selected, and the run fails if every configuration is flagged. Any other
  error propagates.
- Health check (D9): G0's ESN smoke check asserts learning. A 4-unit ESN at the `esn_linear`
  preset, fitted on STM (T = 300, K = 4, 50-row washout, seeds 42/137), must reach a held-out
  memory of at least 1.0 summed over k = 1..4.
- Budget (D5): run manifests (schema 1.3) record `n_configs` and `n_validation_evals` on every
  row: 60 and 300 for the ESN on `alpha_lite.yaml`, 1 and 0 for QRC runs.
- Run path (D8): `qrc-thresher baseline {stm,narma}` writes one row per seed pair of the config
  for every enabled baseline that has a run path, in the row contract G3 and G4 already read:
  task_name `esn` with metric `mc`, and `esn_narma` with metric `nrmse`. It skips
  `random_features` (RKS joins with its bandwidth fix, D13) and `gru` (a stub, D20), and logs
  why.
- G0.7 positive control: two fixed presets are evaluated under G0.7 v1, untuned, with N matched
  to the QRC feature count: `esn_linear` (ρ = 0.9, s = 0.1, a = 1.0) and `esn_nonlinear`
  (ρ = 0.9, s = 1.0, a = 1.0). They are fixed because a tuned feature map would need its tuning
  repeated inside every permutation. Expectation, recorded before any G0.7 evaluation of an ESN:
  `esn_linear` passes the STM clause on every seed (the referee's probe of 2026-09-22 measured
  S = 2.6–3.0 at p = 0.005 for a dense-wired 4-unit ESN). No outcome is registered for either
  preset's parity clause or for the `esn_nonlinear` STM clause; those are reported as findings.
- ReservoirPy stays only as a test cross-check of the state update.

**Supersedes**: BUILD_SPEC E.6's block CV with a per-fold 200-step warmup from a fresh state
(ASSUMED-DEFAULT E.A), for the ESN. The states depend on the inputs only, so one run gives the
validation blocks exactly the states the deployed ESN carries into the test rows. At T = 500 the
350 training rows also give 70-row folds, shorter than a 200-step warmup. The 50-row washout is
the one G0.7 v1 registered (D008); it covers the STM delays (K = 20) and the NARMA-10 transient,
as D16 requires.

**Consequences**: G3 and G4 now have rows to read, but their verdicts are plumbing checks until
CP4. G3 still needs at least 5 runs per arm and pairs rows by list position (D3). QRC runs still
train on [0, train_end), zero-padded targets included (D16). The ESN gets a 60-configuration
search and the QRC none; the manifest shows that difference.

**Rationale**: Dense wiring is what makes a 4-unit ESN a working reservoir. The referee's probe
measured total MC of 3.6–4.0 with it, and a constant predictor without it. One draw per seed
and a shared readout make the search compare hyperparameters under the protocol every model
uses.

**Decided by**: PI, on review at checkpoint CP2 (details proposed by the builder).

---

## 2026-09-23: D010 — Windowed reservoir (design a), matched ablations, G0.5 at one tolerance

**Decision**: Design (a) of D003 is built as follows, fixing defects D1, D12, D14 and D6. Items
marked (builder) were proposed by the builder at checkpoint CP3a; the rest come from the PI's
CP3 instructions and CP3a rulings.

Windowed reservoir (D1)
- Schedule: at every layer, qubit j re-uploads u_{t−(j mod w)} through RY(π·u), using 0 where
  t − (j mod w) < 0. The rest of each layer is unchanged: RZ(θ_{d,j}) and RX(φ_{d,j}) on every
  qubit, then the CNOT ring j → (j + 1) mod n. Every row starts from |0…0⟩, so row t depends on
  u_t … u_{t−w+1} only. A test-local PennyLane circuit, written from this text alone, checks the
  reservoir and its no-entangle ablation against it within 1e-12.
- `reservoir.window` defaults to 1. A window outside 1..n_qubits is refused, by the config and
  by the reservoir. At w = 1 the circuit is today's, and it stays the memoryless negative
  control; `reservoirs/pennylane_qrc.py` is left unchanged as the w = 1 reference. (builder)
  w = 1 runs the same windowed code path as every other w; `alpha_lite.yaml` states
  `window: 1` explicitly.
- The window draws no randomness: the angles come from `build_reservoir_params` with
  `default_rng(reservoir_seed)`, so they are the same at every w. The circuit hash is
  `compute_circuit_hash(params)` at w = 1. For w ≥ 2 it is the SHA-256 of the string
  "<compute_circuit_hash(params)>,window=<w>".
- The reservoir lives in a new module, `reservoirs/windowed_qrc.py`, registered as the builtin
  reservoir plugin `windowed`. Every config-driven build (engine, run command, ablation command,
  G0.7) goes through one helper, `reservoir_from_config(cfg, reservoir_seed, ablation=None)`,
  which reads `reservoir.window` and refuses an unknown ablation name with ValueError. Outside
  the helper's module, only G0.5 and the G0 health check call `build_reservoir_params`
  directly.
- (builder) Speed-up: within one call, each distinct per-qubit input row is simulated once and
  reused. This is not done for phase_random, whose angles change at every step. On binary inputs
  a w-window reservoir has at most 2^w distinct rows. A test checks the result bit for bit
  against the per-step loop.

Matched ablations (D12)
- phase_random, no_entangle and haar inherit the reservoir's per-pair reservoir_seed, readout
  (ZZ included), window and re-upload schedule. Only the tested factor changes:
  - no_entangle removes the CNOT ring;
  - phase_random draws fresh RZ and RX angles at every step;
  - haar replaces each layer's RZ, RX and CNOT ring with an independent Haar-random unitary on
    all n qubits, applied after that layer's re-upload (L draws).
- Random streams are `default_rng([reservoir_seed, tag])`. Today's `default_rng(reservoir_seed
  + 1)` and `(reservoir_seed + 2)` are the reservoir streams of the next seed pairs: pair
  (42, 137)'s phase_random angles at step 0 were pair (43, 138)'s reservoir angles. (builder)
  Tags: phase_random 1, haar 2, random_features 3.
- (builder) Draw order: phase_random draws, for t = 0, 1, …, the step's RZ angles and then its
  RX angles, each Uniform[0, 2π) of shape (L, n), as today. haar draws its L unitaries in layer
  order with `scipy.stats.unitary_group.rvs(2^n, random_state=rng)`.
- Switch-back: restoring the tested factor reproduces the reservoir. For no_entangle and
  phase_random the features are bit-identical (np.array_equal). For haar, feeding each layer the
  unitary of its own RZ, RX and CNOT ring reproduces them within 1e-10, maximum absolute
  difference. (builder) The tolerance is 1e-10 because the two paths apply the same layer as one
  2^n × 2^n matrix instead of 3n gates, so in float64 they differ by rounding only, of order
  1e-15 per operation. That leaves about four orders of magnitude of margin, far below any
  physical effect.
- (builder) An ablation's circuit hash identifies the circuit it simulates. It is the SHA-256 of
  the reservoir's hash followed by ",entangle=False" (no_entangle),
  ",random_phases=[<reservoir_seed>,1]" (phase_random) or ",layer_unitaries=<SHA-256 of the
  unitaries' bytes>" (haar), where the bytes are the L complex128 matrices in C order,
  concatenated in layer order. Restoring the factor (for haar, setting layer_unitaries to None)
  restores the reservoir's hash; a haar ablation fed its own layer unitaries keeps a
  layer_unitaries hash.
- Plugins (PI rulings at CP3a): the builtin reservoir plugins `phase_random`, `no_entangle` and
  `haar` in `plugins/builtin.py` point at the matched ablations, and `windowed` at the reservoir
  itself. (builder) All four take `(u, params, window=1, reservoir_seed=None)` and return
  features of shape (T, F). phase_random and haar need `reservoir_seed` for their streams, and
  they refuse `reservoir_seed=None` with a ValueError that names `reservoir_seed`. The legacy
  functions `extract_features_phase_random`, `extract_features_no_entangle` and
  `extract_features_haar` in `reservoirs/ablations.py` emit a DeprecationWarning that names the
  function, and the referee deletes them in CP4.
- RKS (random_features) gets the reservoir's feature count F (n for z_only, n + n(n−1)/2 for
  z_and_zz) and the per-pair stream `default_rng([reservoir_seed, 3])`. Its input stays u_t, and
  its bandwidth stays σ/F with σ = 1, until D13 (CP4). It is built by `rks_from_config(cfg,
  reservoir_seed)`; `reservoir_from_config(..., ablation='random_features')` raises ValueError,
  because RKS is not a variant of the circuit. (builder) An RKS row's hash is the SHA-256 of F,
  σ, W and b.
- The ablation command runs every seed pair of the config, as the engine does, and writes one
  row per pair under the existing task names `ablation:<name>`. Its train and score rows are
  unchanged (D16, CP4). (builder) Its `--seed` option is removed, since the pairs come from the
  config.
- `random_features` is removed from `baseline.enabled` in `alpha_lite.yaml` until its D13 fix
  (the PI's CP2 ruling).
- The no-entangle ablation isolates entanglement only under the z_only readout. In a product
  state ⟨Z_iZ_j⟩ = ⟨Z_i⟩⟨Z_j⟩, which multiplies delays and can form XOR. Entanglement
  comparisons (G1, G2, G2.5) will be restricted to z_only when they are re-registered in CP4.

G0.5 and one tolerance (D14, D6)
- `CROSSCHECK_TOLERANCE = 1e-6` (float64) is defined once, as a public constant, in
  `reservoirs/qiskit_crosscheck.py`. The gate imports it, and no literal tolerance is passed to
  `verify_crosscheck`. Float32 paths get their own pre-registered tolerance (D006).
- The Qiskit circuit is built independently from the same angles. It has its own per-qubit input
  schedule and imports nothing from PennyLane or from qrc_thresher. It returns the Z
  expectations, then ZZ in PennyLane's pair order (i < j, lexicographic).
- Cases: (n, L, seed) = (2, 2, 2026), (4, 3, 137) and (5, 4, 7), each at every distinct w in
  {1, 2, n}, with both readouts: 16 cases. Each case covers 6 steps, t = 0..5, which include the
  zero-padded rows t < w − 1. (builder) The angles come from `build_reservoir_params` with
  `default_rng(seed)`, and the 6 inputs are the next 6 draws from that generator,
  Uniform(−1, 1). For (2, 2, 2026) the first five are today's G0.5 inputs.
- A case passes when max |PennyLane − Qiskit| ≤ 1e-6 over its rows and features, and G0.5
  passes when every case does. The evidence lists every case with its max |diff|. The tolerance
  is never loosened; any disagreement is reported.
- (builder) The Qiskit side runs Aer's statevector method on the untranspiled circuits (RY, RZ,
  RX and CX are Aer basis gates), one job per case.

G0.7 wiring
- `qrc-thresher gate` gains `--config` (default `configs/alpha_lite.yaml`) and the model
  `no_entangle`, the matched no-entangle ablation of the configured reservoir. G0.7 records the
  window and the ablation in its model details. The G0.7 v1 protocol, its evaluation, clause and
  permutation functions are unchanged.
- `configs/windowed_w2.yaml` and `configs/windowed_w4.yaml` differ from `alpha_lite.yaml` only in
  `reservoir.window` and `experiment_name`.

Expectations, registered before any evaluation (findings, not tests)
- Under G0.7 v1, the w = 2 and w = 4 reservoirs pass the STM clause, the full w = 2 circuit
  passes the parity clause, and its no-entangle ablation (z_only) fails the parity clause. At
  w = 1 the default config reproduces the referee's CP2 host result exactly (FAIL, with
  S = 0.180 / 0.157 / 0.122). No outcome is registered for the w = 4 parity clause or for the
  no-entangle STM clause.
- Per-delay r² is reported, not asserted. RY(πu) alone gives ⟨Z⟩ = cos πu, which is even in u,
  so how linearly a feature carries a delay depends on the random angles (today's r2_k0 is
  0.12 / 0.64 / 0.73).
- What fixes D1 is the exact memory-cliff test. Changing u_{t−k} for any k ≥ w, or any future
  input, leaves row t bit-identical, and changing it for each k < w moves row t by more than
  1e-6. G0.7 verdicts on windowed reservoirs are findings. If a seed fails the STM clause while
  the exact test passes, it is reported. Any design change it motivates, such as the encoding
  scale or the readout, is a new D011, registered before anything is re-evaluated.

Known answer (tests)
- On binary inputs every no-entangle column of the w = 2, z_only reservoir (n = 4, L = 3) has
  interaction contrast X(1,1) − X(1,0) − X(0,1) + X(0,0) = 0 within 1e-12. XOR, whose contrast
  is −2, is therefore outside the span of the columns plus an intercept, and the ablation's
  G0.7 v1 parity clause is expected to fail.
- Being outside the span does not force the clause to fail. Under an additive least-squares fit,
  a (u_{t−1}, u_t) cell is predicted correctly exactly when its training count exceeds the
  harmonic mean of the four cell counts, so a seed can reach 75% held-out accuracy and pass at
  p = 1/201. About 41% of seeds do, so all three seeds, and with them the clause, pass with prior
  probability about 7% in the least-squares limit, or about 1–5% under the registered ridge
  readout. These are the referee's CP3a estimates. Simulating the least-squares rule over 100,000
  sequences, the builder found 41.4% of seeds with three of the four cells right (41.7% with
  held-out accuracy of at least 0.60, a proxy for passing) and 7.3% for all three seeds
  (0.417³).
- The test asserts the clause, not each seed, and it asserts that no seed is degenerate. If the
  clause passes, the red result is reported as a finding about G0.7 v1 and left red: no seed, n,
  depth, encoding scale, window, readout, alpha, threshold or test changes, and no xfail. The
  full circuit's contrast is reported, not asserted.

**Consequences**: A w-window reservoir remembers exactly w − 1 past steps; memory beyond the
window needs a carried-state design (D003). Ablation rows get one distinct hash per seed pair,
and the task names that G1, G2 and G2.5 read are unchanged. No command calls the unmatched
legacy ablations in `reservoirs/ablations.py` any more; they are deprecated until the referee
deletes them in CP4. G0.5 grows from 5 inputs on one 2-qubit circuit to 16 cases.

**Rationale**: The window gives design (a) an exact, testable memory horizon while each row
stays a fresh, stateless simulation, so w = 1 remains today's circuit bit for bit. Matching
the ablations to the reservoir makes each one change a single factor, which an ablation
comparison needs; drawing an independent Haar unitary per layer keeps the re-upload schedule,
so the Haar ablation changes the dynamics and not the input's frequency spectrum (D12 iii).
G0.5 covers the reservoir designs, with both readouts and the padded rows. The ablation
circuits are covered by the switch-back tests and the test-local reference circuit.

**Decided by**: PI, 2026-09-23, in the CP3 instructions. On review at checkpoint CP3a the PI
approved every (builder) item as written, ruled that the legacy ablations are repointed and
deprecated, and removed `ablation --seed`. The Haar byte clause and the plugin signature were
written in the CP3a revision round and confirmed by the PI at the CP3a re-review, which also
ruled that the phase_random and haar plugins refuse `reservoir_seed=None`. D010 is frozen once
committed; any change is proposed as D011.

---
