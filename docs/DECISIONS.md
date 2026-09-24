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
## 2026-09-23: D011 — Matched tuning budgets, and the QRC's hyperparameters

**Decision**: Every tuned model is tuned by one tuner under one budget, fixing the CP2 carry-overs
(the tuning-budget rule, the ESN bias, the ESN tuning metric) and the CP3 carry-over (the encoding
scale). Items marked (builder) were proposed by the builder at checkpoint CP4a; the rest are the
PI's CP4 instructions and rulings.

One tuner
- For each (task, seed pair), every configuration of the model's grid is scored on `cv_folds` = 5
  contiguous validation blocks of the training rows after the washout (D012), with the harness
  readout (`readout.fit_ridge_cv_batched`), one feature matrix per configuration. The best mean
  score wins; ties go to grid order. A configuration whose features or states are non-finite, or
  whose predictions are degenerate (`metrics.scoring.DegeneratePredictionError`), is flagged and
  never selected; if every configuration is flagged the run fails. Test rows are never touched.
- The deployed model equals the validated one: the run row's `circuit_hash` (the ESN's weight
  hash, the RKS hash) equals the tuning record's. `n_configs` and `n_validation_evals` are logged
  on every row.
- Selection metric per task, aligned with D10: STM, the memory sum over k = 1..K of the squared
  Pearson correlation between prediction and u_{t−k} on the validation block (k = 0 excluded),
  named `stm_memory`; parity, held-out accuracy; NARMA-10, NRMSE, lower is better. The ESN's
  tuning score changes from k = 0..K (D009) to k = 1..K. PI ruling 10 (CP4b): the k = 0
  exclusion is one scoring function, `metrics.scoring.stm_memory`, used by every writer, every
  tuner and every ablation; there are no per-member scoring paths.
- (builder) The tuner lives in `qrc_thresher/tuning.py` as one model-agnostic selection routine
  over a list of candidate feature matrices; `baselines/esn.py`'s `tune_esn` becomes a wrapper
  around it, keeping only the ESN-specific lines (the grid, the draw, the states). Each
  configuration record carries the model's hyperparameters, its `circuit_hash`, `score`,
  `degenerate` and `reason`; the record states the metric name and whether higher or lower wins.

Trigger and record
- Tuning is triggered only by a `tuning:` block in the config, holding the three grids `qrc`,
  `esn` and `rks` (below). `alpha_lite.yaml`, `windowed_w2.yaml` and `windowed_w4.yaml` have
  none, so their behaviour and the CP3 routing tests are unchanged. (builder) When a `tuning`
  block is present, `baseline.esn_grid` must be absent: the ESN grid is `tuning.esn`. Without a
  tuning block the ESN is still tuned in-line over `baseline.esn_grid` as D009 describes
  (its rows are `design=tuned` with an empty sweep_id), and RKS runs at D010's default
  (σ = 1, d = 1) as `design=default`.
- `qrc-thresher tune TASK --config FILE` runs the tuner for every seed pair of the config and for
  every tuned model (QRC, ESN, RKS), and writes the tuning record
  `results/tuning/<config_hash>/<task>.json`, keyed by (task_seed, reservoir_seed), with depth,
  window, encoding_scale, circuit_hash, n_configs, n_validation_evals, the validation score of
  every configuration and the winner; the record carries one `sweep_id`, a UTC stamp, that every
  row deployed from it inherits. (builder) The record has one section per model (`qrc`, `esn`,
  `rks`), each keyed by "<task_seed>/<reservoir_seed>", plus `config_hash`, `task`,
  `selection_metric`, `washout` and the grids. PI ruling 3 (CP4b): the record also carries
  `selection_scope: train_cv` (the tuner asserts that no test index is touched), `cv_folds`, the
  seeds used (`seeds`, the list of pairs), the reservoir block it was tuned for, and its own
  SHA-256 (`record_sha256`, over the canonical JSON of the record without that field); every
  row deployed from it inherits `sweep_id` and `tuning_record_sha`. The tuner varies a deep copy
  of the config and builds every QRC configuration through `reservoir_from_config`.
- `run`, the engine and `ablation` read the record when the config has a tuning block, and fail
  if it is missing. `run TASK` deploys design_TASK(pair) and writes `design=tuned`;
  `ablation NAME TASK` builds the matched ablation of design_TASK(pair), whose circuit_hash is the
  design's hash plus the D010 suffix, and writes `design=inherited`; `baseline TASK` deploys the
  record's ESN and RKS configurations and writes `design=tuned`.
- The tuned designs are per task: design_STM(pair), design_parity(pair), design_NARMA(pair).
  G0.7, G1, G2.5 and G3 use design_STM; G2 uses design_parity; G4 uses design_NARMA.
- (builder) G1(b) (D014) needs design_STM evaluated on the parity task. `run` and `ablation`
  take `--design-task {stm,parity,narma}` (default: the task being run) naming the task whose
  tuned design is deployed; the row's task_name stays the task run, and its circuit_hash
  identifies the design. `run`, `ablation` and `baseline` take `--design {tuned,default}`
  (default `tuned`); see "Untuned defaults".

Grids, 60 configurations each
- ESN: spectral_radius {0.8, 0.9, 0.95, 0.99, 1.0} × input_scaling {0.1, 0.5, 1.0} × leak_rate
  {0.1, 0.3, 0.5, 1.0}, unchanged from D009, now with an input bias: b_in ~ Uniform(−1, 1) of
  shape (N,), drawn after W_in from the same stream (W and W_in are unchanged for every seed),
  scaled by input_scaling and added inside the tanh:
  x_t = (1 − a) x_{t−1} + a tanh(ρ Ŵ x_{t−1} + s (W_in u_t + b_in)). `weight_hash` covers b_in.
  D009's "no bias" is superseded. The presets `esn_linear` and `esn_nonlinear` gain the bias and
  their G0.7 results are re-run as findings with no new registered expectation. The G0 smoke bar
  (1.0) and test_esn's known-answer bar (0.75 N) are not moved; if either goes red under the
  bias, it is reported.
- QRC: depth {1, 2, 3, 4, 5} × window {1, 2, 4} × encoding_scale {π/4, π/2, 3π/4, π}. The angles
  for depth L are `build_reservoir_params(depth=L, rng=default_rng(reservoir_seed))`, so each depth
  is a fixed function of the seed and a different depth is a different draw (the builder draws
  all thetas then all phis at the given depth). This is disclosed rather than changed because
  depth 3 at π must stay today's circuit, so that the CP3b findings remain the untuned default.
- RKS: sigma in 20 log-spaced values from 0.1 to 10 (numpy.logspace(−1, 1, 20)) × window
  d {1, 2, 4}. Bandwidth σ/√d on the zero-padded input window of dimension d (D13):
  x_t = (u_t, u_{t−1}, …, u_{t−d+1}) with 0 where the index is negative, and
  φ(x_t) = cos(W x_t + b) with W of shape (F, d), W_{ij} ~ N(0, (σ/√d)²), drawn row-major from
  the stream `default_rng([reservoir_seed, 3])`, then b ~ Uniform(0, 2π) of shape (F,). F is the
  reservoir's feature count as D010. At d = 1 the stream positions of today's draw are unchanged
  and only the scale of W changes (σ/F becomes σ). (builder) The RKS hash is the SHA-256 of
  "rks:n_features=<F>,sigma=<repr(σ)>,window=<d>:" followed by the bytes of W and b.
- (builder) RKS leaves the ablation command: `ablation random_features` is removed, and
  `baseline TASK` runs RKS when `random_features` is enabled, with the same tuner. Baseline rows
  are named by model and task: `esn`, `esn_parity`, `esn_narma` and `rks`, `rks_parity`,
  `rks_narma` (`esn` and `esn_narma` are D009's names). PI ruling 2 (CP4b): the (model, task) →
  task_name mapping and the (member, task) → row-selector mapping live in one helper
  (`gates/comparative.py`) used by the family evaluator and `summary`; nothing else parses
  task_name suffixes. The suffix convention is logged as debt.

Encoding scale
- A config key `reservoir.encoding_scale` (default π) and a `WindowedReservoir` field
  `encoding_scale`, replacing `windowed_qrc`'s import of `_ENCODING_SCALE`;
  `reservoirs/pennylane_qrc.py` stays frozen at π as the w = 1 reference. The circuit hash's
  preimage includes the scale unconditionally (PI ruling 8, CP4b): after D010's optional
  ",window=<w>" the hash appends ",encoding_scale=<repr(float(scale))>" for every reservoir, then
  any ablation suffix. So `WindowedReservoir.circuit_hash` at w = 1 is no longer
  `compute_circuit_hash(params)` (D010's rule is superseded); the features at w = 1 and π stay bit
  identical to today. Schema 1.4 already breaks row compatibility and no gated manifest exists;
  no frozen artifact or untouched test pins a literal circuit hash (checked at CP4b). G0.7's
  model_details record the scale. G0.5 gains the scale axis (D014's
  companion item in the CP4 instructions): `qiskit_features` takes `encoding_scale` (default π),
  and the gate adds the registered cases `G05_SCALE_CASES` = {π/4, π/2, 3π/4} × (4, 3, 137) ×
  w ∈ {1, 2, 4} × both readouts (18 cases), separate from `G05_TRIPLES`, for 34 cases in all.

Untuned defaults
- Untuned defaults are run and reported beside the tuned rows, as `design=default`: the QRC at
  depth 3, scale π, at w = 1 (today's circuit) and at w = 2 (design (a) as D010 defaulted it);
  the ESN as the `esn_nonlinear` preset with the bias (n_configs 1, n_validation_evals 0).
  (builder) `run TASK --design default` writes both QRC rows per pair; `baseline TASK --design
  default` writes the ESN preset row; `ablation NAME TASK --design default` writes the matched
  ablation of the default design at w = 1 and w = 2. Default rows are reported, never joined
  into a family test. PI ruling 1 (CP4b): exactly one default QRC row per member and task enters
  the default table, and the family evaluator must never see two candidate QRC rows for one
  pair (two candidates make the member INSUFFICIENT_EVIDENCE, never a choice). So the w = 2 row
  carries `design=default` and is the compared default (design (a) as D010 defaulted it, the
  design of the CP3b findings), and the w = 1 row carries its own label `design=default_w1` and
  is reporting-only (mean ± std beside the table). `COMPARATIVE.v1.yaml` names both.

G0.7 v1 on the tuned design
- `g07.MODELS` gains `tuned_qrc`, whose feature map for a pair is design_STM(pair) from the
  record. It runs on `alpha_lite.yaml`'s 3 pairs (i < 3) under the frozen every-seed rule; the
  parity clause on that design is a finding. G0.7 v1 stays frozen. (builder) `gate G0.7 --model
  tuned_qrc --config configs/alpha_lite.yaml --tuning-config configs/comparative.yaml`: the
  pairs, readout and measurement come from the experiment config, the designs from the tuning
  config's record, which must hold every pair of the experiment config and share its n_qubits,
  readout and backend. The model details record each pair's depth, window, scale and hash, the
  tuning config's hash and the sweep_id.

Cost, host
- About 8 s per QRC configuration on a continuous task at T = 500 (depth 5 about 1.7× depth 3;
  parity rows are cached), so the sweep is about 60 × 12 × 2 × 8 s ≈ 3.2 h, plus the ablation
  and G0.7 runs, about 4 h. It is a findings run, never a test; the builder runs the plumbing on
  grids of 2–3 configurations.

**Supersedes**: D009's "no bias" and its ESN tuning score over k = 0..K; D010's "RKS ... input
stays u_t, and its bandwidth stays σ/F with σ = 1, until D13" and its RKS hash of F, σ, W and b;
D010's listing of `random_features` among the ablation command's names. The ESN and RKS
budgets of D009 (60 configurations, 300 validation fits) are now matched by the QRC.

**Consequences**: Comparative rows exist only for configs with a tuning block, after `tune` has
written the record. The QRC's untuned default (depth 3, π, w = 1 and w = 2) is reported beside
the tuned design, never in its place. The CP3b G0.7 findings on windowed_w2/w4 stand as the
untuned default's findings. Any ESN row written before this decision has a different hash.

**Rationale**: The CP2 manifest showed the ESN searching 60 configurations and the QRC none.
Giving every model the same budget over its own hyperparameters, chosen on validation blocks
with the harness readout, is what makes a margin a statement about the models rather than about
the search. The encoding scale enters only through the registered grid because the CP3b failure
(RY(πu) on Uniform(−1, 1) inputs gives zero-mean harmonics) would otherwise invite a hand-picked
fix. The RKS bandwidth σ/√d is the standard random-Fourier-feature scaling for a d-dimensional
input, which D13 found the code did not implement.

**Decided by**: PI, 2026-09-23, in the CP4 instructions and rulings (details marked (builder)
proposed by the builder at CP4a). D011 is frozen once committed.

---
## 2026-09-23: D012 — One washout for every model (D16)

**Decision**: `training.washout` = 50 replaces `baseline.esn_washout`, which is refused
(`BaselineConfig` gains `extra='forbid'`; `alpha_lite.yaml`, `windowed_w2.yaml`,
`windowed_w4.yaml` and the tiny config of tests/test_baseline_run.py change).
- Every writer drops rows [0, washout) from training and tuning: `engine.py` (its three task
  branches), `commands/run.py` (three), `commands/ablation.py`, `commands/baseline.py`, the tuner
  of D011, and `proof/benchmark_health.py`'s ESN smoke check (hard-coded 50 today). Test rows
  stay [train_end, T). Scoring is unchanged: the test rows never included the padded rows.
- The value must be at least max(K, the 10 NARMA-10 zero rows plus its transient,
  parity_window − 1, w − 1) and must leave at least 2·cv_folds training rows. At T = 500 and
  train_frac 0.7 it leaves 300 rows = 5 × 60; the T = 100 test configs leave 20. (builder) The
  config refuses a washout below `task.delay_max`, below 10 on NARMA-10, below
  `task.parity_window − 1`, below `reservoir.window − 1`, or one that leaves fewer than
  2·cv_folds rows in [washout, train_end); the transient beyond the 10 zero rows is covered by
  the registered value, 50, and is not a validator. PI (CP4b): the validator enforces structural
  minima only; the NARMA-10 transient is a registered choice. At u ~ Uniform(0, 0.5) the
  NARMA-10 recurrence contracts by roughly 0.5–0.6 per step near its stationary mean, so 50
  steps put the transient below 1e-11.
- G0.7 v1 is frozen at 50 (its own `stm.washout` and `parity.washout`) and unchanged.
- Proof: a spy on `readout.fit_ridge_cv`, `readout.fit_ridge_cv_batched`, g07's imported
  `fit_ridge_cv_batched` and sklearn's `RidgeCV.fit` shows no target row with index below the
  washout, and no NARMA-10 row 0..9, reaching any fit on any path. (builder) The test marks the
  rows [0, washout) of every task's targets with a sentinel value before the writer runs and
  asserts that no fitted target contains it.

**Supersedes**: D009's `baseline.esn_washout` (same value, new home) and its statement that
"QRC runs still train on [0, train_end), zero-padded targets included (D16)".

**Consequences**: Every model, quantum or classical, is fitted and tuned on the same rows
[washout, train_end) and scored on the same rows [train_end, T). The QRC's training set shrinks
by 50 rows, so its metrics on `alpha_lite.yaml` change slightly from the CP2/CP3 values; those
were never registered expectations. A stale `esn_washout` key fails loudly instead of being
ignored, which the previous `BaselineConfig` allowed.

**Rationale**: STM targets are zero-padded for t < k, NARMA-10's first ten targets are exactly
zero, and parity targets before window − 1 are zero. Fitting on them teaches every model a
false constant, and the ESN alone was protected. One washout for every model removes a
difference between arms that has nothing to do with the models.

**Decided by**: PI, 2026-09-23, in the CP4 instructions (the validator list marked (builder)
proposed by the builder at CP4a). D012 is frozen once committed.

---
## 2026-09-23: D013 — Paired statistics and the registered family (D3, D4, D11, D20)

**Decision**: Every comparative gate is a paired comparison decided inside one pre-registered
family. Items marked (builder) were proposed by the builder at checkpoint CP4a.

One config file per comparison
- Every arm of a comparison is run from one config file, so all rows share `config_hash`:
  `run TASK`, `baseline TASK` and `ablation NAME TASK` take the task as an argument (`baseline`
  gains parity; `run` runs every seed pair of the config, serial and parallel; `run --seed` is
  removed).

Pairing
- A paired comparison joins two arms on (config_hash, sweep_id, design, task_seed,
  reservoir_seed): tuned rows against tuned rows (G2, G3, G4), tuned rows against their
  inherited ablations (G1b, G2.5). Budget equality (`n_configs`, `n_validation_evals`) is
  required between two tuned arms, pair by pair. An inherited arm must instead match by design:
  per pair, the ablation row's `circuit_hash` equals the tuned design's hash plus the ablation
  suffix (D010).
- (builder) Arms are identified by the tuning record: a QRC arm is the rows of the named task
  whose `circuit_hash` equals the record's design hash for that pair (design_STM, design_parity or
  design_NARMA as D014 names); an inherited arm is the `ablation:<name>` rows whose hash equals
  that design hash plus the suffix; the classical arms are the `esn*` and `rks*` rows with
  `design=tuned`.
- A pair present in one arm only, a duplicate pair within an arm that is not an exact rerun (the
  same circuit_hash and the same value collapse to one row; anything else is refused), a budget or
  design mismatch, or an all-zero or non-finite difference vector gives INSUFFICIENT_EVIDENCE
  naming the pairs. Nothing is truncated or pooled, and row order never matters.

Statistics per comparison
- n_pairs; the mean difference (arm A minus arm B, in the arms' registered order); the paired
  t-test, one-sided in the registered direction; the Wilcoxon signed-rank test with the
  alternative in the registered direction, zero_method `wilcox`, the effective n (non-zero
  differences) reported, and method `exact` when there are no ties among the non-zero absolute
  differences (`approx` otherwise); a 95% BCa bootstrap CI of the mean difference
  (B = 2000, rng seed 20260923, `metrics.stats.bca_ci`); d_z = mean(diff)/sd(diff) with
  sd over n − 1, written as null when sd = 0. `paired_test`'s field is renamed `d_z` (null at
  zero variance, no longer 0.0). `power_analysis` takes a required `sided` argument (1 or 2)
  and uses the noncentral t distribution of the paired t statistic: 12 pairs detect d_z ≥ 0.77
  at 80% power one-sided, 5 pairs d_z ≥ 1.36; G6's call passes `sided=2`. (builder) The
  two-sided p-value of the paired t-test is reported beside the one-sided one; a t statistic
  that is not finite (sd = 0) is written as null with the p-value scipy gives (0 or 1). Every
  JSON is written with `allow_nan=False`.
- (builder) The comparison lives in `qrc_thresher/metrics/paired.py`: `compare_arms(arm_a,
  arm_b, ...)` takes two DataFrames of manifest rows and returns the pairing verdict and every
  statistic; the gates call nothing else for a comparison.

Decision rule
- The decision statistic is the one-sided paired-t p-value; Holm adjusts the five family members
  together (`metrics.stats.holm_bonferroni`, unchanged); the Wilcoxon p and the CI are reported,
  never decided on. A gate PASSes iff its adjusted p ≤ 0.05 and its floor holds, where one is
  registered (and, for G1, its G0.7 clause). A significant result in the other direction is
  reported as "baseline better" with its two-sided p; the gate FAILs.
- The family is evaluated as a unit: `qrc-thresher gate family --config configs/comparative.yaml`
  (and any member name, G1, G2, G2.5, G3 or G4, which evaluates the whole family and prints that
  member) computes all five raw p-values in one pass, m = 5 fixed. A member that is INSUFFICIENT
  contributes p = 1 and is shown as INSUFFICIENT, never as FAIL. It writes one timestamped family
  JSON plus one per gate, never overwriting, each carrying config_hash, sweep_id, the git commit,
  the measurement label, the protocol hash, n_pairs and every statistic. (builder) File names:
  `results/gates/COMPARATIVE.v1.<UTC stamp>.json` and `results/gates/<gate>.<UTC stamp>.json`,
  in G0.7's style. The family reads `results/runs.csv` restricted to rows whose config_hash is
  the config's and whose sweep_id is the tuning record's. test_baseline_run's fixed-name read of
  G3.json changes accordingly. PI ruling 7 (CP4b): `gate G1`..`gate G4` (and `G2.5`) evaluate
  the whole family, print that member, exit with that member's code and write both files; the
  per-gate file carries the family file's path and SHA-256 (the family file is the record, the
  per-gate file a view). PI ruling 9: `COMPARATIVE.v1.yaml` names the exact set of p-values
  that enter `holm_bonferroni` (`statistics.holm_family`: the one-sided paired-t p of each of
  G1, G2, G2.5, G3 and G4), and the family JSON echoes it as `holm_family`.

Registration
- `configs/gates/COMPARATIVE.v1.yaml`, a frozen pydantic spec with its SHA-256 pinned by a test,
  in G0.7 v1's style: the members, their metrics, directions and floors, alpha 0.05, min_pairs
  12, B and the bootstrap seed, the washout, and the grids by reference to `comparative.yaml`.
- `configs/comparative.yaml`: seeds 42/137 + i for i < 12, `training.washout` 50,
  `task.parity_window` 3, the tuning block with the three grids, `baseline.enabled`
  [esn, random_features], the same readout and measurement as `alpha_lite.yaml`, and no gates
  block. The `gates:` block and `GateThresholds` are removed from every config; the thresholds
  they held (G1 MC > 1.0 and 20% margin, G2 0.70 and 0.60, G2.5 one SE) are superseded by D014.

Manifest schema 1.4
- New columns: `secondary_metrics` (JSON), `device` and `precision` (the constants `cpu` and
  `float64` on every row until D006's GPU extra exists), `design` (tuned | default | default_w1 |
  inherited; PI ruling 1), `sweep_id` (the tuning record's stamp; empty for a config without a
  tuning block) and `tuning_record_sha` (the record's SHA-256; empty likewise; PI ruling 3).
- STM rows carry the memory sum over k ≥ 1 as the primary metric under the name `stm_memory`
  (builder's proposal), with `mc_total` (k = 0..K) and `mc_k0` as secondary metrics; MC_0 is no
  longer folded into any decision. G5 reads `stm_memory` (its semantics wait for Session 2).
  Parity rows keep `accuracy`, NARMA-10 rows `nrmse`.
- (builder) `proof/run_manifest.py` exposes one row builder, `manifest_row(manifest)`, and
  `db.py` writes its CSV rows through it, so the two writers cannot drift. The runs.csv header
  guard refuses old files, as designed; the referee archives any existing `results/runs.csv`.

**Supersedes**: the gate code's list-position pairing and `[:min_len]` truncation (D3), the
single-p Holm (D4), the unpaired means and pooled SE of G1, G2 and G2.5 (D11), `cohens_d` and the
two-sided normal-approximation `power_analysis` (D20), `run --seed`, the fixed-name gate JSONs
of G1–G4, D009's row contract "task_name esn with metric mc" (now `stm_memory`), and the
`gates:` block of D001's config.

**Consequences**: No comparative verdict exists until the tuning record and every arm's rows
exist under one config hash and one sweep_id. A missing pair makes a gate INSUFFICIENT rather
than silently shrinking the sample. Twelve pairs give 80% power for d_z ≥ 0.77 one-sided at
α = 0.05 before the Holm adjustment, so a real margin smaller than that will often be reported as
not significant; that is disclosed, not hidden.

**Rationale**: Pairing by seed removes the between-seed variance of the task sequence from the
comparison, which list-position pairing did only by accident. Registering the family and its
size before the sweep is what makes the Holm adjustment meaningful. Reporting the Wilcoxon p and
the BCa CI alongside the decision statistic shows whether the parametric decision rests on a
few pairs.

**Decided by**: PI, 2026-09-23, in the CP4 instructions and rulings (items marked (builder)
proposed by the builder at CP4a). D013 is frozen once committed.

---
## 2026-09-23: D014 — The comparative gates, re-registered (D10, D11)

**Decision**: G1, G2, G2.5, G3 and G4 are the five members of the family of D013, registered in
`configs/gates/COMPARATIVE.v1.yaml`. Every margin below is the paired comparison of D013 on the
tuned designs of D011, decided by its Holm-adjusted one-sided paired-t p-value at α = 0.05 with
at least 12 pairs. Items marked (builder) were proposed by the builder at checkpoint CP4a.

- G1: (a) design_STM passes G0.7 v1 on `alpha_lite.yaml`'s 3 pairs; (b) the paired margin of
  design_STM over its inherited no-entangle ablation on parity accuracy under the z_only readout
  passes the family rule. D014 records that (b)'s metric was chosen with the CP3b G0.7 findings
  known: the no-entangle ablation of the w = 2 reservoir beats the full circuit on linear memory
  (STM S 0.47/0.77/0.75 against 0.153/0.194/0.255) while the full circuit alone solves parity.
  The STM-memory margin of design_STM over no-entangle is reported beside (b), not gated.
  (builder) Clause (a) is read from the newest `results/gates/G0.7.tuned_qrc.*.json`; the family
  JSON records which file, and G1 is INSUFFICIENT when none exists. PI ruling 4 (CP4b):
  "newest" is the discovery rule only; the evaluator verifies that the file's tuning config hash
  and sweep_id match the rows under evaluation and records the file's path and SHA-256 in the
  COMPARATIVE evidence; a mismatch is INSUFFICIENT_EVIDENCE with both hashes printed.
- G2: design_parity's mean parity accuracy > 0.70 (the floor), and its paired margin over the
  tuned RKS on parity accuracy passes the family rule. G2 is a baseline comparison, not an
  entanglement comparison: D014 supersedes D010's listing of G2 among the z_only-restricted
  comparisons; G1b and G2.5 stay z_only, and are INSUFFICIENT under any other readout.
- G2.5: the paired margin of design_STM over its inherited Haar ablation on the STM memory sum
  (k ≥ 1).
- G3: the paired margin of design_STM over the tuned ESN on the STM memory sum (k ≥ 1); MC_0 is
  reported separately (the mean `mc_k0` of each arm), never compared.
- G4: design_NARMA's mean NRMSE < 0.60 (the floor, BUILD_SPEC §15.7), and its paired margin over
  the tuned ESN on NARMA-10 NRMSE (lower is better) passes the family rule.
- Every gate also reports the default-design comparison beside the tuned one, never gated and
  never part of the Holm family. (builder) The compared default QRC design is `design=default`
  (depth 3, scale π, w = 2; PI ruling 1); the w = 1 row (`default_w1`) is reported as a mean
  beside it. G1b and G2.5 pair the default with the matched ablation of that default design
  (`ablation ... --design default`); G3 and G4 pair it with the default ESN (`esn_nonlinear`
  with the bias); G2 pairs it with the tuned RKS, which has no default. Each default comparison
  carries the same statistics as the tuned one, and INSUFFICIENT when its rows are missing.
  PI ruling 5 (CP4b), rationale: G2's RKS clause is a task-triviality floor, so the strongest
  RKS is the honest comparator in both tables; G3 and G4 are contests, so budgets must match
  (tuned against tuned, default against default). The tuned design is the verdict-bearing
  design: the default table is reported in the same JSON with identical statistics and can
  never flip a verdict; a member that passes on the default and fails on the tuned design is
  FAIL.
- Registered directions: G1b, G2, G2.5 and G3 test "QRC greater"; G4 tests "QRC less".
  Registered metrics: parity accuracy (G1b, G2), `stm_memory` (G2.5, G3), `nrmse` (G4).

**Supersedes**: the G1 of D001's config (`MC > 1.0` with a 20% unpaired margin, D10), the
unpaired G2 (0.70 against an RKS ceiling of 0.60) and the pooled-SE G2.5 of the gate code (D11),
BUILD_SPEC §15.6's "within 1 SE on at least one task" for G3 (D18), and the untested G4. The G2
accuracy floor 0.70 and the G4 NRMSE floor 0.60 are kept as floors.

**Consequences**: G1 depends on the tuned design's G0.7 verdict and so on the encoding-scale
grid of D011: if design_STM fails G0.7 v1 on the 3 pairs, G1 FAILs whatever its margin. A
significant margin in the baseline's favour is a reported finding ("baseline better"), not an
INSUFFICIENT. The family cannot be evaluated until `tune`, the runs, the ablations and the
baselines have all been written from `comparative.yaml`; the referee runs them on the host.

**Rationale**: G1(b) tests the one thing the CP3 findings showed entanglement doing (the
cross-delay product parity needs), against the one ablation that removes it, at the readout
where the ablation isolates it (D010). G2, G3 and G4 are baseline comparisons under matched
budgets, which is the bar the 2026 literature sets. Choosing (b) after seeing the CP3b G0.7
findings is a post-hoc choice, and D014 says so rather than presenting it as pre-registered.

**Decided by**: PI, 2026-09-23, in the CP4 instructions and rulings (items marked (builder)
proposed by the builder at CP4a). D014 is frozen once committed.

---

## 2026-09-24: D015 — (builder) Implementation notes for D011–D014 at CP4b

**Decision**: The choices below were made by the builder while making the CP4a tests green.
None changes a threshold, a metric, a direction, a floor or the protocol file; each is listed
so Chris can reverse it. Items Chris rejects are to be reverted before the CP4b commits land.

- Sweep ids are per tuning record. `tune TASK` stamps its own `sweep_id`, so a config has three
  records and three stamps. A row inherits the stamp (and `tuning_record_sha`) of the record it
  was deployed from: `run parity --design-task stm` carries the STM record's stamp. The family
  evaluator therefore filters each member's tuned rows by the stamp of its *design* task (G1,
  G2.5, G3: STM; G2: parity; G4: NARMA), and G1(a) checks the G0.7 `tuned_qrc` file against the
  STM record's stamp (PI ruling 4). Default-design rows are untuned and carry the stamp of
  whichever record deployed them, so the default table admits rows from any of the config's
  three stamps. The family JSON echoes `sweep_id` and `tuning_record_sha` as one value when the
  three records agree and as a `{task: value}` map otherwise.
- Ablation rows are told apart by task through `primary_metric_name` (`stm_memory`, `accuracy`,
  `nrmse`), since `ablation:<name>` carries no task; `task_names.task_metric` is the one place
  that spells this. Part of the task-name suffix debt of PI ruling 2.
- A QRC arm is the rows of the member's task whose `design` label matches and whose
  `circuit_hash` equals the design's hash *for their own pair*; rows of another design (for
  example design_parity rows on parity when G1(b) wants design_STM) drop out before pairing.
  An inherited arm is selected the same way through the D010 suffix rule, so `ablation ...
  --design default` rows never count as candidates for the tuned comparison. Two candidate rows
  for one pair after this selection are a duplicate and make the member INSUFFICIENT (PI ruling
  1).
- `--design default` deploys two rows per pair (w = 2 `default`, w = 1 `default_w1`); the ESN
  default is the `esn_nonlinear` preset with the bias; RKS has no default design and is skipped
  under `--design default` (PI ruling 5).
- The NARMA-10 minimum of 10 washout rows applies to every config (the task is a CLI argument),
  and the validator reports every violated minimum in one message.
- `summary` writes a per-arm table (task_name, design, metric, n, mean, std) through
  `task_names.parse_task_name`, without the optional `tabulate` dependency.
- The G1..G4 and G2.5 entry points in `pyproject.toml` stay as thin wrappers over the family
  evaluator (`commands.gate._family_member`); `pyproject.toml` is untouched (debt).

**Decided by**: builder, 2026-09-24, for PI review with the CP4b report.

---

## 2026-09-24: D016 — CP4b.1: integrity fixes, one sweep id, hygiene, and the disclosures

**Decision**: The PI accepts D015 as written, with its first item (per-record sweep ids)
superseded for the registered run by B.6 below. The referee's two passes over the CP4b modules
(the PI's read of `gates/comparative.py` and `metrics/paired.py`: no verdict-affecting defect;
an adversarial review of the rest) produced the items below, each implemented with a test.
`configs/gates/COMPARATIVE.v1.yaml` and `configs/gates/G0.7.v1.yaml` are untouched; nothing is
re-pinned.

A. Integrity
- A1. The deployed QRC equals the validated one, by hash. `deploy.qrc_deployments` rebuilds
  design(pair) from the record's (depth, window, scale) and the pair's reservoir seed and refuses
  it (`DesignHashMismatch`, naming the pair and both hashes) unless the rebuilt circuit hash is
  the record's; the matched ablation is built only after that check on the design's own hash.
  `gates/g07.tuned_feature_map` does the same and records the built hash with
  `hash_verified: true`. `run`, `ablation` and `gate G0.7 --model tuned_qrc` refuse a tampered
  record and write nothing. ESN and RKS already refused a mismatch.
- A2. `selection_scope: train_cv` is structural. `tuning.select_configuration` receives only
  the training rows (`targets[:train_end]`; every candidate's feature matrix is built on
  `u[:train_end]`, which for these causal reservoirs equals the first train_end rows of the full
  matrix); it has no `train_end` argument and no assertion. Poisoning every row at or beyond
  train_end with NaN leaves the winners, every score and `record_sha256` unchanged. The record
  also carries `selection_rows = [washout, train_end)`.
- A3. A failed runs.csv write fails the run. `append_to_csv` raises `RunsCsvWriteError` naming
  the path; `ExperimentDB.insert` writes runs.csv before the database, so a failure leaves no
  row anywhere; `run`, `ablation` and `baseline` exit 1 with the message.
- A4. A missing record, an absent pair, a hash mismatch or an unusable design flag is resolved
  for every pair before the first run (`deploy.resolve_deployments`); the writers exit 1 with the
  message and write nothing. The former failure rows (`design=default`, `circuit_hash=n/a`) no
  longer exist.
- A5. The family JSON's `git_commit` is `proof.run_manifest._git_commit_hash()`, with `-dirty`
  when the tree has changes, as every manifest row records it.

B. One sweep id
- B6. `qrc-thresher tune --config FILE` (TASK omitted) tunes stm, parity and narma in one
  invocation and stamps one `sweep_id` across the three records; each record keeps its own
  `record_sha256`. `tune TASK` remains for reruns and stamps its own record. The family
  evaluator echoes one `sweep_id` when the three records agree and falls back to the
  `{task: stamp}` map of D015 otherwise. Under one stamp, design_STM(pair) and
  design_parity(pair) may coincide: `run parity` and `run parity --design-task stm` then write
  two parity rows with one hash, which collapse as exact reruns when their values agree and make
  the member INSUFFICIENT, naming the pair, when they differ (`metrics.paired._collapse`).
  The registered CP4c sequence therefore starts with one `tune --config configs/comparative.yaml`.

C. Hygiene
- C7. One evaluated family per synthetic sweep is shared across `TestFamilyEvaluation`
  (module-scoped fixture, no assertion changed). Linux per-file times before this: family 14.8 s,
  tuning 14.2 s, commands 7.9 s, washout 5.3 s, encoding 3.7 s, schema 2.6 s, paired 2.0 s; the
  15 s host budget is retired.
- C8. Every config model forbids unknown keys, so a misspelled `washout` or `encoding_scale`
  fails validation. An STM config must set `task.delay_max` and a parity config
  `task.parity_window`; `tuning.task_data` refuses to run STM or parity from a config that omits
  them (no silent K = 20 or d = 3).
- C9. `task_names.parse_task_name` raises on an unknown name. `AblationConfig.name` drops
  `random_features` (RKS is a baseline; D011). `--design default` and `--design-task` on a
  config without a tuning block are errors. `gate family` and `gate G1..G4/G2.5` require an
  explicit `--config`. `summary` writes `n/a`, not `nan`, for a group without values.

**Cross-environment reproduction**: the CP3 G0.7 v1 findings on both ESN presets (S, accuracy,
p and the null q95 for every seed) reproduce to every printed digit on Linux / Python 3.11 /
numpy 2.4.6 from the same `uv.lock`, against the sandbox's Windows / Python 3.13 run. The CP4b
suite is 403 passed / 2 deselected in three environments (sandbox 197 s, host 130 s,
Linux 110 s).

**Parity accuracy disclosure** (METHODOLOGY §1.2): an additive readout carries no population
signal about XOR, so its fitted slopes are finite-sample noise and the thresholded accuracy has a
two-sided pathology. Derivation for d = 2: the four input cells (u_{t-1}, u_t) are equiprobable;
the target is 1 on the two mixed cells and 0 on the two pure cells. A linear score
s = a u_{t-1} + b u_t + c takes the values c, c + a, c + b, c + a + b, one per cell, so a
threshold classifies whole cells. The two mixed cells can never be separated from both pure
cells at once (c + a and c + b lie between c and c + a + b when a and b share a sign, and are
the extremes otherwise). Best case: both mixed cells above the threshold with one pure cell
below, three cells right (0.75). Worst case: one pure cell alone above the threshold, both mixed
cells and the other pure cell below, one cell right (0.25). Chance (0.50) is the centre of this
range, not its floor, and a finite test set scatters the observed accuracy around whichever cell
pattern the noisy slopes produced. D010
recorded the upper tail; CP3's `esn_linear` at task_seed 43 (window-2 parity, 48/150 = 0.32;
confusion TP 14, FP 50, FN 52, TN 34; ridge alpha = 1.0 where the other seeds collapse to a
constant at alpha = 100) is the lower tail. The permutation null covers both tails (p = 1.0
there), the paired family comparisons cancel it by seed, and the G2 floor at 0.70 sits above it;
sub-0.5 single-arm accuracies in the default table are this, not a finding.

**BUILD_SPEC drift**: §17 and Appendix F still showed `run --seed` / `ablation --seed` (removed
by D010/D013) and a gate list without `family`; they now show the CP4 commands (every seed pair
from the config; `ablation NAME TASK`; `tune`; `baseline`; `gate family`), with a v1.1 line in
the changelog. README likewise.

**Decided by**: PI, 2026-09-24, in the CP4b.1 instructions; implemented by the builder.

---
