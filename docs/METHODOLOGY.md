# Methodology

## 1. Tasks

All tasks are synthetic, deterministic, and seeded via `numpy.random.Generator`.

### 1.1 Short-Term Memory (STM)

- **Input**: $u_t \sim \text{Uniform}(-1, 1)$, length $T$.
- **Targets**: $y_t^{(k)} = u_{t-k}$ for $k \in [0, K]$, with $K$ typically 20.
- **Metric**: the memory sum $\text{stm\_memory} = \sum_{k \ge 1} \text{corr}(\hat{y}^{(k)}, y^{(k)})^2$
  over the held-out rows; $k = 0$ is reported separately as `mc_k0` (with `mc_total` over
  $k = 0..K$) and never counts as memory (docs/DECISIONS.md D011, D013). One scoring function,
  `metrics.scoring.stm_memory`, is used by every model, tuner and ablation.
- **Train/test split**: Chronological (no shuffling). Default 70/30. Rows $[0, \text{washout})$
  with `training.washout` = 50 are dropped from training and tuning for every model (D012); the
  test rows are $[\text{train\_end}, T)$.
- **Implementation**: `src/qrc_thresher/tasks/stm.py`.

### 1.2 Temporal Parity / XOR

- **Input**: Binary $u_t \in \{0, 1\}$, length $T$.
- **Target**: $y_t = \text{XOR}(u_{t-d+1}, \ldots, u_t)$ for window $d$.
- **Metric**: Classification accuracy at delay $d$.
- **Window**: `task.parity_window`; `configs/comparative.yaml` uses $d = 3$ (D013), and the parity clause of G0.7 v1 uses $d = 2$ (`configs/gates/G0.7.v1.yaml`, D005).
- **Implementation**: `src/qrc_thresher/tasks/temporal_parity.py`.

**Single-arm accuracy is two-tailed noise for an additive readout (disclosure, D016).** A
readout that is linear in the inputs of the window carries no population signal about XOR: the
fitted slopes are finite-sample noise, and thresholding them yields a held-out accuracy that is
not centred on chance. With $d = 2$ the four equiprobable input cells take one linear score
each ($c$, $c + a$, $c + b$, $c + a + b$), so a threshold classifies whole cells and the two
mixed (target 1) cells can never be separated from both pure (target 0) cells at once: up to
0.75 when one pure cell lands on the right side of the threshold, as low as 0.25 when one pure
cell sits alone on the wrong side (derivation in D016). D010 recorded the upper tail; CP3's `esn_linear` run at task_seed 43
(window-2 parity, 48/150 = 0.32; confusion TP 14, FP 50, FN 52, TN 34; ridge $\alpha$ = 1.0
where the other seeds collapse to a constant at $\alpha$ = 100) is the lower tail. Three things
make this harmless for the gates: the permutation null of G0.7 covers both tails (p = 1.0 for
that seed), the paired comparisons of the family cancel it by seed, and the G2 floor at 0.70 sits
above the upper tail. Readers of the default table will see sub-0.5 single-arm accuracies; this
is why, and they are never a claim.

### 1.3 NARMA-10 (G4's task, run in the family; D013, D014)

- **Input**: $u_t \sim \text{Uniform}(0, 0.5)$.
- **Recurrence**: $y_{t+1} = 0.3 y_t + 0.05 y_t \sum_{i=0}^{9} y_{t-i} + 1.5 u_{t-9} u_t + 0.1$.
- **Metric**: NRMSE.
- **Implementation**: `src/qrc_thresher/tasks/narma10.py`.

## 2. Quantum Reservoir Architecture

Implementation in `src/qrc_thresher/reservoirs/windowed_qrc.py` (design (a), D010);
`reservoirs/pennylane_qrc.py` is the frozen $w = 1$, $\alpha = \pi$ reference.

### 2.1 Circuit Pattern

1. At every layer, qubit $j$ re-uploads $u_{t-(j \bmod w)}$ via $R_y(\alpha \, u)$, with 0 where the
   index is negative; $w$ is `reservoir.window` (1..n_qubits) and $\alpha$ is
   `reservoir.encoding_scale` (default $\pi$; a tuned hyperparameter, D011).
2. Fixed random $R_z(\theta_{d,j})$, $R_x(\phi_{d,j})$ drawn once from `default_rng(reservoir_seed)`
   at the given depth (each depth is its own draw, D011).
3. Ring topology entangling layer: CNOT between qubit $j$ and $(j+1) \bmod N$.
4. Repeat steps 1–3 for depth $L$.
5. Readout: $\langle Z_i \rangle$ for each qubit. Optionally $\langle Z_i Z_j \rangle$ for $i < j$.
6. Stack readouts to form feature matrix $X$ of shape $(T, F)$.
7. The harness readout on $X$: RidgeCV over `training.ridge_alphas` with `training.cv_folds`
   contiguous folds, fitted on rows $[\text{washout}, \text{train\_end})$.

The circuit hash covers the angles, $w$ (when $w > 1$), $\alpha$ (always) and any ablation
suffix (D010, D011).

### 2.2 Tuning Grid (D011)

Every tuned model gets the same budget, 60 configurations, each scored on 5 contiguous
validation blocks of the training rows after the washout with the harness readout (one feature
matrix per configuration; test rows never touched). Selection metric: `stm_memory` (STM),
accuracy (parity), NRMSE, lower is better (NARMA-10). The QRC grid is depth {1, 2, 3, 4, 5} ×
window {1, 2, 4} × $\alpha \in \{\pi/4, \pi/2, 3\pi/4, \pi\}$; the untuned default (depth 3, $\pi$,
$w = 2$; `design=default`) and today's circuit ($w = 1$; `design=default_w1`) are reported
beside the tuned design, never in its place. `qrc-thresher tune [TASK] --config` (TASK optional, D016 B6: without it the three tasks are tuned under one `sweep_id`) writes the
record `results/tuning/<config_hash>/<task>.json`; `run`, `ablation` and `baseline` deploy from
it and every row inherits its `sweep_id` and `tuning_record_sha`.

## 3. Classical Baselines

### 3.1 Feature Dimension Matching

**N_ESN = N_quantum_features** (NOT $2^N$).

- `z_only`: $F = N$.
- `z_and_zz`: $F = N + N(N-1)/2$.

### 3.2 Echo State Network (ESN)

$x_t = (1 - a)\,x_{t-1} + a \tanh(\rho \hat W x_{t-1} + s\,(W_{in} u_t + b_{in}))$ from $x_{-1} = 0$
(D009, D011): dense wiring, one draw per `reservoir_seed` ($\hat W$ normalised to spectral radius
1, then $W_{in} \sim U(-1, 1)$, then the input bias $b_{in} \sim U(-1, 1)$), $N = F$. Hyperparameter
grid (60 configurations; the readout penalty is not a grid entry, the harness readout is shared):

| Parameter | Values |
|-----------|--------|
| spectral_radius | 0.8, 0.9, 0.95, 0.99, 1.0 |
| input_scaling | 0.1, 0.5, 1.0 |
| leak_rate | 0.1, 0.3, 0.5, 1.0 |

Selection uses the validation blocks of the post-washout training rows only, on `stm_memory`
($k \ge 1$), accuracy or NRMSE. Test indices are NEVER used in hyperparameter selection. The
deployed ESN's weight hash equals the validated one. The untuned preset `esn_nonlinear` (with the
bias) is the reported default.

### 3.3 Random Kitchen Sinks (RKS)

$\phi(x_t) = \cos(W x_t + b)$ on the zero-padded input window $x_t = (u_t, \ldots, u_{t-d+1})$,
$W \in \mathbb{R}^{F \times d}$ with $W_{ij} \sim \mathcal{N}(0, \sigma^2/d)$, $b \sim \text{Uniform}(0, 2\pi)$,
stream `default_rng([reservoir_seed, 3])` (D010, D011, defect D13). Dimension $F$ matched to the
QRC features. Grid: $\sigma$ in 20 log-spaced values from 0.1 to 10 × $d \in \{1, 2, 4\}$. RKS runs
through `baseline` (task names `rks`, `rks_parity`, `rks_narma`), not through `ablation`.

## 4. Ablations (D010)

Matched to the reservoir: each inherits the per-pair `reservoir_seed`, readout, window,
encoding scale and re-upload schedule; only the tested factor changes. Under a tuned design the
ablation inherits design_TASK(pair) (`design=inherited`).

| Name | Description |
|------|-------------|
| phase_random | Fresh $R_z$, $R_x$ angles at every step, stream `default_rng([seed, 1])` |
| no_entangle | The CNOT ring removed (isolates entanglement under `z_only` only) |
| haar | Each layer's rotations and ring replaced by an independent Haar unitary, `default_rng([seed, 2])` |

## 5. Statistical Methodology (D013, D014)

The comparative gates G1, G2, G2.5, G3 and G4 are one pre-registered family
(`configs/gates/COMPARATIVE.v1.yaml`), evaluated as a unit on `configs/comparative.yaml` (12 seed
pairs):
- every comparison is paired on (config_hash, sweep_id, design, task_seed, reservoir_seed);
  unpaired, duplicated (non-identical), budget- or design-mismatched rows give
  INSUFFICIENT_EVIDENCE, nothing is truncated or pooled;
- per comparison: the mean difference, the one-sided paired t-test in the registered direction
  (the decision statistic), the Wilcoxon signed-rank test, a 95% BCa bootstrap CI (B = 2000, seed
  20260923) and $d_z$ (null at zero variance);
- Holm adjusts the five one-sided p-values together ($m = 5$; an INSUFFICIENT member contributes
  $p = 1$); a gate PASSes iff its adjusted $p \le 0.05$ and its floor holds (G2: accuracy > 0.70;
  G4: NRMSE < 0.60; G1 also needs the tuned design's G0.7 v1 PASS);
- a significant margin in the baseline's favour is reported as "baseline better" with its
  two-sided p; the untuned defaults are reported beside every member with the same statistics
  and never flip a verdict.
- 12 pairs give 80% power for $d_z \ge 0.77$ one-sided at $\alpha = 0.05$ before Holm.

`qrc-thresher gate family --config configs/comparative.yaml` writes
`results/gates/COMPARATIVE.v1.<stamp>.json` (the record) and one `<gate>.<stamp>.json` view per
member, never overwriting; each carries config_hash, sweep_id, the git commit, the measurement
label and the protocol hash. G0.7 (`configs/gates/G0.7.v1.yaml`) is unchanged.

## 6. Measurement Model (D004, D018)

Every QRC and ablation number, and every G0.7 number on a quantum model, is computed from exact
expectation values (config `measurement.model: exact`, the only accepted config value; QRC and
ablation rows record it in `measurement_model`, ESN and RKS rows record `classical`, D018). Each
report labels them **"exact (oracle upper bound)"**: a device estimates each expectation from a
finite number of shots, so exact values bound from above what hardware could measure, and they are
never a headline claim (D004). Finite-shot measurement has not been built; D004's commitment that it
lands before any comparative sweep was not met by run 4, and the finite-shot evidence is pending
under a v2 protocol (D017's ledger, CP5a ruling 1).

Classical comparators carry a label of their own kind, **"classical, no measurement cost"** (the ESN
and RKS rows; on the gate's console line and in the scorecard also the ESN presets of G0.7, whose
files record exact by protocol, §7; D018): a classical model has no shot budget, so the oracle
qualifier does not apply to it. Rows written before D018 recorded `exact` on classical arms; the
gate and scorecard writers of CP5 label each arm by its kind, and `MEASUREMENT_LABELS` in
`config.py` is the one table of labels.

## 7. G0.7 v1: the Memory Sanity Gate (D005, D008)

Protocol `configs/gates/G0.7.v1.yaml`, registered before any evaluation that used it and never
edited (a changed protocol is a new version file; `tests/test_preregistration.py` pins its hash).
On every seed pair of the config (at least three; `rule: every_seed`), the reservoir features go
through the harness readout (one RidgeCV fit per clause over `training.ridge_alphas`, contiguous
folds, intercept; the gate refuses a config whose `ridge_alphas` or `cv_folds` differ) and each
clause is tested against a permutation null of 200 permutations that shuffle the train rows and the
test rows independently and refit the readout every time (`default_rng([20260922, clause,
task_seed, reservoir_seed])`, train order drawn before test order). The p-value is
$p = (1 + \#\{\text{null} \ge \text{observed}\}) / (n + 1)$ and a seed passes a clause at $p \le 0.05$.

- **STM clause** (length 500, 70/30 split, washout 50, `delay_max` 20): the statistic is
  $S = \sum_{k=1}^{20} r^2_k$, the held-out squared Pearson correlation between the prediction and
  $u_{t-k}$; $k = 0$ is reported and never counted.
- **Parity clause** (window 2, same length, split and washout): held-out accuracy of the ridge
  output thresholded at 0.5.
- **Degenerate seeds** fail their clause without a score: every training feature column with
  std $\le 10^{-12}$, any held-out prediction column with std $\le 10^{-12}$, or a non-finite
  feature, prediction or statistic.

The gate PASSes only if every seed passes both clauses. `qrc-thresher gate G0.7 --config PATH
[--model M] [--tuning-config PATH]` writes `results/gates/G0.7.<model>.<stamp>.json` (with
`config_hash` and `git_commit_hash` at the top level, files written after D018) and the forgetting-curve figure beside it;
`--model tuned_qrc` evaluates the tuned design_STM of the tuning config on alpha_lite.yaml's pairs,
which is G1's clause (a).
Its measurement label is `exact (oracle upper bound)` by protocol, also for the ESN presets
(`esn_linear`, `esn_nonlinear`), which the gate's console line and the scorecard show by kind (the
scorecard with a footnote).

## 8. The Per-Member Claims (D014)

Each family member is one paired comparison on the tuned designs of D011, decided under §5.
"Tuned" means the design that `tune` selected for that task and seed pair under the matched 60-configuration
budget; the registered metric and direction are fixed in `configs/gates/COMPARATIVE.v1.yaml`.

| Member | Claim tested | Arm A | Arm B | Metric, direction | Floor |
|---|---|---|---|---|---|
| G1 (a) | The tuned STM design has memory at all | design_STM | G0.7 v1 permutation null | G0.7 v1 PASS on alpha_lite.yaml's 3 pairs (COMPARATIVE.v1 clause_a) | — |
| G1 (b) | Removing entanglement costs parity accuracy | design_STM | its inherited `no_entangle` ablation (z_only) | parity accuracy, QRC greater | — |
| G2 | Parity is not trivially solved by random projection | design_parity | tuned RKS | parity accuracy, QRC greater | mean accuracy > 0.70 |
| G2.5 | The circuit structure, not access to a $2^n$ unitary, carries the memory | design_STM | its inherited Haar ablation (z_only) | `stm_memory` ($k \ge 1$), QRC greater | — |
| G3 | The QRC beats a matched classical reservoir on memory | design_STM | tuned ESN | `stm_memory` ($k \ge 1$), QRC greater; `mc_k0` reported, never compared | — |
| G4 | The QRC beats a matched classical reservoir on NARMA-10 | design_NARMA | tuned ESN | NRMSE, QRC less | mean NRMSE < 0.60 |

G1 is PASS only if (a) and (b) both hold. G2 is a baseline comparison, not an entanglement
comparison (D014). D014 records that G1(b)'s metric was chosen after the CP3b G0.7 findings were
known, so it is not presented as pre-registered blind. The STM-memory margin of design_STM over
`no_entangle` is reported beside (b) and not gated.

**The default table.** Beside every tuned comparison the same statistics are reported for the
untuned default design (`design=default`: depth 3, scale $\pi$, $w = 2$), never gated and never in
the Holm family: G1(b) and G2.5 pair it with the matched ablation of that default design
(`ablation ... --design default`); G3 and G4 pair it with the default ESN (`esn_nonlinear` with the
bias, run with `baseline ... --design default`); G2 pairs it with the tuned RKS, which has no
default. The $w = 1$ circuit (`design=default_w1`) is reported as a mean beside it. A member that
passes on the default and fails on the tuned design is FAIL.

## 9. Diagnostics (D011)

The CP3b failure that D011 records: with $R_y(\pi u)$ encoding on inputs
$u_t \sim \text{Uniform}(-1, 1)$ the single-qubit readouts are harmonics of $\pi u$ whose mean over
the input distribution is zero, so the untuned $\alpha = \pi$ features carry little linear memory
(DECISIONS.md D011, rationale). D011's response was to make the encoding scale a registered
hyperparameter ($\alpha \in \{\pi/4, \pi/2, 3\pi/4, \pi\}$) rather than a hand-picked fix. The
per-delay $r^2_k$ curve of G0.7 against its null quantile (`reporting.null_quantile` 0.95) is the
diagnostic figure (`*.forgetting_curve.png`); the tuned design_STM is chosen per seed pair, and each pair's depth, window
and scale are recorded in the tuning record (`tuning/stm.json`) and, for the three G0.7 pairs, in
the G0.7 file's `model_details.designs` (both under `docs/evidence/8ed2df4/`). D014 records the
CP3b G0.7 STM statistics of the untuned w = 2 reservoir and of its no-entangle ablation; a
quantitative account of the harmonic cancellation itself is a gap, not a result.

## 10. Ablation Rows and Their Task (D019, item A.4)

An ablation row's `task_name` is `ablation:<name>` and does not say which task was ablated; its
task is its metric's task, read through `task_names.task_of_metric(primary_metric_name)`
(`stm_memory` → STM, `accuracy` → parity, `nrmse` → NARMA-10; the map is a bijection). Its
deployment (the tuned design it inherited, or the default) is read from `circuit_hash`, which the
family evaluator (by expected circuit hash) and `summary` (`deployment_label`) resolve through the tuning record. A schema
field for the ablated task is deferred to a future schema and protocol; COMPARATIVE.v1 was
evaluated with `primary_metric_name` as the discriminator and run 4's rows stay valid inputs.
