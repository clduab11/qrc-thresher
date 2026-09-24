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
- **Default**: $d \in \{1, 2, 3, 5\}$.
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

### 1.3 NARMA-10 (Phase 1.5, gated behind G3)

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
beside the tuned design, never in its place. `qrc-thresher tune TASK --config` writes the
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
