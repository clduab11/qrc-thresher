# Methodology

## 1. Tasks

All tasks are synthetic, deterministic, and seeded via `numpy.random.Generator`.

### 1.1 Short-Term Memory (STM)

- **Input**: $u_t \sim \text{Uniform}(-1, 1)$, length $T$.
- **Targets**: $y_t^{(k)} = u_{t-k}$ for $k \in [0, K]$, with $K$ typically 20.
- **Metric**: Memory Capacity $\text{MC} = \sum_k \text{corr}(\hat{y}^{(k)}, y^{(k)})^2$.
- **Train/test split**: Chronological (no shuffling). Default 70/30.
- **Implementation**: `src/qrc_thresher/tasks/stm.py`.

### 1.2 Temporal Parity / XOR

- **Input**: Binary $u_t \in \{0, 1\}$, length $T$.
- **Target**: $y_t = \text{XOR}(u_{t-d+1}, \ldots, u_t)$ for window $d$.
- **Metric**: Classification accuracy at delay $d$.
- **Default**: $d \in \{1, 2, 3, 5\}$.
- **Implementation**: `src/qrc_thresher/tasks/temporal_parity.py`.

### 1.3 NARMA-10 (Phase 1.5, gated behind G3)

- **Input**: $u_t \sim \text{Uniform}(0, 0.5)$.
- **Recurrence**: $y_{t+1} = 0.3 y_t + 0.05 y_t \sum_{i=0}^{9} y_{t-i} + 1.5 u_{t-9} u_t + 0.1$.
- **Metric**: NRMSE.
- **Implementation**: `src/qrc_thresher/tasks/narma10.py`.

## 2. Quantum Reservoir Architecture

Implementation in `src/qrc_thresher/reservoirs/pennylane_qrc.py`.

### 2.1 Circuit Pattern

1. Angle-encode $u_t$ via $R_y(\pi u_t)$ on each qubit.
2. Fixed random $R_z(\theta_i)$, $R_x(\phi_i)$ drawn once at construction (seeded by `reservoir_seed`).
3. Ring topology entangling layer: CNOT between qubit $i$ and $(i+1) \bmod N$.
4. Repeat steps 1–3 for depth $d$.
5. Readout: $\langle Z_i \rangle$ for each qubit. Optionally $\langle Z_i Z_j \rangle$ for $i < j$.
6. Stack readouts to form feature matrix $X$ of shape $(T, F)$.
7. Ridge regression on $X$ to predict targets.

### 2.2 Initial Parameter Sweep

| Parameter | Values |
|-----------|--------|
| n_qubits  | 4, 6, 8 |
| depth     | 2, 3, 4 |
| seeds     | 3 initially; 5 for final reporting |
| readout   | z_only first; z_and_zz if needed |
| ridge_alpha | {1e-8, 1e-6, 1e-4, 1e-2, 1, 100} |

## 3. Classical Baselines

### 3.1 Feature Dimension Matching

**N_ESN = N_quantum_features** (NOT $2^N$).

- `z_only`: $F = N$.
- `z_and_zz`: $F = N + N(N-1)/2$.

### 3.2 Echo State Network (ESN)

Hyperparameter grid:

| Parameter | Values |
|-----------|--------|
| spectral_radius | 0.8, 0.9, 0.95, 0.99, 1.0 |
| input_scaling | 0.1, 0.5, 1.0 |
| leak_rate | 0.1, 0.3, 0.5, 1.0 |
| ridge_alpha | 1e-8, 1e-6, 1e-4, 1e-2, 1.0 |

CV is performed on training data only. Test indices are NEVER used in hyperparameter selection.

### 3.3 Random Kitchen Sinks (RKS)

$\phi(u) = \cos(Wu + b)$, $W \sim \mathcal{N}(0, \sigma^2/d)$, $b \sim \text{Uniform}(0, 2\pi)$.
Dimension matched to QRC features.

## 4. Ablations

| Name | Description |
|------|-------------|
| phase_random | Random phases at each time step |
| no_entangle | Single-qubit rotations only, no CNOT |
| random_features | Classical random projection (also a baseline) |
| haar | Haar-random unitary $U \sim \text{Haar}(2^N)$ |

## 5. Statistical Methodology

At G3 and beyond:
- Paired t-test or Wilcoxon signed-rank test, $n \geq 5$ seeds minimum.
- Report mean, std, p-value, and Cohen's $d$.
- Bootstrap 95% CIs over 1000 resamples for headline metrics.

Gate evaluation writes `results/gates/<name>.json` with decision, evidence, run_ids, p-values.
