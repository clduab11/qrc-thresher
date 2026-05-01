# qrc-thresher Build Specification

This document is the canonical in-repo build specification for the qrc-thresher project.

## 1. Mission

Build a reproducible, falsification-first benchmark harness that tests one narrow scientific
question:

> Do small simulated quantum reservoirs generate temporal features that are useful, in a measurable
> way, beyond what tuned classical baselines and ablation controls can produce?

The methodology IS the contribution. Negative results are a publishable, valuable outcome.

## 2. Hard Constraints

- CPU-first (Phase 1). No CUDA, cuQuantum, JAX-GPU, or TPU paths.
- Synthetic data only in Phase 1.
- No web servers, APIs, or UI frameworks.
- Cost cap: $0–$50 total for Phase 1.
- Python 3.11+.

See the full brief for the complete constraint list.

## 3. Repository Structure

```
qrc-thresher/
  src/qrc_thresher/
    cli.py                     # Click-based CLI
    config.py                  # Pydantic v2 config models
    tasks/                     # STM, temporal parity, NARMA-10
    reservoirs/                # PennyLane QRC, Qiskit crosscheck, ablations
    baselines/                 # ESN, random kitchen sinks, GRU stub
    metrics/                   # MC, NRMSE, accuracy, stats, timing
    proof/                     # Run manifests, health checks
    viz/                       # Matplotlib plots
  tests/                       # pytest test suite
  configs/alpha_lite.yaml      # Phase 1 default config
  results/                     # Run artifacts (tracked)
```

## 4. Dependencies

See `pyproject.toml` for pinned versions. Key packages:
- `pennylane>=0.44,<0.45` + `pennylane-lightning`
- `qiskit[all]>=2.3,<3` + `qiskit-aer`
- `scikit-learn>=1.4`, `reservoirpy>=0.3.11`
- `numpy>=1.26`, `scipy>=1.11`, `pandas>=2.1`
- `pydantic>=2.6`, `click>=8.1`

## 5. Decision Gates

| Gate | Condition |
|------|-----------|
| G0   | health command passes |
| G0.5 | PennyLane↔Qiskit expectation values match within 1e-6 |
| G1   | QRC STM MC > 1.0 AND >20% above entanglement-suppressed ablation |
| G2   | Parity accuracy >70% at delay 3, random features <60% |
| G2.5 | Full QRC outperforms Haar-random by ≥1 SE on STM or parity |
| G3   | QRC matches/exceeds parameter-matched best ESN within 1 SE |
| G4   | NARMA NRMSE < 0.60 AND within 2x tuned ESN |
| G5   | PennyLane/Qiskit match within 1e-6 on full circuit |

## 6. Proof Layer

Every run writes a schema v1.1 manifest to `results/runs.csv` and updates
`results/cumulative_compute.json` atomically.

## 7. Phases

- **Phase 1**: All Section 7 tasks, gates G0–G5. CPU only. This repo.
- **Phase 1.5**: NARMA-10, GRU baseline (after G3 passes).
- **Phase 2**: IBM Runtime, hardware execution (after Phase 1.5 passes).
- **Phase 3**: Q-CTRL integration (only if G6 fails due to noise).
