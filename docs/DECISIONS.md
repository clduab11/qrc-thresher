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
