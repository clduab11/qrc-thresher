# References

## Primary References

### QRC-Lab (arXiv:2602.03522, Feb 2026)

Nearest architectural neighbor. Uses the same 4 qubits, depth 3, ring entanglement, Pauli-Z
readout, ridge regression on STM + parity + NARMA-10 tasks.

**Differentiation memo required by G1** before arXiv submission. Key differentiators:
- This repo emphasizes falsification-first methodology over result maximization.
- Full ablation suite (phase-randomized, entanglement-suppressed, Haar-random) is mandatory.
- All gates must pass before any positive claim is made.
- Negative results are treated as a publishable, valuable outcome.

Do NOT copy code from QRC-Lab without verifying license compatibility.

### Published Bar: arXiv:2510.25183

Reports ESN NRMSE = 0.185 vs QRC NRMSE = 0.485 on NARMA-10.
This is the expected performance gap on NARMA. Frame honest reporting accordingly.

## Secondary References

- **ReservoirPy documentation** — primary ESN reference for the `reservoirpy` library.
- **PennyLane qml docs** — primary QRC primitives reference.
- **Qiskit 2.x migration guide** — for G5 cross-validation and Phase 2 IBM Runtime.

## Forbidden References

- QHACK23_QRC: unlicensed. Do not import or reference its code.

## License Compatibility Note

All code in this repo is original or derived from Apache-2.0 / MIT compatible libraries.
Any third-party reservoir code requires explicit license check before inclusion.
