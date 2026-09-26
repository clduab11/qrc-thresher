# References

Each arXiv entry was checked against its abstract page on 2026-09-25 (D018, item F.4); primary
sources only: an arXiv abstract page with its version number is citable; a newsletter or a summary
of a paper is not.

## The Published Bar

### arXiv:2510.25183v1 — verified 2026-09-25 against the arXiv abstract page

- **Pinned version:** v1. The numbers this repository frames itself against — ESN NRMSE 0.185
  against QRC NRMSE 0.485 on NARMA-10 — are v1's Table I ("Benchmark metrics (NARMA-10 split)":
  ESN (N=300) NRMSE 0.185, QRC NRMSE 0.485), checked in the v1 PDF.
- **Title / authors / date:** "Sustainable NARMA-10 Benchmarking for Quantum Reservoir Computing".
  Avyay Kodali, Priyanshi Singh, Pranay Pandey, Krishna Bhatia, Shalini Devendrababu, Srinjoy
  Ganguly. v1, 27 Oct 2025.
- **Later versions:** v2 (4 Sep 2026) is retitled "Sustained Performance and Energy Accounting for
  Nonlinear Forecasting Across Classical and Simulated Quantum Models" and reaches different
  conclusions; it is not the bar. A move to v2 is a decision in `docs/DECISIONS.md`, not an edit
  here.
- This file is the one place the bar is stated (BUILD_SPEC §25 points here and does not repeat
  the numbers). "Honest reporting" in the harness means: a QRC NARMA-10 NRMSE is reported against
  the tuned ESN of the same run under a matched budget (G4, D014), and the published pair above is
  context, not a comparator.

## Matched-Baseline Literature

### arXiv:2607.09905v1 (T. Pandey, Jul 2026) — verified 2026-09-25 against the arXiv abstract page

- Reported finding, as far as this repository relies on it: when classical baselines are given the
  same size and tuning budget, two commonly cited QRC advantages disappear at this scale (exact
  simulations of up to eleven qubits).
- **Title / author / date / version:** "When Classical Baselines Are Tuned as Carefully as the
  Quantum Model, Does Quantum Reservoir Computing Still Win?". Tushar Pandey (sole author). v1,
  10 Jul 2026.
- Why it is here: it states the bar the harness was rebuilt to meet in D009 and D011 — every model
  tuned under the same budget over its own hyperparameters, chosen on validation blocks with the
  shared readout.

### arXiv:2607.18552v1 (survey) — verified 2026-09-25 against the arXiv abstract page

- Reported finding, as far as this repository relies on it: current results do not establish a
  broad quantum advantage over well-matched classical reservoirs.
- **Title / authors / date / version:** "Quantum Reservoir Computing: Recent Advances and Future
  Directions". Shehbaz Tariq, Muhammad Talha, Arshid Ali, Muhammad Diyan, Symeon Chatzinotas. v1,
  20 Jul 2026.
- Why it is here: the family's negative verdicts on G3 and G4 (run 4, D019) are read against this
  background, not as a surprise.

## Architectural Neighbours

### QRC-Lab (arXiv:2602.03522v1, Feb 2026) — verified 2026-09-25 against the arXiv abstract page

- **Title / author / date / version:** "QRC-Lab: An Educational Toolbox for Quantum Reservoir
  Computing". Anderson Fernandes Pereira dos Santos. v1, 3 Feb 2026.
- The abstract confirms the tasks (short-term memory, temporal parity (XOR), NARMA10). The circuit
  shape in BUILD_SPEC §25 (ring entanglement, Pauli-Z readout, ridge regression) is not verified
  against the paper.

An architectural neighbour whose method differs. The harness's own method, which this file does not
compare with QRC-Lab's:

- the input enters through a window of past inputs re-uploaded at every layer, with the window and
  the encoding scale tuned as hyperparameters (D010, D011);
- every classical comparator is tuned under the same 60-configuration budget as the QRC (D011);
- ablations are matched to the reservoir realisation they test (D010);
- verdicts come from one pre-registered family of paired comparisons with Holm adjustment (D013,
  D014), and every QRC number is labelled by its measurement model (D004, D018).

Whether QRC-Lab shares any of these is **not** claimed here. A differentiation memo against QRC-Lab
is required before any arXiv submission (BUILD_SPEC §25). Do not copy code from QRC-Lab without a
licence check.

## Secondary References

Documentation, not papers; cited by name only.

- **PennyLane documentation** — QRC primitives (`default.qubit`).
- **Qiskit 2.x documentation** — the independent G0.5 cross-check build and any later hardware
  path.
- ReservoirPy is no longer a reference: the ESN is a dense numpy implementation of the harness
  (D009), `reservoirpy` left the health check's package list (D018), and is kept only as a test
  cross-reference of the numpy ESN (D018).

## Forbidden References

- QHACK23_QRC: unlicensed. Do not import or reference its code.

## License Compatibility Note

All code in this repo is original or derived from Apache-2.0 / MIT compatible libraries.
Any third-party reservoir code requires explicit license check before inclusion.
