# qrc-thresher

[![Phase 1](https://img.shields.io/badge/Phase-1%20In%20Progress-yellow)](docs/BUILD_SPEC.md)
[![Pre-registered](https://img.shields.io/badge/Gates-pre--registered-blue)](configs/gates/)
[![License](https://img.shields.io/badge/License-Apache--2.0-green)](LICENSE)

**A benchmark built to answer one narrow question honestly: do small quantum circuits, used as
"reservoirs" for learning from time series, produce anything a well-tuned classical model of the
same size cannot?**

Most results in quantum machine learning are hard to trust — not because the physics is wrong,
but because the comparison is. The classical baseline is undertuned, the success criteria are
chosen after the data are in, and nobody else can rerun the experiment. This project fixes the
method before it looks at the answer: the pass/fail rules are written down and locked before the
experiment runs, every classical baseline gets exactly the same tuning budget as the quantum
model, the control experiments are designed so a spurious result retracts itself, and every
number carries a manifest that lets anyone reproduce it to every recorded digit. A negative result is a
valid, publishable outcome here. No quantum advantage is claimed — the point is to find out, and
to be unable to fool ourselves along the way.

## What this is, in plain English

A *reservoir computer* is a fixed, random dynamical system whose evolving state is read out by a
simple linear model. It is cheap to train because only the readout is learned. A *quantum*
reservoir uses a small quantum circuit as that dynamical system, and the literature claims the
circuit produces richer temporal features than a classical one. The claim is plausible. It is
also, so far, poorly tested.

qrc-thresher tests it under rules that make the usual escape hatches unavailable:

- **Pre-registered gates.** Every decision rule — which metric, which direction, what threshold,
  which statistical test, how many seeds — is written to a versioned file whose hash is pinned by
  a test *before* any data that could be judged by it exists. Changing a rule means a new version
  file and a logged decision, never an edit.
- **Matched budgets.** The quantum reservoir, an echo state network and random Fourier features
  each get 60 hyperparameter configurations, selected the same way (contiguous validation blocks
  of the training data, one shared readout). Test data never touch the selection. If the quantum
  model wins, it is not because it was the only one anybody tuned.
- **Self-falsifying controls.** The full circuit is run against versions of itself with the
  entanglement removed, the phases randomized at every step, or each layer replaced by a random
  unitary. If a control matches the full circuit, the positive result is void by construction.
- **Paired statistics, one family.** Every comparison is paired by seed; nothing is truncated or
  pooled. Five gates are decided together with a Holm correction, with bootstrap confidence
  intervals and effect sizes reported alongside. A missing pair makes a gate *inconclusive*, never
  quietly smaller.
- **Reproducibility that has been checked.** Every result row records the commit, the config hash,
  the circuit hash, the seeds and the environment. The same locked dependencies reproduced the
  same gate statistics to every recorded digit on Windows and on Linux, and two independent runs
  of the registered sweep agreed row for row.

The tasks are synthetic and standard — short-term memory, temporal parity (XOR over a window) and
NARMA-10 — on 4-qubit simulated circuits against a 4-unit echo state network and 4-feature random
kitchen sinks, twelve seed pairs each. The simulation uses exact expectation values, which is an
*oracle upper bound*: a real device, with finite measurements and noise, can only do worse, and
every report says so.

## Where it stands

The harness is complete and the first comparative protocol (`COMPARATIVE.v1`) is frozen. The
first registered sweep is the next step. There is no comparative result yet, and when there is,
it will be published as `pass`, `fail` or `inconclusive` — the same way whichever way it goes.

## How it is built

Three roles, written into the decision log: a human principal investigator who makes every
scientific and scope decision; a *builder* AI agent that writes code and tests in a sandbox with no
access to git; and a *referee* that reviews every change, commits it, and runs the authoritative
tests in a separate environment built from the lock file. Every decision since the project's
redirection in September 2026 is recorded in [docs/DECISIONS.md](docs/DECISIONS.md) (D001–D016),
including the ones that were wrong the first time. This is as much a working example of
accountable AI-assisted research as it is a quantum computing benchmark.

---

## Installation

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/clduab11/qrc-thresher.git
cd qrc-thresher
uv sync --frozen          # the locked environment CI and the referee use
```

Without uv: `pip install -e .` (optional extras: `.[docs]` for Sphinx, `.[observability]` for
OpenTelemetry).

## Quick start

```bash
uv run qrc-thresher health                                        # G0: environment and harness sanity
uv run qrc-thresher run stm --config configs/alpha_lite.yaml      # every seed pair of the config
uv run qrc-thresher gate G0.7 --config configs/alpha_lite.yaml    # memory sanity gate, timestamped JSON
```

The registered comparative sweep, in order:

```bash
uv run qrc-thresher tune --config configs/comparative.yaml        # QRC, ESN and RKS under one budget, one sweep id
uv run qrc-thresher run stm --config configs/comparative.yaml     # ... and parity, narma, --design-task stm, --design default
uv run qrc-thresher ablation no_entangle stm --config configs/comparative.yaml
uv run qrc-thresher baseline stm --config configs/comparative.yaml
uv run qrc-thresher gate G0.7 --config configs/alpha_lite.yaml --model tuned_qrc --tuning-config configs/comparative.yaml
uv run qrc-thresher gate family --config configs/comparative.yaml # G1, G2, G2.5, G3, G4 decided together
uv run qrc-thresher summary --phase phase1
```

Appendix F of [docs/BUILD_SPEC.md](docs/BUILD_SPEC.md) lists the full sequence. Other commands:
`ablation NAME TASK`, `baseline TASK`, `plugins`, `perf`, `noise-sweep`, `plot`. Global flags
`--verbose`, `--json-logs`, `--trace`.

## How a sweep works

1. `tune` scores every configuration of every model on validation blocks of the training rows
   and writes a tuning record (`results/tuning/<config_hash>/<task>.json`) with every score, the
   winner, the seeds, and its own SHA-256.
2. `run`, `ablation` and `baseline` deploy the validated designs from that record — a deployment
   is refused unless the rebuilt model's hash equals the record's — and append one row per seed
   pair to `results/runs.csv` (manifest schema 1.4: 31 provenance columns).
3. `gate G0.7` checks that the tuned reservoir actually remembers (a permutation-tested memory
   statistic plus a parity clause), against a frozen protocol.
4. `gate family` pairs every arm by seed, computes the five registered comparisons in one pass,
   applies Holm, and writes `results/gates/COMPARATIVE.v1.<stamp>.json` — the record — plus one
   view per gate. Files are never overwritten.

## Architecture

A Click CLI dispatching to handlers in `src/qrc_thresher/commands/`. Tasks, reservoirs, baselines,
gates, visualisations and noise models are discoverable through Python entry points
(`qrc_thresher.*`) and a plugin registry, so third parties can add their own without forking.

- `reservoirs/windowed_qrc.py` — the windowed-input quantum reservoir (PennyLane) and its matched
  ablations; `reservoirs/qiskit_crosscheck.py` — an independent Qiskit implementation the G0.5
  gate compares against at 1e-6.
- `baselines/esn.py`, `baselines/random_features.py` — the classical baselines, dimension-matched
  to the quantum feature count.
- `tuning.py`, `deploy.py` — one tuner for every model; deployment by verified hash.
- `metrics/paired.py`, `gates/comparative.py`, `gates/g07.py` — the paired statistics and the
  gate evaluators, each bound to a frozen protocol file under `configs/gates/`.
- `proof/run_manifest.py`, `db.py` — the manifest schema, the runs.csv writer (file-locked,
  header-guarded) and an optional SQLite store.

## Phase 1 status

- [x] Deterministic, seeded task generators: STM, temporal parity, NARMA-10
- [x] Windowed-input quantum reservoir with a tunable encoding scale; w = 1 kept as the memoryless
      negative control; independent Qiskit cross-check (G0.5, 34 cases)
- [x] Matched ablations: no-entangle, phase-random, Haar-random
- [x] ESN (dense wiring, input bias) and windowed random kitchen sinks, dimension-matched
- [x] One tuner, 60 configurations per model, validation-block selection, hash-verified deployment
- [x] Manifest schema 1.4 with design labels, sweep id and tuning-record hash on every row
- [x] Gates as frozen, hash-pinned protocol files: `G0.7.v1`, `COMPARATIVE.v1`
- [x] Paired statistics: one-sided paired t, Wilcoxon, BCa bootstrap CI, d_z, Holm over the family
- [x] Executable gates G0, G0.5, G0.7, the family (G1, G2, G2.5, G3, G4), G5, G6, G7
- [x] Cross-platform reproduction of gate statistics (Windows ↔ Linux, same lock file)
- [x] CI matrix on Python 3.11 / 3.12 / 3.13 from the lock file
- [ ] The first registered comparative sweep and its verdicts
- [ ] Finite-shot measurement model (exact expectation values are the oracle upper bound until then)
- [ ] Carried-state reservoir designs (Phase 2)

## Documentation

- [docs/BUILD_SPEC.md](docs/BUILD_SPEC.md) — the build specification (v1.1)
- [docs/METHODOLOGY.md](docs/METHODOLOGY.md) — tasks, circuit, baselines, statistics, disclosures
- [docs/DECISIONS.md](docs/DECISIONS.md) — the append-only decision log, D001–D016
- [configs/gates/](configs/gates/) — the frozen gate protocols
- [docs/REFERENCES.md](docs/REFERENCES.md) — literature

## License

- **Apache-2.0** — code in `src/` and `tests/` ([LICENSE](LICENSE))
- **CC BY 4.0** — documentation in `docs/` ([docs/LICENSE-DOCS](docs/LICENSE-DOCS))
- **CC0 1.0** — synthetic data outputs in `data/` ([data/LICENSE-DATA](data/LICENSE-DATA))

## Citation

A preprint follows the first registered sweep; BibTeX will be added then. Maintained by
[@clduab11](https://github.com/clduab11).
