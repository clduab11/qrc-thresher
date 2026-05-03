# qrc-thresher

[![Phase 1](https://img.shields.io/badge/Phase-1%20In%20Progress-yellow)](docs/BUILD_SPEC.md)
[![Augmentation 2026](https://img.shields.io/badge/Augmentation-2026-blue)](docs/MASTER_AUGMENTATION_PLAN_2026.md)

Falsification-first benchmark harness for quantum reservoir computing (QRC) vs classical baselines.
This is a reproducible, falsification-first benchmark that tests one narrow scientific question:

> Do small simulated quantum reservoirs generate temporal features that are useful, in a measurable
> way, beyond what tuned classical baselines and ablation controls can produce?

The methodology IS the contribution. Negative results are a publishable, valuable outcome.

## Installation

### WSL2 / Ubuntu 22.04+ / macOS / Linux

```bash
git clone https://github.com/cld-maindev/qrc-thresher.git
cd qrc-thresher
pip install -e .
```

The project is also reproducibly installable via [uv](https://github.com/astral-sh/uv):

```bash
uv pip sync uv.lock
```

Optional extras:

```bash
pip install -e ".[docs]"            # Sphinx documentation toolchain
pip install -e ".[observability]"   # OpenTelemetry tracing exporters
```

### Quick start

```bash
qrc-thresher health
```

Run a benchmark:

```bash
qrc-thresher run stm --config configs/alpha_lite.yaml
qrc-thresher run stm --config configs/alpha_lite.yaml --workers 4   # parallel seeds
```

Other commands:

```bash
qrc-thresher ablation no_entangle --config configs/alpha_lite.yaml
qrc-thresher gate G1                    # evaluate a decision gate
qrc-thresher plugins                    # list discovered plugins
qrc-thresher perf --iterations 5        # micro-benchmarks
qrc-thresher noise-sweep                # Aer noise-model sweep scaffold
qrc-thresher summary --phase phase1     # aggregate runs.csv into markdown
```

Global flags `--verbose`, `--json-logs`, and `--trace` enable DEBUG logging,
machine-parseable JSON log lines, and OpenTelemetry console tracing
respectively.

## Architecture

The CLI is a thin Click dispatcher that delegates to per-command handlers in
`src/qrc_thresher/commands/`. Tasks, reservoirs, baselines, gates, viz, and
noise models are all discoverable via Python entry-points (group prefix
`qrc_thresher.*`) and managed by the plugin registry in
`src/qrc_thresher/plugins/`. Third parties can add new tasks, reservoirs,
gates, or visualisations by exposing entry-points in their own packages —
no fork required.

Key subsystems:

- `qrc_thresher.engine` — parallel multi-seed execution via
  `ProcessPoolExecutor`, with file-locked atomic appends to `results/runs.csv`.
- `qrc_thresher.db` — optional SQLite (WAL-mode) result store with CSV
  import/export, for analyses that outgrow flat files.
- `qrc_thresher.observability` — structured (JSON) logging plus optional
  OpenTelemetry tracing.
- `qrc_thresher.metrics.perf` — lightweight wall-clock micro-benchmarks
  surfaced via `qrc-thresher perf`.
- `qrc_thresher.reservoirs.noise_models` — Qiskit Aer depolarizing /
  relaxation noise-model builders for hardware-realism sweeps.
- `qrc_thresher.reservoirs.stateful_qrc` — stateful (memory-carrying) QRC
  feature extractor for studies that require persistent reservoir state.

## Phase 1 Status

- [x] Package scaffold
- [x] STM and temporal parity task generators (deterministic, seeded)
- [x] PennyLane QRC (angle encoding, ring entanglement, Pauli-Z readout)
- [x] Stateful QRC variant with carried reservoir state
- [x] ESN and random kitchen sinks baselines
- [x] Ablations: phase-randomized, entanglement-suppressed, Haar-random
- [x] Metrics: MC, NRMSE, accuracy, bootstrap CIs, BCa CIs, paired tests,
      Holm–Bonferroni correction, power analysis
- [x] Proof layer: run manifests (schema v1.1), health checks, NARMA-10
      recurrence verification
- [x] CLI: health, run (parallel), ablation, gate, plot, summary, plugins,
      perf, noise-sweep
- [x] Plugin SDK with entry-point discovery (tasks, reservoirs, baselines,
      gates, viz, noise models)
- [x] Parallel execution engine and SQLite result store
- [x] Pre-registered gate thresholds in config (`gates:` section)
- [x] Config overlay system (`load_config_with_overlays`)
- [x] Executable gate evaluators G0, G0.5, G1, G2, G2.5, G3, G4, G5, G6, G7
- [x] CI matrix on Python 3.11 / 3.12 / 3.13 with `uv` lock-file installs;
      health and gate steps now block on failure
- [x] Visualisations: STM/MC, comparison, delay heatmap, runtime breakdown,
      gate decision tree, metric correlation matrix
- [x] Sphinx documentation scaffold (`docs/sphinx/`)
- [ ] Full G1–G7 evaluations against benchmark sweeps

See [docs/MASTER_AUGMENTATION_PLAN_2026.md](docs/MASTER_AUGMENTATION_PLAN_2026.md)
for the roadmap driving the recent augmentations.

## License

Triple-license:
- **Apache-2.0** — all code in `src/` and `tests/` (see [LICENSE](LICENSE))
- **CC BY 4.0** — documentation in `docs/` (see [docs/LICENSE-DOCS](docs/LICENSE-DOCS))
- **CC0 1.0** — synthetic data outputs in `data/` (see [data/LICENSE-DATA](data/LICENSE-DATA))

## Documentation

- [BUILD_SPEC.md](docs/BUILD_SPEC.md) — full build specification
- [METHODOLOGY.md](docs/METHODOLOGY.md) — task definitions, circuit spec, statistical methodology
- [MASTER_AUGMENTATION_PLAN_2026.md](docs/MASTER_AUGMENTATION_PLAN_2026.md) — 2026 augmentation roadmap
- [REFERENCES.md](docs/REFERENCES.md) — reference literature
- [DECISIONS.md](docs/DECISIONS.md) — decision log
- `docs/sphinx/` — API reference scaffold (build with `sphinx-build`)

## Citation

BibTeX will be added once a preprint is available.
