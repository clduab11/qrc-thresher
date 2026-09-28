# qrc-thresher

[![Phase 1](https://img.shields.io/badge/Phase-1%20In%20Progress-yellow)](docs/BUILD_SPEC.md)
[![Augmentation 2026](https://img.shields.io/badge/Augmentation-2026-blue)](docs/MASTER_AUGMENTATION_PLAN_2026.md)

A quantum reservoir computing (QRC) workbench with pre-registered gates as its quality bar (D002):
reservoirs that remember (windowed input, D010), fair classical baselines built in and tuned under
the same budget (D009, D011), matched ablations (D010), paired statistics over a registered family
(D013, D014), and diagnostics that aim to show what the quantum parts contribute (so far the
forgetting curve). The gates are not the product; they make the results trustworthy, positive or
negative. The question they test:

> Do small simulated quantum reservoirs generate temporal features that are useful, in a measurable
> way, beyond what tuned classical baselines and ablation controls can produce?

Negative results are reported with the same prominence as positive ones. Every QRC number so far is
computed from exact expectation values and labelled "exact (oracle upper bound)" (D004): it bounds
from above what a device could measure, and finite-shot measurement is still to come. G0.7, the
memory sanity gate, tests each model on every seed pair: its summed STM memory over delays k ≥ 1
against a permutation null, and its window-2 parity against a shuffled-label null (D005, D008). It
draws the forgetting curve (per-delay r²_k against the null quantile, `*.forgetting_curve.png`)
beside the JSON verdict.

## Installation

### WSL2 / Ubuntu 22.04+ / macOS / Linux

```bash
git clone https://github.com/cld-maindev/qrc-thresher.git
cd qrc-thresher
pip install -e .
```

The project is also reproducibly installable via [uv](https://github.com/astral-sh/uv):

```bash
uv sync --locked
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
qrc-thresher ablation no_entangle stm --config configs/alpha_lite.yaml
qrc-thresher tune --config configs/comparative.yaml           # tuning records for the family
qrc-thresher baseline stm --config configs/comparative.yaml   # ESN and RKS rows
qrc-thresher gate G0.7 --config configs/alpha_lite.yaml       # memory sanity gate, protocol v1
qrc-thresher gate G0.7 --config configs/alpha_lite.yaml --model tuned_qrc --tuning-config configs/comparative.yaml
qrc-thresher gate family --config configs/comparative.yaml    # G1, G2, G2.5, G3, G4 as one unit
qrc-thresher evidence results/gates/COMPARATIVE.v1.<stamp>.json --config configs/comparative.yaml --runs-csv results/runs.csv
qrc-thresher scorecard   # every docs/evidence/*/ folder -> docs/scorecard.md
qrc-thresher plugins                    # list discovered plugins
qrc-thresher perf --iterations 5        # micro-benchmarks
qrc-thresher noise-sweep                # Aer noise-model sweep scaffold
qrc-thresher summary --phase phase1     # aggregate runs.csv into markdown (one row per config, sweep, deployment and task)
qrc-thresher plot RUN_ID                # a stub: writes no figures yet
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
- `qrc_thresher.reservoirs.smoothed_qrc` — classical smoothing of the QRC
  features over a window of past rows (`reservoirs.stateful_qrc` is a
  deprecated alias, D018). Design (b), a reservoir with carried quantum
  state, is not built (D003).

## Phase 1 Status

- [x] Package scaffold
- [x] STM and temporal parity task generators (deterministic, seeded)
- [x] Windowed PennyLane QRC (D010): qubit j re-uploads u_{t-(j mod w)} at every layer through Ry(α·u), fixed random Rz/Rx layers, ring CNOTs, Pauli-Z (optionally ZZ) readout; window w and scale α are tuned hyperparameters (D011)
- [x] `reservoirs/smoothed_qrc.py`: classical smoothing of the memoryless (w = 1) `pennylane_qrc` features: each row is blended 50/50 with the mean of itself and the `carry_depth` rows before it; reported as a variant and never as quantum memory (D018)
- [x] Classical baselines: a dense numpy ESN (D009) and windowed random kitchen sinks (bandwidth σ/√d, D011), both tuned under the QRC's budget; RKS is enabled in `configs/comparative.yaml` only
- [x] Ablations: phase-randomized, entanglement-suppressed, Haar-random
- [x] Metrics: MC, NRMSE, accuracy, bootstrap CIs, BCa CIs, paired tests,
      Holm–Bonferroni correction, power analysis
- [x] Proof layer: run manifests (schema 1.4), health checks, NARMA-10
      recurrence verification
- [x] CLI: health, run (every seed pair, `--workers`, `--design`, `--design-task`), ablation
      (`--design`, `--design-task`), baseline (`--design`), tune, gate (`--config`, `--model`,
      `--tuning-config`), evidence, scorecard, summary (`--config`), plugins, perf, noise-sweep;
      `plot` is a stub
- [x] Plugin SDK with entry-point discovery (tasks, reservoirs, baselines,
      gates, viz, noise models)
- [x] Parallel execution engine and SQLite result store
- [x] Pre-registered gate protocols in `configs/gates/` (G0.7.v1.yaml, COMPARATIVE.v1.yaml); the config `gates:` block was removed (D013)
- [x] Config overlay system (`load_config_with_overlays`)
- [x] Executable gate evaluators G0, G0.5, G0.7 (v1, `configs/gates/G0.7.v1.yaml`), the COMPARATIVE.v1 family (G1, G2, G2.5, G3, G4 as one unit), G5, G6, G7
- [x] CI matrix on Python 3.11 / 3.12 / 3.13 with `uv` lock-file installs;
      health and gate steps now block on failure
- [x] Visualisations: STM/MC, comparison, delay heatmap, runtime breakdown,
      gate decision tree, metric correlation matrix
- [x] Sphinx documentation scaffold (`docs/sphinx/`)
- [x] Run 4 (commit 8ed2df4): the COMPARATIVE.v1 family and G0.7 on the tuned design, exported
      under `docs/evidence/8ed2df4/`; its verdicts and numbers are rendered in
      [docs/scorecard.md](docs/scorecard.md). Every QRC number there is exact (oracle upper bound)
      and not a headline claim until the finite-shot v2 rerun (D004, D017, D019).
- [ ] Finite-shot measurement (D004; pending under a v2 protocol) and the G5–G7 checks

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
- [DECISIONS.md](docs/DECISIONS.md) — decision log (append-only)
- [scorecard.md](docs/scorecard.md) — gate verdicts rendered from `docs/evidence/`
- [AUDIT_2026-09.md](docs/AUDIT_2026-09.md) — the 2026-09 audit: defect register, decision map, reproduction
- `docs/sphinx/` — API reference scaffold (build with `sphinx-build`)

## Citation

BibTeX will be added once a preprint is available.
