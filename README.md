# qrc-thresher

[![Phase 1](https://img.shields.io/badge/Phase-1%20In%20Progress-yellow)](docs/BUILD_SPEC.md)

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

### Quick start

```bash
python -m qrc_thresher.cli health
```

Run a benchmark:

```bash
python -m qrc_thresher.cli run stm --config configs/alpha_lite.yaml
```

## Phase 1 Status

- [x] Package scaffold
- [x] STM and temporal parity task generators (deterministic, seeded)
- [x] PennyLane QRC (angle encoding, ring entanglement, Pauli-Z readout)
- [x] ESN and random kitchen sinks baselines
- [x] Ablations: phase-randomized, entanglement-suppressed, Haar-random
- [x] Metrics: MC, NRMSE, accuracy, bootstrap CIs, paired t-test
- [x] Proof layer: run manifests (schema v1.1), health checks
- [x] CLI: health, run, ablation, gate, plot, summary
- [ ] G1: STM gate evaluation (requires benchmark runs)
- [ ] G2: Parity gate evaluation
- [ ] G3: ESN comparison gate

## License

Triple-license:
- **Apache-2.0** — all code in `src/` and `tests/` (see [LICENSE](LICENSE))
- **CC BY 4.0** — documentation in `docs/` (see [docs/LICENSE-DOCS](docs/LICENSE-DOCS))
- **CC0 1.0** — synthetic data outputs in `data/` (see [data/LICENSE-DATA](data/LICENSE-DATA))

## Documentation

- [BUILD_SPEC.md](docs/BUILD_SPEC.md) — full build specification
- [METHODOLOGY.md](docs/METHODOLOGY.md) — task definitions, circuit spec, statistical methodology
- [REFERENCES.md](docs/REFERENCES.md) — reference literature
- [DECISIONS.md](docs/DECISIONS.md) — decision log

## Citation

BibTeX will be added once a preprint is available.
