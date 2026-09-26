# Evidence exported at commit 8ed2df4

Byte-exact copies of gate records from `results/`, exported by `qrc-thresher evidence` (docs/DECISIONS.md D019, ruling P4). Nothing here is re-serialised; every recorded path inside the files is as recorded.

- Family: `COMPARATIVE.v1.20260924T224106784228Z.json`; protocol COMPARATIVE v1; config_hash `509b9d0be086f1a8d8ab8b7af4965c28dc79a1db686b4fdf1a940d5b821666cf`; sweep_id 20260924T214530147999Z; commit 8ed2df4d082ee6613e9a9480d5f5f41b99b23f4e.
- Rows: 336 rows of sweep(s) 20260924T214530147999Z.

## Files

| file | sha256 | recorded as |
|---|---|---|
| `COMPARATIVE.v1.20260924T224106784228Z.json` | `3da53714d0728d7d3d4276f0c1084c59fb1a3e3daf05eeb81539943c5ef85826` | results/gates/COMPARATIVE.v1.20260924T224106784228Z.json |
| `configs/comparative.yaml` | `ad064d75ee5f6b18f9e12ff6039897dcab46334144bc8c26de6a263280541f50` | configs/comparative.yaml |
| `G0.7.tuned_qrc.20260924T224100291861Z.forgetting_curve.png` | `b71181a110f132c3b6b28708cc6a320cfa80f2f0fc9e9537998ebca2816341e9` | results/gates/G0.7.tuned_qrc.20260924T224100291861Z.forgetting_curve.png |
| `G0.7.tuned_qrc.20260924T224100291861Z.json` | `5164fcd97bd63e36e8135b91fc9d802223310ba423fc77187f5e07fac72f0a8a` | results/gates/G0.7.tuned_qrc.20260924T224100291861Z.json |
| `G1.20260924T224106784228Z.json` | `903ea80c44a8d4e323ed440de9d4b2f8abea68dacae0ff07acd0ba4eecf8a4ee` | results/gates/G1.20260924T224106784228Z.json |
| `G2.20260924T224106784228Z.json` | `b035a9a176f8f570ef7659030a2e68c700b833e00786b93155e2fbbdf6a71771` | results/gates/G2.20260924T224106784228Z.json |
| `G2.5.20260924T224106784228Z.json` | `1203a85c37391ff2c5889a1e5f13dcecd3eebf03c55fe13e5b9502e5e50f47b5` | results/gates/G2.5.20260924T224106784228Z.json |
| `G3.20260924T224106784228Z.json` | `766f5150d4506cb8bd93eb8e4f0cfec8173d93335bd24e515b1a453626a64777` | results/gates/G3.20260924T224106784228Z.json |
| `G4.20260924T224106784228Z.json` | `8145d873f609813185425d999448f43729ad4ab8531bde18864747cc18b34857` | results/gates/G4.20260924T224106784228Z.json |
| `runs.sanitised.csv` | `687d7a509ed199d7a088b402da38ade9cc71ecadd1ec941fc6a9cbc0af5d4b10` | the rows of the family's config and sweep in results/runs.csv, allowlisted columns only |
| `tuning/narma.json` | `1533adce9a184def27c4d038db9442e800d6b2411aaa7f5a0a1134f2e8806040` | results/tuning/509b9d0be086f1a8d8ab8b7af4965c28dc79a1db686b4fdf1a940d5b821666cf/narma.json |
| `tuning/parity.json` | `98646f95e68fdd498e94f8b4e35054d8aa8bf39ded14d3416817abb0b05b7e39` | results/tuning/509b9d0be086f1a8d8ab8b7af4965c28dc79a1db686b4fdf1a940d5b821666cf/parity.json |
| `tuning/stm.json` | `ce0abb9400d00796db9d21d0977137d5d29e1c9efd636d5455eaf166f7547787` | results/tuning/509b9d0be086f1a8d8ab8b7af4965c28dc79a1db686b4fdf1a940d5b821666cf/stm.json |

MANIFEST.sha256 lists every file in this folder except itself, in sha256sum format; this README is listed there and does not hash the manifest.

## Path mappings

| recorded path | in this folder |
|---|---|
| `results/gates/<X>` | `<X>` |
| `results/tuning/509b9d0be086f1a8d8ab8b7af4965c28dc79a1db686b4fdf1a940d5b821666cf/<task>.json` | `tuning/<task>.json` |
| results/tuning/509b9d0be086f1a8d8ab8b7af4965c28dc79a1db686b4fdf1a940d5b821666cf/stm.json | tuning/stm.json |
| results/tuning/509b9d0be086f1a8d8ab8b7af4965c28dc79a1db686b4fdf1a940d5b821666cf/parity.json | tuning/parity.json |
| results/tuning/509b9d0be086f1a8d8ab8b7af4965c28dc79a1db686b4fdf1a940d5b821666cf/narma.json | tuning/narma.json |
| configs/comparative.yaml | configs/comparative.yaml |
| results/gates/COMPARATIVE.v1.20260924T224106784228Z.json | COMPARATIVE.v1.20260924T224106784228Z.json |
| results/gates/G0.7.tuned_qrc.20260924T224100291861Z.forgetting_curve.png | G0.7.tuned_qrc.20260924T224100291861Z.forgetting_curve.png |
| results/gates/G0.7.tuned_qrc.20260924T224100291861Z.json | G0.7.tuned_qrc.20260924T224100291861Z.json |
| results/gates/G1.20260924T224106784228Z.json | G1.20260924T224106784228Z.json |
| results/gates/G2.20260924T224106784228Z.json | G2.20260924T224106784228Z.json |
| results/gates/G2.5.20260924T224106784228Z.json | G2.5.20260924T224106784228Z.json |
| results/gates/G3.20260924T224106784228Z.json | G3.20260924T224106784228Z.json |
| results/gates/G4.20260924T224106784228Z.json | G4.20260924T224106784228Z.json |

## Dropped row columns

runs.sanitised.csv keeps only the allowlisted columns (`run_id`, `success`, `git_commit_hash`, `config_hash`, `sweep_id`, `tuning_record_sha`, `circuit_hash`, `task_seed`, `reservoir_seed`, `task_name`, `design`, `primary_metric_name`, `primary_metric_value`, `measurement_model`, `n_configs`, `n_validation_evals`, `secondary_metrics`, `python_version`, `backend_device`, `device`, `precision`, `entanglement_metric`, `runtime_per_stage_seconds`). These columns of results/runs.csv were dropped because they can carry a local path, a machine identity or a timestamp: `cli_command`, `git_branch`, `platform`, `artifact_paths`, `package_versions`, `config_path`, `timestamp_utc`, `failure_reason`. Every kept cell is the identical text of its source cell (metric strings are never passed through a float).

## The CP4c command sequence

The runbook first archived results/runs.csv, results/experiments.db and results/tuning/<config_hash>/, then ran, in this order:

```
qrc-thresher tune --config configs/comparative.yaml
qrc-thresher run stm --config configs/comparative.yaml
qrc-thresher run parity --config configs/comparative.yaml
qrc-thresher run narma --config configs/comparative.yaml
qrc-thresher run parity --config configs/comparative.yaml --design-task stm
qrc-thresher run stm --config configs/comparative.yaml --design default
qrc-thresher run parity --config configs/comparative.yaml --design default
qrc-thresher run narma --config configs/comparative.yaml --design default
qrc-thresher ablation no_entangle stm --config configs/comparative.yaml
qrc-thresher ablation no_entangle parity --config configs/comparative.yaml --design-task stm
qrc-thresher ablation no_entangle stm --config configs/comparative.yaml --design default
qrc-thresher ablation no_entangle parity --config configs/comparative.yaml --design default
qrc-thresher ablation haar stm --config configs/comparative.yaml
qrc-thresher ablation haar stm --config configs/comparative.yaml --design default
qrc-thresher baseline stm --config configs/comparative.yaml
qrc-thresher baseline stm --config configs/comparative.yaml --design default
qrc-thresher baseline parity --config configs/comparative.yaml
qrc-thresher baseline parity --config configs/comparative.yaml --design default
qrc-thresher baseline narma --config configs/comparative.yaml
qrc-thresher baseline narma --config configs/comparative.yaml --design default
qrc-thresher gate G0.7 --config configs/alpha_lite.yaml --model tuned_qrc --tuning-config configs/comparative.yaml
qrc-thresher gate family --config configs/comparative.yaml
qrc-thresher summary --phase cp4c
```
