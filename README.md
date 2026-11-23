# EvEMTBench Benchmark

[![ci](https://github.com/EvEMTBench/evemtbench-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/EvEMTBench/evemtbench-benchmark/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2609.28149-b31b1b)](https://arxiv.org/abs/2609.28149)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.24463496-blue)](https://doi.org/10.5281/zenodo.24463496)
![benchmark v1.3.0](https://img.shields.io/badge/benchmark-v1.3.0-blue)
![python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
[![code: MIT](https://img.shields.io/badge/code-MIT-green)](LICENSE)
[![data: CC-BY-4.0](https://img.shields.io/badge/data-CC--BY--4.0-green)](LICENSES/DATA-LICENSE.md)

A reproducible machine-learning benchmark for **power-system protection** on EMT waveforms. The
dataset provides the data; this repository defines how it is used: tasks, protocols, splits,
leakage checks, metrics and baselines.

**Paper:** J. Oelhaf, G. Kordowich, C. Bergler, A. Maier, J. Jäger, S. Bayer, *EvEMTBench: An Open
Benchmark for Machine Learning in Power System Protection*, arXiv:2609.28149 (2026).

**Results (v1.3.0):** [RESULTS.md](RESULTS.md) ·
[leaderboard](https://evemtbench.github.io/evemtbench-benchmark/leaderboard.html) ·
[provenance dashboard](https://evemtbench.github.io/evemtbench-benchmark/audit/index.html).
The recorded run state is in `coverage_snapshot.json`.

## Quickstart

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
export EVEMTBENCH_DATA_ROOT=/path/to/EvEMTBench
```
```python
from evemtbench import load_task, evaluate
task = load_task("fault_detection_local", "held_out", grid="double_line")
result = evaluate(task, my_model.predict(task.test))
```

More in [docs/quickstart.md](docs/quickstart.md). Your own model needs only `data`, `tasks`,
`splits` and `evaluation`; it never has to import `baselines` (CI enforces this).

Command-line tools: `evemtbench-evaluate`, `evemtbench-run`, `evemtbench-splits`,
`evemtbench-leaderboard`, `evemtbench-significance`, `evemtbench-validate-submission`
([submitting](docs/submission.md)).

## Dataset

The published results use **EvEMTBench dataset v0.1** (CC-BY-4.0),
<https://doi.org/10.48742/fau.abpd-r521>, on FAUDataCloud.
The newer dataset **V1**
(`10.48742/fau.3383-vd93`) is not used by benchmark 1.x. The data-generation code is archived at
<https://doi.org/10.5281/zenodo.21624649>.

The dataset holds the raw simulation exports. The benchmark reads windowed partitions; the tool that
builds them is not public yet. Until it is, build windows to the contract in
`src/evemtbench/contract/`: the loader rejects partitions that do not match it, and
`python -m evemtbench.data.input_digests --verify` checks them against the reference digests
(manifest, label and channel-descriptor files byte for byte; `--deep` also hashes the waveform
arrays where a pin is committed).
Layout: [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md); provenance: [docs/datasheet.md](docs/datasheet.md).

## Grids and protocols

Four reference grids: CigreMV (20 kV), DoubleLine (110 kV), TestGrid110kV (110 kV) and IEEE-39
(345 kV, 60 Hz). A `multi_grid` template family is used only for pre-training. Splits are by
simulation episode. Details: [docs/grids.md](docs/grids.md), [docs/protocol.md](docs/protocol.md).

| Protocol | Train on | Score on |
|---|---|---|
| `held_out` | one grid's `adapt_grid` family (70/15/15) | its test split and the unseen `benchmark` family |
| `multi_grid_pretrain` | the `multi_grid` family (topology-grouped split) | its own test split |
| `transfer_zeroshot` | nothing on the target (pre-trained checkpoint) | the target grid's test sets |
| `transfer_finetune` | pre-trained checkpoint, fine-tuned on the target grid | the target grid's test sets |

Pre-training and transfer use the fixed-width `local` and `line` views only.

### Names in the paper and in this repository

| Paper | Repository and result files | Dataset archive |
|---|---|---|
| `eval_grid` family | `benchmark` (test set `benchmark`) | `benchmark-*.tar.gz` |
| `adapt_grid` family | `adapt_grid` (test set `test`) | `adapt_grid-*.tar.gz` |
| `multi_grid` family | `multi_grid` | `multigrid-*.tar.gz` |
| CIGRÉ MV grid (20 kV) | `cigre_mv` | `CigreMVGrid` |
| Double Line grid (110 kV) | `double_line` | `DoubleLine` |
| Protection Reference grid (110 kV) | `testgrid_110kv` | `TestGrid110kV` |
| IEEE-39 (345 kV) | `ieee39` | `IEEE39BusSystem` |

## Tasks

24 task ids over 12 functions, in three difficulty tiers. Each is scored at up to three
**observability** views: `local` (one line end, 6 channels), `line` (both ends, 12 channels),
`global` (all measurements, grid-dependent width).

| Group | Function | Views | Tier · type |
|---|---|---|---|
| Fault (fault-active windows) | fault detection (FD) | local/line/global | 1 · binary |
| | fault classification (FC) | local/line/global | 2 · 9 classes |
| | fault location (FL) | local/line/global | 3 · regression (% of line) |
| | grounded | local/line/global | 1 · binary |
| | phase | local/line/global | 2 · 7 classes |
| | category | local/line/global | 2 · 3 classes |
| Event (all windows) | event detection (ED) | global | 1 · binary |
| | switch detection | global | 1 · binary |
| | event state | global | 2 · 3 classes |
| | switch type | global | 2 · 8 classes |
| | fault origin | global | 2 · binary |
| | event classification (EC) | global | 3 · 23 classes |

Compare results only within the same view, grid and protocol.

**Baselines:** `majority`, `threshold`, `impedance`, `random_forest`, `mlp`, `gru`, `cnn`,
`resnet`, `tst` ([docs/baselines.md](docs/baselines.md)). Deep baselines need a GPU.

## Documentation

[docs/README.md](docs/README.md) is the index: protocol, grids, tasks, metrics, baselines,
submission, versioning, known issues.

## Reproducibility

- [reproduce/](reproduce/) has one guide per group of paper results: where the numbers are,
  how to retrain them, which script rebuilds each table and figure.
- Splits are committed (`src/evemtbench/splits/`) and regenerate deterministically.
- Every run writes `run_record.json` (seeds, git commit, environment hashes).
- `env/results-v<version>.lock` pins the environment the published results ran in;
  `requirements-ci-py3.{10,12}.lock` pin the torch-free CI.
- `RESULTS.md`, the leaderboard and the dashboard are rebuilt from the run outputs, and a
  pre-commit check keeps them in sync. The reference runs used SLURM on NHR@FAU Alex (`hpc/`).

## License and citation

Code: MIT ([LICENSE](LICENSE)). Dataset: CC-BY-4.0 ([LICENSES/DATA-LICENSE.md](LICENSES/DATA-LICENSE.md)).
Release v1.3.0 is archived on Zenodo: <https://doi.org/10.5281/zenodo.24463496> (version DOI; all versions:
<https://doi.org/10.5281/zenodo.24463495>). Please cite the paper and the dataset, and the software archive when you
use the code; identifiers are in [CITATION.cff](CITATION.cff).

```bibtex
@misc{oelhaf_evemtbench_2026,
  title         = {{EvEMTBench}: An Open Benchmark for Machine Learning in Power System Protection},
  author        = {Oelhaf, Julian and Kordowich, Georg and Bergler, Christian and Maier, Andreas
                   and J{\"a}ger, Johann and Bayer, Siming},
  year          = {2026},
  eprint        = {2609.28149},
  archivePrefix = {arXiv},
  primaryClass  = {eess.SY},
  doi           = {10.48550/arXiv.2609.28149},
  url           = {https://arxiv.org/abs/2609.28149}
}

@software{oelhaf_evemtbench_benchmark_v130_2026,
  title     = {{EvEMTBench Benchmark}: evaluation software for machine learning in power system protection},
  author    = {Oelhaf, Julian and Kordowich, Georg and Bergler, Christian and Maier, Andreas
               and J{\"a}ger, Johann and Bayer, Siming},
  year      = {2026},
  version   = {v1.3.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24463496},
  url       = {https://doi.org/10.5281/zenodo.24463496}
}
```

## Development

Development happens in a private university GitLab. This GitHub repository is an export of tagged
releases. Issues and pull requests are welcome ([CONTRIBUTING.md](CONTRIBUTING.md)).
