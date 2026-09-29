# EvEMTBench Benchmark

[![ci](https://github.com/EvEMTBench/evemtbench-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/EvEMTBench/evemtbench-benchmark/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2609.28149-b31b1b)](https://arxiv.org/abs/2609.28149)
![benchmark v1.1.0](https://img.shields.io/badge/benchmark-v1.1.0-blue)
![python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
[![code: MIT](https://img.shields.io/badge/code-MIT-green)](LICENSE)
[![data: CC-BY-4.0](https://img.shields.io/badge/data-CC--BY--4.0-green)](LICENSES/DATA-LICENSE.md)

A protocol-driven, reproducible ML benchmark for **power-system protection** on EMT waveform
data. The benchmark layer turns the EvEMTBench dataset into a fair, comparable evaluation artifact
— it defines the *tasks, protocols, splits, leakage controls, metrics, and baselines*.

> The dataset provides the data; the benchmark protocol defines how the data may be used for
> fair and reproducible evaluation.

**Paper:** J. Oelhaf, G. Kordowich, C. Bergler, A. Maier, J. Jäger, S. Bayer, *EvEMTBench: An Open
Benchmark for Machine Learning in Power System Protection*, arXiv:2609.28149 (2026) —
<https://arxiv.org/abs/2609.28149>

**Status (benchmark contract v1.1.0).** The reference evaluation is complete across the full
expected matrix: every task on every reference grid under `held_out`, and the portable `local` /
`line` fault tasks under pre-training and both transfer protocols (deep baselines only). The
committed `coverage_snapshot.json` is the recorded state. Browse the outcomes in
[RESULTS.md](RESULTS.md), the [live leaderboard](https://evemtbench.github.io/evemtbench-benchmark/leaderboard.html) and the
[provenance dashboard](https://evemtbench.github.io/evemtbench-benchmark/audit/index.html) (see **Results, leaderboard & audit** below).

## Design: four layers, one rule

```
src/evemtbench/
├── data/         # Layer 1 — windowed-data loading (roots, windows memmap, schema)
├── tasks/        # Layer 2 — meaning (task defs, label derivation, line-view, valid samples)
├── splits/       # Layer 3 — committed split FILES + generator + leakage checks
├── evaluation/   # THE PRODUCT — frozen, versioned metrics + evemtbench-evaluate CLI
└── baselines/    # demos only — import-isolated
```

**The rule:** you can evaluate your own model using only `data` / `tasks` / `splits` /
`evaluation` — never importing `baselines`. CI enforces it. The evaluation layer is the product;
baselines are demos.

## Dataset

The benchmark evaluates on the **EvEMTBench dataset** (point-on-wave EMT waveform windows,
window-level labels, per-episode settings and topology graphs; CC-BY-4.0, separate from this
code). Download it from FAUDataCloud:
<https://data.fau.de/share/0e8d60feb7e65616c60aab78b93db77053275da53fd894bf5b75fc5e9ee7dfbf/>
(a dataset DOI, `10.48742/fau.1fkx-ef11`, is reserved but not yet active — cite the share link
and the dataset paper until it resolves). Point `EVEMTBENCH_DATA_ROOT` (or `EVEMT_ROOT`) at the
unpacked root — the expected layout is documented in [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md)
and provenance in [docs/datasheet.md](docs/datasheet.md).

> **Windowed data is not downloadable yet.** The share holds the **raw simulation exports**
> (one tarball per family × grid, with per-simulation `data/result*.csv`). The benchmark reads
> **windowed** partitions (`<family>/windows/<Grid>/X_*.raw` + manifest, meta and label files —
> [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md)). The producer pipeline that turns the raw exports
> into windows is not public yet (its release is planned); until then, windows have to be built
> to the pinned **window contract** in `src/evemtbench/contract/` (`contract.json`, the
> `manifest`/`meta` JSON schemas, version in `CONTRACT_VERSION`). The loader checks every
> partition's manifest against that contract and fails loudly on a mismatch, and
> `python -m evemtbench.data.input_digests --verify` re-hashes the partitions on disk against the
> committed digests of the reference windows (`src/evemtbench/data/input_digests.json`).

## Quickstart

See [docs/quickstart.md](docs/quickstart.md). In short:

```bash
uv venv && uv pip install -e ".[dev]"
export EVEMTBENCH_DATA_ROOT=/path/to/EvEMTBench
```
```python
from evemtbench import load_task, evaluate
task = load_task("fault_detection_local", "held_out", grid="double_line")
result = evaluate(task, my_model.predict(task.test))
```

CLIs: `evemtbench-evaluate` (score predictions), `evemtbench-run` (run a baseline),
`evemtbench-splits` (regenerate splits), `evemtbench-leaderboard`, `evemtbench-significance`.

## Grids & protocols

A **multi-grid** benchmark over **four reference grids** spanning voltage levels — CigreMV
(20 kV, MV), DoubleLine (110 kV, HV), TestGrid110kV (110 kV, HV) and IEEE-39 (345 kV, EHV; the
only 60 Hz grid) — plus a `multi_grid` **template** family used only for pre-training, spanning
**20 kV, 110 kV and 345 kV (EHV)** template grids. Splits are at the **simulation-episode level**
(`sample_id`). Full per-grid topology,
channel layout and the observability oracle are in [docs/grids.md](docs/grids.md); the canonical
contract is [docs/protocol.md](docs/protocol.md).

Four protocols make up the evaluation matrix:

| Protocol | Fit on | Reported on |
|---|---|---|
| **`held_out`** | a reference grid's `adapt_grid` family (70/15/15, in-distribution) | that grid's held-out test **and** the whole never-trained `benchmark` family |
| **`multi_grid_pretrain`** | a topology-grouped split of the `multi_grid` template family (v1.1) | its own held-out test (a portable pre-trained model) |
| **`transfer_zeroshot`** | *nothing on the target* — reuses a `multi_grid_pretrain` checkpoint | a reference grid's held-out sets, no target-grid training |
| **`transfer_finetune`** | a `multi_grid_pretrain` checkpoint, then fine-tuned on the reference grid's train/val | that grid's held-out sets |

Pre-training and transfer apply to the **portable** fixed-width `local` / `line` views (6 / 12 ch)
only; `global` is grid-specific width and stays `held_out`-only. A cross-topology protocol (train on
diverse topologies → test on unseen grids, with a topology-aware GNN) is planned.

## Tasks

**24 task ids across 12 functions on a 3-tier difficulty ladder** (1 = easy anchor … 3 = hard
frontier). Each function is scored at one or more of three **observability** settings — a nested
oracle centred on the faulted line that maps to conventional protection: `local` = single-end
(overcurrent/distance, 6 ch), `line` = two-ended **differential** (12 ch), `global` =
centralized/wide-area (grid-dependent width).

| Group (windows) | Function | Views | Tier · Type |
|---|---|---|---|
| **Fault triad** (fault-active) | FD fault detection | local/line/global | 1 · binary |
| | FC fault classification | local/line/global | 2 · 9-class `flt_*` |
| | FL fault location | local/line/global | 3 · regression (% along line) |
| **Fault attributes** (fault-active) | grounded | local/line/global | 1 · binary |
| | phase | local/line/global | 2 · 7-class (A/B/C combo) |
| | category | local/line/global | 2 · 3-class (shc/incipient/hif) |
| **Event scope** (all windows) | event detection (ED) | global | 1 · binary |
| | switch detection | global | 1 · binary |
| | event state | global | 2 · 3-class (normal/switching/fault) |
| | switch type | global | 2 · 8-class |
| | fault origin | global | 2 · binary (line vs bus) |
| | event classification (EC) | global | 3 · 23-class |

Event-scope tasks are `global`-only: switching/normal windows have no faulted line for the
local/line oracle. **Fairness rule:** never compare across observability settings, grids, or
protocols without stating which. Vocabularies: [docs/taxonomy.md](docs/taxonomy.md); per-task
cards: [docs/task_cards/](docs/task_cards/); metrics: [docs/metrics.md](docs/metrics.md); the tier
ladder and roadmap: [docs/task_suite_roadmap.md](docs/task_suite_roadmap.md).

Baselines: `majority`, `threshold` (overcurrent; FD binary + `local` only), `random_forest`, `mlp`,
`gru`, `cnn`, `resnet` (1-D waveform); a topology-aware `gnn` is planned. See
[docs/baselines.md](docs/baselines.md).

## Documentation

[protocol](docs/protocol.md) · [grids](docs/grids.md) · [taxonomy](docs/taxonomy.md) ·
[metrics](docs/metrics.md) · [task cards](docs/task_cards/) · [task-suite roadmap](docs/task_suite_roadmap.md)
· [submission](docs/submission.md) · [extending](docs/extending.md) · [datasheet](docs/datasheet.md)
· [versioning](docs/versioning.md) · [determinism](docs/determinism.md)

## Reproducibility

- Two environments, two pins: `env/results-v<version>.lock` records the conda prefix the
  **published results** ran in (torch + CUDA included; one lock per results version);
  `requirements-ci-py3.{10,12}.lock` hash-pin the **CI test gate** (torch-free). HPC layers the
  package onto the `py_dl` conda base.
- `hpc/verify_results_env.py` checks every run record's independently captured package versions
  against that lock — two independent captures agreeing is the actual guarantee. It needs the
  `outputs/` tree, which is not part of this repository.
- Committed splits are the source of truth and regenerate deterministically: `splits/v1.0/` for
  `held_out` (reused by the transfer protocols on the target grid) and the topology-grouped
  `splits/v1.1/multi_grid_pretrain/` for pre-training.
- Every run emits `run_record.json` (seeds, git sha, lockfile + results-env hashes, env). See
  [docs/versioning.md](docs/versioning.md) and [docs/determinism.md](docs/determinism.md).

## HPC

The reference sweep runs on an HPC cluster (SLURM, GPU; NHR@FAU **Alex**). All scripts drive the
same `evemtbench-run` entrypoint.

| Script | Use for |
|---|---|
| `hpc/sweep.sh` | grid-parametric worker — runs the manifest-`pending` cells for one (grid, baseline-group), staging to `/scratch` |
| `hpc/submit.sh` | submit driver — fans out per-grid CPU + GPU jobs (dry-run by default; `--submit` to queue) |

(The development repository additionally runs a hands-off submission loop and an auto-commit
publisher around these entrypoints; machine-local defaults live in the untracked
`hpc/clusters/local.env`.)

`evemtbench.coverage` is the manifest (expected matrix vs. `result.json` → done/pending/failed) that
drives the skip-existing logic and progress tracking — it is the single source of truth for what is
done (`python -m evemtbench.coverage --summary`; persisted as the committed `coverage_snapshot.json`).
`--summary` defaults to `held_out` (pass `--protocols` to widen) and reads a local `outputs/` tree,
which is not part of the repository — in a fresh clone every cell reads as pending, so use the
committed snapshot for the recorded state.
Deep baselines (`mlp`, `gru`, `cnn`, `resnet`) need a CUDA-capable GPU. `hpc/make_results.py`
regenerates `RESULTS.md` from `outputs/`; see [docs/extending.md](docs/extending.md) for the full
run/scratch/memory workflow.

## Results, leaderboard & audit

- **[RESULTS.md](RESULTS.md)** — tier-grouped headline results, deterministically rebuilt from
  `outputs/`.
- **`leaderboard.html`** — [browse online](https://evemtbench.github.io/evemtbench-benchmark/leaderboard.html) (GitHub Pages) or open it
  locally from a clone; it ships in the repo as static HTML. Regenerated by `evemtbench-leaderboard`.
- **`audit/index.html`** — [browse online](https://evemtbench.github.io/evemtbench-benchmark/audit/index.html) — the reviewer-facing
  provenance dashboard: every claim resolves to its
  script, verbatim command, artifact, runtime, hardware and verifying gate; the full coverage matrix
  with per-cell reproduce commands; a code tour of the verification gates; and per-task/grid/baseline
  drill-downs. Every number is read from committed artifacts at build time, and the pre-commit
  freshness gate keeps it byte-fresh (`hpc/make_audit.py`; concept docs in
  [docs/results_audit/](docs/results_audit/)).

## Licensing & citation

Code: MIT ([LICENSE](LICENSE)). Dataset: separate, CC-BY-4.0 — see
[LICENSES/DATA-LICENSE.md](LICENSES/DATA-LICENSE.md).

If you use the benchmark, please cite the benchmark paper and the dataset (separate DOIs) — see
[CITATION.cff](CITATION.cff) for the current identifiers. The benchmark paper:

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
```

## Development

Development happens on a private university GitLab; the public GitHub repository is a curated,
leak-scanned export of tagged states (built by `hpc/make_public_release.sh`). Issues and pull
requests on GitHub are welcome and are applied upstream — see
[CONTRIBUTING.md](CONTRIBUTING.md).
