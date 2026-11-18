# EvEMTBench documentation — where to look

The benchmark in one line: **24 protection tasks in 3 difficulty tiers × 4 reference grids ×
3 observability settings, scored on two held-out sets under a frozen, versioned contract.**

## I want to evaluate my own model (BYO)

1. [quickstart.md](quickstart.md) — environment, data root, first run.
2. [`examples/byo_model.py`](../examples/byo_model.py) — the two supported BYO paths
   (plug into the runner via `register_baseline`, or `load_task → predict → evaluate`).
3. [submission.md](submission.md) — what a valid, reproducible submission contains
   (artifact schema, seeds, the no-tuning attestation).

## I want to understand the benchmark's science

| doc | contents |
|---|---|
| [protocol.md](protocol.md) | the protocols: episode-grouped train/val/test, the `benchmark` family, pre-training and transfer |
| [taxonomy.md](taxonomy.md) | the event taxonomy and which grid has which event |
| [task_suite_roadmap.md](task_suite_roadmap.md) | the 24-task suite by difficulty tier + open challenges |
| [task_cards/](task_cards/) | one card per task family (labels, vocab, caveats) |
| [metrics.md](metrics.md) | frozen metrics, `mean/std/ci95` aggregation, undefined ≠ 0 |
| [grids.md](grids.md) | the four reference grids (MV/HV/EHV) and the observability views |
| [datasheet.md](datasheet.md) | dataset identity, provenance, families, limitations |
| [determinism.md](determinism.md) | what "reproducible" means here; seed policy |
| [known_issues.md](known_issues.md) | known dataset and scoring issues that affect published numbers, with their status |
| [versioning.md](versioning.md) | benchmark/splits/contract semver; what a major bump retires |

## I want to contribute or extend

- [../CONTRIBUTING.md](../CONTRIBUTING.md) — contributor workflow (branches, CI gates, Definition
  of Done).
- [extending.md](extending.md) — add a grid, a task, a baseline, a protocol.
- [DATA_LAYOUT.md](DATA_LAYOUT.md) — the on-disk contract: families × grids × artifacts,
  the auxiliary overlay (topology graphs + per-episode settings), join keys.
- [baselines.md](baselines.md) — the reference baselines (majority, threshold, impedance,
  random_forest, mlp, gru, cnn, resnet, tst) and their training regime.
- [design/](design/) — design notes: pre-training and transfer (built); relay selectivity, noise
  robustness and cross-topology/GNN (designed, not built).
