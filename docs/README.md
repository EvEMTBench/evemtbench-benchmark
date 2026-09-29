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
| [protocol.md](protocol.md) | the `held_out` protocol: episode-grouped train/val/test + the cross-distribution `benchmark` family |
| [taxonomy.md](taxonomy.md) | the task taxonomy (function × observability × label kind) |
| [task_suite_roadmap.md](task_suite_roadmap.md) | the 24-task suite by difficulty tier + open challenges |
| [task_cards/](task_cards/) | one card per task family (labels, vocab, caveats) |
| [metrics.md](metrics.md) | frozen metrics, `mean/std/ci95` aggregation, undefined ≠ 0 |
| [grids.md](grids.md) | the four reference grids (MV/HV/EHV) |
| [datasheet.md](datasheet.md) | dataset provenance, families, auxiliary artifacts |
| [determinism.md](determinism.md) | what "reproducible" means here; seed policy |
| [versioning.md](versioning.md) | benchmark/splits/contract semver; what a major bump retires |

## I want to contribute or extend

- [../CONTRIBUTING.md](../CONTRIBUTING.md) — contributor workflow (branches, CI gates, Definition
  of Done) + the add-a-task walkthrough; the development repo additionally carries an internal
  onboarding guide.
- [extending.md](extending.md) — machinery: add a grid, a baseline, a protocol.
- [DATA_LAYOUT.md](DATA_LAYOUT.md) — the on-disk contract: families × grids × artifacts,
  the auxiliary overlay (topology graphs + per-episode settings), join keys.
- [design/](design/) — designed-but-not-built slots (relay selectivity, noise robustness,
  cross-topology/GNN) = the paper's Open Challenges.
- [baselines.md](baselines.md) — the 7 reference baselines and their training regime.

## Internal (operations, agent memory, history)

The development repository additionally keeps `docs/internal/` — the operating playbook, lessons
log, agent session memory and archived result sets. It is needed to *run* the benchmark's compute,
not to *use* the benchmark, and is not part of the public release.
