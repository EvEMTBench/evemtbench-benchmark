# EvEMTBench — benchmark protocol (canonical contract)

The single source of truth for **what is scored and how**. Task cards and the quickstart link
here rather than restating it. Grids are documented in [grids.md](grids.md), events in
[taxonomy.md](taxonomy.md), metrics in [metrics.md](metrics.md), versioning policy in
[versioning.md](versioning.md).

## Task matrix

The registry (`src/evemtbench/tasks/registry.py`) holds **24 task ids across 12 functions**
(`<function>_<observability>`, e.g. `fault_detection_line`): the fault triad (FD, FC, FL) and the
three fault attributes (`fault_grounded`, `fault_phase`, `fault_category`) at each of
**{local, line, global}** (18 ids), plus six **`global`-only** event-scope tasks
(`event_detection_global`, `switch_detection_global`, `event_state_global`,
`switch_type_global`, `fault_origin_global`, `event_classification_global`). The full list with
tiers and label types is the task table in the [README](../README.md#tasks). Observability is
defined in [grids.md](grids.md). The table below covers the fault triad; the other functions are
specified in their [task cards](task_cards/).

| function | id stem | label kind | classes | valid samples (windows scored) |
|---|---|---|---|---|
| Fault **detection** (FD) | `fault_detection` | binary | 2 | all windows of fault episodes (`y_fault_line` defined): pre-fault negatives + fault-active positives |
| Fault **classification** (FC) | `fault_classification` | multiclass | **9** | fault-active windows whose fault type is in the 9-class vocab |
| Fault **location** (FL) | `fault_location` | regression | — | fault-active windows of **line** faults (`y_fault_line` = `MainLn*`) with a defined `y_fault_location`. Bus-target episodes are excluded ([known_issues.md](known_issues.md) §2) |

> **The fault tasks are ORACLE tasks — read the "valid samples" column as part of the task
> definition, not as a filtering detail.** Only fault episodes are scored, so FD's negatives are the
> *pre-onset windows of episodes that do fault*; and at `local`/`line` the input cubicles are chosen
> by the ground-truth faulted line (`y_fault_line`). FD is therefore **line-conditioned onset
> detection**, not grid-wide detection, and FL presupposes a correctly identified faulted line. This
> is intentional — it holds information constant so representations can be compared — but it
> means deployed performance is **overstated** by construction: a model that trips on any disturbance
> scores near-perfectly on `fault_detection_local` and would be unusable as a relay. See
> [task_cards/fault_detection.md](task_cards/fault_detection.md) and
> [task_cards/relay_selectivity.md](task_cards/relay_selectivity.md).

- **FC vocabulary (frozen, 9 classes):** the 7 universal SHC/incipient subtypes + the 2 MV-only HIF
  subtypes (`flt_1phg_hif`, `flt_1phg_hif_w_arc`), appended last so indices 0–6 are stable. HV/EHV
  grids simply have zero HIF support. See [taxonomy.md](taxonomy.md).
- Label derivation is pure functions of the windows `y` (`tasks/labels/`), valid-sample masks in
  `tasks/valid_samples.py`. A window outside a task’s valid set is never scored.

**Event-scope tasks** extend beyond faults across the full [taxonomy](taxonomy.md) and are
registered and scored (`global` only): **Event Detection** (relay trip/no-trip incl. benign
switching / `no_event` negatives — [task_cards/event_detection.md](task_cards/event_detection.md))
and **Event Classification** (the full event-type taxonomy —
[task_cards/event_classification.md](task_cards/event_classification.md)), alongside event state,
switch detection/type and fault origin ([task_cards/event_state.md](task_cards/event_state.md),
[task_cards/switch_tasks.md](task_cards/switch_tasks.md),
[task_cards/fault_attributes.md](task_cards/fault_attributes.md)).

## Protocols

**`held_out`** (current). Per grid:
- `train` / `val` / `test` = the grid’s **`adapt_grid`** family episodes split **70 / 15 / 15** by
  `sample_id` (episode-level, seeded → bit-reproducible; `splits/generator.py`, `BASE_SEED` +
  per-grid offset).
- `benchmark` = the grid’s **entire `benchmark`** family — a second, never-trained-on held-out set
  (the systematic discrete-parameter sweep). If that family is not yet windowed, the benchmark
  partition is **empty** and only the in-distribution `test` set is scored.
- Committed under `src/evemtbench/splits/v<SPLITS_VERSION>/held_out/<grid>/` (4 `.txt` files of
  `family:grid:sim_idx` triples + `meta.json`).

**Pre-training / cross-grid transfer** (`multi_grid_pretrain`, `transfer_zeroshot`,
`transfer_finetune`). Train on the diverse **`multi_grid`** pre-training corpus
(`template_20kv`/`template_110kv`/`template_345kv`) and generalize to the reference grids.
Scored at the **portable `local`/`line` views only** (fixed 6/12 channels):
`multi_grid`'s windows are stored in a 12-channel `EndA/EndB` "double-line" view (grid/line identity
in `y`, not the channel axis), which equals the reference grids' `line` view (and `EndA` = `local`),
so a model transfers without reshape. `global` is grid-specific width and out of scope (needs the GNN).
Design: [design/pretrain_transfer.md](design/pretrain_transfer.md).

- **`multi_grid_pretrain`** — `train`/`val`/`test` = a **topology-grouped** split (70/15/15 on
  episode counts, stratified by `event_type` at the group level) of the `multi_grid` family of the
  pre-training grid; **no benchmark family** (in-corpus). Produces the checkpoints the transfer
  protocols initialise from. Committed under `splits/v1.1/multi_grid_pretrain/<grid>/` (per-protocol
  version — `evemtbench.version.splits_version`; held_out stays `v1.0`), each with a committed
  machine-readable `audit.json` (per-partition topology/simulation/episode/fault counts, class
  support, pairwise topology/simulation/episode overlap counts — all zero — partition hashes,
  grouping strategy + seed).
  Grouping by topology keeps every simulation and every sibling line in one partition. A
  simulation-grouped variant exists (`evemtbench-splits --grouping simulation`). Leakage gates:
  `splits.leakage.assert_group_disjoint`, `verify_protocol_regenerates` and the per-grid
  `audit.json`. Corpus sizes: [design/pretrain_transfer.md](design/pretrain_transfer.md); why the
  splits changed in v1.1: [CHANGELOG](../CHANGELOG.md).
- **`transfer_zeroshot`** — initialise from that checkpoint and evaluate a reference grid's
  `adapt_grid` `test` and `benchmark` families with **no target training** (target topology unseen).
- **`transfer_finetune`** — initialise from the checkpoint, fine-tune on the reference grid's
  `adapt_grid` train split, then score its `adapt_grid` `test` and `benchmark` families (the same
  test sets as `held_out`; `coverage.expected_test_sets`). Reuses the target's committed
  `held_out` split at load time (no new split files).
- **Normalization:** these default to **per-window z-score** (each window standardized by its own
  per-channel mean/std over time) — leakage-free and voltage-invariant, so a 20 kV-pretrained model
  transfers across voltage levels. `held_out` keeps `none`. Recorded in `run_record.json`.
- **Baselines:** the deep baselines only (`mlp/gru/cnn/resnet/tst` — transferable weights). The target's
  in-distribution `held_out` majority is the reference floor.
- **Sources:** every transfer cell is scored once per pre-training source grid
  (`template_20kv`, `template_110kv`, `template_345kv`; `coverage.PRETRAIN_GRIDS`). Reporting
  groups results **by source** (matched- vs cross-voltage source→target pairs), never pooled
  across sources.

**`cross_topology` / GNN** (planned, not yet implemented). The `global`-view cross-grid track: a
graph-aware model consuming variable cubicle sets across the 4 reference grids. See
[design/cross_topology.md](design/cross_topology.md).

## Evaluation rule

For each seed: **fit on `task.train` only**, predict each **non-empty** held-out view, score with
the official metrics. Across seeds: report **mean ± std** and the **95% CI half-width** (`ci95`,
Student t across seeds; plus `n_defined` for metrics undefined on some seeds). Two held-out sets are reported per grid: in-distribution (`test`, from `adapt_grid`)
and the cross-distribution `benchmark` family. The runner (`baselines/runner.py`) writes one
`result.json` per `(task, protocol, grid, baseline, test_set)`; schema in
[submission.md](submission.md).

## Frozen vs. mutable (per [versioning.md](versioning.md))

**Frozen** within a benchmark major version: the metric set + directions, the label derivations,
the FC class vocabulary, the window contract (50 ms / 5 ms), and the committed splits. Changing any
of these is a **major** version bump. New grids, baselines, or protocols are **additive** and do
not bump the contract for existing results.

## Reporting

Results are conditional on **(task × observability × grid × protocol × test_set × metric)** — never
a single scalar. The leaderboard ([metrics.md](metrics.md)) ranks per task section by a headline
metric but always exposes the per-grid, per-test_set breakdown; aggregate numbers can hide failure
modes (e.g. a view that collapses on `benchmark` transfer).
