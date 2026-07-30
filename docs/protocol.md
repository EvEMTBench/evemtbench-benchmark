# EvEMTBench — benchmark protocol (canonical contract)

The single source of truth for **what is scored and how**. Task cards and the quickstart link
here rather than restating it. Grids are documented in [grids.md](grids.md), events in
[taxonomy.md](taxonomy.md), metrics in [metrics.md](metrics.md), versioning policy in
[versioning.md](versioning.md).

## Task matrix

Tasks = **{FD, FC, FL} × {local, line, global}** = 9 task ids per grid
(`<function>_<observability>`, e.g. `fault_detection_line`). Observability is defined in
[grids.md](grids.md).

| function | id stem | label kind | classes | valid samples (windows scored) |
|---|---|---|---|---|
| Fault **detection** (FD) | `fault_detection` | binary | 2 | all windows of fault episodes (`y_fault_line` defined): pre-fault negatives + fault-active positives |
| Fault **classification** (FC) | `fault_classification` | multiclass | **9** | fault-active windows whose fault type is in the 9-class vocab |
| Fault **location** (FL) | `fault_location` | regression | — | fault-active windows of **line**-fault episodes (`y_fault_location` defined; NaN on bus faults) |

- **FC vocabulary (frozen, 9 classes):** the 7 universal SHC/incipient subtypes + the 2 MV-only HIF
  subtypes (`flt_1phg_hif`, `flt_1phg_hif_w_arc`), appended last so indices 0–6 are stable. HV/EHV
  grids simply have zero HIF support. See [taxonomy.md](taxonomy.md).
- Label derivation is pure functions of the windows `y` (`tasks/labels/`), valid-sample masks in
  `tasks/valid_samples.py`. A window outside a task’s valid set is never scored.

**Planned (not yet implemented/scored) — event-scope tasks** that extend beyond faults across the
full [taxonomy](taxonomy.md): **Event Detection** (relay trip/no-trip incl. benign switching /
`no_event` negatives — [task_cards/event_detection.md](task_cards/event_detection.md)) and **Event
Classification** (family 3-class, then the full event-type taxonomy —
[task_cards/event_classification.md](task_cards/event_classification.md)). Those cards are the
implementation contract; the data already supports them.

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
`transfer_finetune`). Train on the diverse **`multi_grid`** pre-training corpus (formerly `train`;
`template_20kv`/`template_110kv`/`template_345kv`) and generalize to the reference grids — the cross-grid-generalization track
(paper goals G2/G3). Scored at the **portable `local`/`line` views only** (fixed 6/12 channels):
`multi_grid`'s windows are stored in a 12-channel `EndA/EndB` "double-line" view (grid/line identity
in `y`, not the channel axis), which equals the reference grids' `line` view (and `EndA` = `local`),
so a model transfers without reshape. `global` is grid-specific width and out of scope (needs the GNN).
Design: [design/pretrain_transfer.md](design/pretrain_transfer.md).

- **`multi_grid_pretrain`** — `train`/`val`/`test` = an episode split (70/15/15, stratified by
  `event_type`) of the `multi_grid` family of the pre-training grid; **no benchmark family** (in-corpus).
  Produces the checkpoints the transfer protocols initialise from. Committed under
  `splits/v<SPLITS_VERSION>/multi_grid_pretrain/<grid>/`.
- **`transfer_zeroshot`** — initialise from that checkpoint and evaluate a reference grid's
  `adapt_grid` `test` and `benchmark` families with **no target training** (target topology unseen).
- **`transfer_finetune`** — initialise from the checkpoint, fine-tune on the reference grid's
  `adapt_grid` train split, then score its `benchmark` family. Reuses the target's committed
  `held_out` split at load time (no new split files).
- **Normalization:** these default to **per-window z-score** (each window standardized by its own
  per-channel mean/std over time) — leakage-free and voltage-invariant, so a 20 kV-pretrained model
  transfers across voltage levels. `held_out` keeps `none`. Recorded in `run_record.json`.
- **Baselines:** the deep baselines only (`mlp/gru/cnn/resnet` — transferable weights). The target's
  in-distribution `held_out` majority is the reference floor.
- Reporting groups targets by **voltage match vs the 20 kV source** (same-voltage `cigre_mv` vs
  cross-voltage 110/345 kV), never pooled.

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
