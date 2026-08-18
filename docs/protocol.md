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

> **The fault tasks are ORACLE tasks — read the "valid samples" column as part of the task
> definition, not as a filtering detail.** Only fault episodes are scored, so FD's negatives are the
> *pre-onset windows of episodes that do fault*; and at `local`/`line` the input cubicles are chosen
> by the ground-truth faulted line (`y_fault_line`). FD is therefore **line-conditioned onset
> detection**, not grid-wide detection, and FL presupposes a correctly identified faulted line. This
> is intentional — it holds information constant so representations can be compared (§2.4) — but it
> means deployed performance is **overstated** by construction: a model that trips on any disturbance
> scores near-perfectly on `fault_detection_local` and would be unusable as a relay. See
> [task_cards/fault_detection.md](task_cards/fault_detection.md) and
> [task_cards/relay_selectivity.md](task_cards/relay_selectivity.md).

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

- **`multi_grid_pretrain`** — `train`/`val`/`test` = a **topology-grouped** split (70/15/15 on
  episode counts, stratified by `event_type` at the group level) of the `multi_grid` family of the
  pre-training grid; **no benchmark family** (in-corpus). Produces the checkpoints the transfer
  protocols initialise from. Committed under `splits/v1.1/multi_grid_pretrain/<grid>/` (per-protocol
  version — `evemtbench.version.splits_version`; held_out stays `v1.0`), each with a committed
  machine-readable `audit.json` (per-partition topology/simulation/episode/fault counts, class
  support, pairwise topology/simulation/episode overlap counts — all zero — partition hashes,
  grouping strategy + seed).

  > **History — the v1.0 split-granularity deviation, fixed in v1.1.** A `multi_grid` episode is
  > *one line of one simulation* (the portable two-terminal view; `sample_id = sim_idx × 10⁴ +
  > line_idx`; see [design/pretrain_transfer.md](design/pretrain_transfer.md)). Corpus structure,
  > measured from the window metadata (`result_id` = simulation, `graph_id` = generated topology):
  >
  > | template | simulations | episodes | mean lines/sim | topologies |
  > |---|---|---|---|---|
  > | `template_20kv`  | 9 753 | 119 153 | 12.2 | 104 |
  > | `template_110kv` | 9 879 | 138 512 | 14.0 | 105 |
  > | `template_345kv` | 9 895 | 173 407 | 17.5 | 104 |
  >
  > The v1.0 split was a random 70/15/15 **episode** split, which put sibling lines of the same
  > simulation — same event, same operating state, same 0.5 s of grid history seen at other
  > terminals — into different partitions, and reused every topology across all three. No reported
  > benchmark number was contaminated (every scored metric lives on a reference grid whose
  > topology and simulations never appear in `multi_grid`, and whose own splits are
  > simulation-level as the dataset paper recommends), but the pre-training early-stopping and the
  > zero-shot *checkpoint selection* read that leaky `val`. The v1.1 splits group by **generated
  > topology** (`graph_id`; the dataset paper's recommendation for `multi_grid`), so no topology —
  > and therefore no simulation and no sibling line — crosses a partition boundary; a
  > simulation-grouped (`result_id`) variant is implemented as the documented fallback
  > (`evemtbench-splits --grouping {simulation,topology}`). Topology grouping shipped because the
  > metadata identifies topologies reliably (0 inconsistencies across all 431 072 episodes; both
  > ids live in the y-parquet, no aux overlay needed) and every event class spans 87–104 of ~104
  > topologies per corpus, so per-class support survives grouping. Leakage gates:
  > `splits.leakage.assert_group_disjoint`, `verify_protocol_regenerates`, and the committed
  > per-grid `audit.json`. Adopting the grouped splits invalidated the v1.0 pre-trained
  > checkpoints and transfer results (144 pre-training + 1 152 transfer cells → re-run under
  > `outputs/1.1.0/`; see `docs/versioning.md`).
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
