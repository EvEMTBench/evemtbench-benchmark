# EvEMTBench — extending & operating the benchmark

For maintainers/collaborators: how to add grids, tasks, and baselines, and how the HPC run/results
machinery works. Public-facing usage is in [submission.md](submission.md). **How to *operate* the
benchmark** (dispatch checklist, cadence tiers, loop/watcher conventions, results-integrity gate) is
in [operating.md](internal/operating.md); dated mistakes + fixes are in [lessons.md](internal/lessons.md). The
engineering *principles* to apply when writing/changing code (SWE + MLOps, with primary sources) are in
[internal/agent/engineering.md](internal/agent/engineering.md).

## Add a grid

1. Register a `GridSpec` in [`data/roots.py`](../src/evemtbench/data/roots.py) `GRIDS`
   (key, topology id, raw folder, voltage, frequency, level).
2. Generate its split: `python -m evemtbench.splits.generator --grids <key>` — writes
   `splits/v<ver>/held_out/<key>/`. If the `benchmark` family isn't windowed yet, the generator
   writes an **empty** benchmark partition (in-distribution `test` only) and the runner/coverage
   handle it; re-run the generator when the benchmark lands.
3. **Nothing else** — the observability oracle derives automatically from that grid's
   `feature_names` ([`tasks/line_view.py`](../src/evemtbench/tasks/line_view.py) `build_oracle`:
   keep `Bus_<bus>_Line_<id>` cubicles, pair terminals sharing a `lineid`). Verify with a quick
   `build_oracle(key, feature_names)` check (line count, all `y_fault_line` labels resolve).

## Add a task

`taxonomy.py` (new `Function`) → `tasks/labels/<x>.py` (pure deriver + frozen class vocab) →
`valid_samples.py` (admissible-window mask) → `task.py::_LABEL_DERIVERS` → `registry.py` (`_make`)
→ `evaluation/evaluate.py::_CLASS_VOCAB` (multiclass). Keep a window's label `-1` only for
windows the valid mask excludes; never feed `-1` to a CrossEntropy head (the FC/HIF bug — a
fault type outside the vocab must be excluded by `valid_samples`).

## Add a baseline

Implement `fit(task)` / `predict(view) -> predictions_frame` (mirror `baselines/cnn.py`); register
in `runner.py::_make_baseline`, the `--baseline` choices, and `coverage.py::ALL_BASELINES`
(+ `baseline_applies` if it's restricted, like `threshold`). See [baselines.md](baselines.md).

## Add a protocol

`splits/protocols.py` (`Protocol` entry) → `task.py::_protocol_triples` (which `(family, grid)` each
split draws from) → the task's `protocols` tuple in `registry.py` → `coverage.py`
(`protocol_grid_applies` / `baseline_applies` / `expected_test_sets` for grid & baseline scope). If
the protocol needs committed split files, add it to `generator.py::_GENERATED_PROTOCOLS` and commit
`splits/v<ver>/<protocol>/<grid>/`. The pre-training/transfer protocols
([design/pretrain_transfer.md](design/pretrain_transfer.md)) are the worked example: the `multi_grid`
double-line view (`tasks/line_view.py`), per-window normalization (`task.py::build_input_transform`),
and checkpoint transfer (`baselines/_transfer.py`). Adding a protocol is a **minor** bump (additive;
existing `held_out` results stay comparable).

## Run machinery

- **Coverage manifest** — [`coverage.py`](../src/evemtbench/coverage.py): the expected matrix
  (tasks × grids × protocols × applicable baselines × test_sets) vs. produced `result.json` →
  `done / pending / failed`. The single source of truth for skip-existing + progress.
  `python -m evemtbench.coverage --summary | --emit-pending`.
- **`hpc/sweep.sh`** — grid-parametric worker (Alex): runs only the manifest-pending cells for one
  grid. With prebuilt projection caches it stages **lazily per cell** (the current cell's ≤4 view
  files, atomic copy, one-task-ahead background prefetch); without a complete cache it falls back to
  staging the grid's canonical `W0p050_S0p005` windows to node-local `$TMPDIR` (7–14 TB SSD;
  `hpc/lib_stage.sh`) and projecting once per job — reading the `$WORK`/NFS mount per epoch idles
  the GPU. Writes `result.json` directly under `outputs/`.
- **`hpc/submit.sh`** — submit driver (login node, **dry-run by default**; `--submit` to queue).
  Queues GPU jobs per grid only where cells are pending; an all-classical `BASELINES` set
  auto-routes to a MIG `a100small` slice (`a100mig`, `MIG_CPUS` cores). Knobs: `JOBTAG` (per-grid
  job-name suffix, used by the loop's two waves), per-grid `SOFT_DEADLINE_SEC` defaults
  (`grid_soft_deadline`). `gres` selects the partition (`gpu:a40:1` default; `GRES=gpu:a100:1`);
  no `--partition`, no node excludes, no `--mem`.
- A single cell runs via `sweep.sh` with a narrow `TASKS`/`BASELINES` selection, or via the runner
  directly (`python -m evemtbench.baselines.runner`). The development repo additionally runs a
  hands-off submission loop (hourly refresh+push of `RESULTS.md`/`leaderboard.html`, idle-guarded
  resubmission via `submit.sh` with a no-progress halt) and profiling wrappers around the same
  entrypoints.
- **`hpc/make_results.py`** regenerates `RESULTS.md`; `evemtbench-leaderboard` regenerates the HTML.
- **`hpc/make_results_bundle.py`** regenerates the public `results/<version>/` bundle (aggregated
  `results.{csv,json}` + the LaTeX `tables/`); **`hpc/make_audit.py`** the audit site + provenance index.

### Add a committed derived view (the pattern)
Every committed artifact derived from `outputs/` follows one contract: a **deterministic, pure
builder** is the single source of truth, and the **freshness gate** (`hpc/check_results_fresh.py`)
byte-compares the committed file against a fresh rebuild (pre-commit hook + loop). To add one, mirror
the results bundle / provenance index: (1) write a pure `build_*() -> {relpath: content}` builder (no
clock, sorted, version-scoped via `BENCHMARK_VERSION`); (2) in `check_results_fresh.py` add a
`_fresh_*()` wrapper, a `checks += [...]` block, a `--fix` regeneration call, and stray-detection for
files a rebuild no longer emits; (3) if it lives at a new top-level path, un-ignore it in `.gitignore`
and add it to the pre-commit trigger regex; (4) add it to the development repo's publisher
`git add` lists and CI upload; (5) document it in `docs/DATA_LAYOUT.md`.

## Memory routing (the `global` view on big grids)

The `global` view scales with topology (48 → 420 ch). Two mechanisms keep it in budget:
- **Chunked projection** ([`task.py::_observe_chunked`](../src/evemtbench/tasks/task.py)) — reads
  the memmap and projects per row-chunk, bounding the full-channel transient.
- **Per-batch standardization** (mlp/gru/cnn) — no full standardized copy; predict moves batches to
  the GPU one at a time (a 33 GB view won't fit a 16 GB GPU otherwise).

With these two mechanisms the `global` view runs on a single GPU at **`GPU_MEM=50G`** for every grid
(streaming view; per-batch standardization never materializes the full array) — the loop submits one
GPU job per global family (FD/FC/FL) so they run concurrently. IEEE-39 just gets a longer walltime
(`TIME=23:00:00`); the classical CPU baselines run at `CPU_MEM=120G`.

## Gotchas & lessons

Operating mistakes (duplicate loops, NFS-idle staging, noisy watchers, result drift, …) and the
standing gotchas (`GROUPS` builtin, running-cells-read-as-pending, refit-from-scratch) now live in
the dated [lessons.md](internal/lessons.md) log. Add a row there when something bites.
