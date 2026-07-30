# Changelog

Benchmark-contract versions (see [docs/versioning.md](docs/versioning.md)). The `BENCHMARK_VERSION`
in `evemtbench.version` is stamped into every result; a **major** bump means scores are not
comparable across versions.

## 1.0.0 — full dataset re-simulation + multi_grid template grids (major; supersedes 0.2)

- **New dataset.** The entire EvEMTBench dataset was re-simulated and re-uploaded; every grid was
  re-windowed to the psp_helper window-contract 1.0.0. Split files regenerated (`splits/v1.0/`,
  `stratify_by=event_type`, 0.7/0.15/0.15). All pre-1.0 scores are NOT comparable; results land
  under `outputs/1.0.0/`.
- **multi_grid corpus extended.** Added `template_110kv` (HV) as a `multi_grid_pretrain` grid
  (double-line 12-channel view); `template_345kv` (EHV) is registered in `GRIDS` and joins the
  pretrain set once its windows/split land. New producer dataset configs `hv_template_110kv` /
  `ehv_template_345kv` and a grid-parametric `phaseB_line_view_alex.sh`.
- **GPU-utilisation fixes.** Pinned the projection-cache root in `hpc/sweep.sh` so the sweep finds
  `hpc/build_cache.sh`'s prebuilt caches (was silently `USE_CACHE=0` → per-job GPU-idle raw
  stage + inline cache build); made `build_cache.sh` family-aware; enabled the cache fast path for
  `multi_grid_pretrain`.
- **Audit fix.** `hpc/make_audit.py` / `hpc/audit_manifest.py` resolve split paths via
  `SPLITS_VERSION` (were hardcoded to `v0.2`).

**Why:** a full upstream re-simulation of the dataset. Splits are regenerated and the grid set
changed, so this is a major bump.

## 0.2.0-dev — family rename `fine_tune` → `adapt_grid` (major; supersedes 0.1)

- **Provenance hardening (results audit):** `run_record.json` additionally records the executing
  host (`hostname`, SLURM job id/nodelist, cluster), the CUDA device name, and input-data
  provenance — SHA-256 of every split file read plus the manifest fingerprint (`feature_hash`,
  `config_fingerprint`, `manifest_sha256`, shape) of each windowed partition actually loaded.
  Additive only; `result.json` unchanged; older records stay valid (see
  `src/evemtbench/baselines/run_record.schema.json`).
- **Results audit site:** `audit/` (built by `hpc/make_audit.py` from `hpc/audit_manifest.py`) —
  claim ledger, coverage matrix, verification-gate code tour, environment/reproduce/provenance
  pages and per-task/grid/baseline drill-downs; covered by the freshness gate and pre-commit hook.
- **Deep audit stacks** (adapted from UnderstandingVCS; all additive, `result.json` unchanged):
  - *Evaluator self-test* — `evemtbench-evaluate --self-test`: dataless synthetic-task controls
    (perfect-predictions ceiling, analytic 1/k floor, alignment must-raise, metric-direction
    sanity) through the real `evaluate()` (`src/evemtbench/evaluation/selftest.py`).
  - *Per-run harness positive control* — the runner scores the true labels before writing any
    `result.json`; anything but a perfect score aborts the cell. Recorded as
    `run_record.harness_control`.
  - *Snapshot self-check block* — `coverage_snapshot.json` embeds `self_check` (completeness,
    floor plausibility, value ranges hard; dominance findings reported); the freshness gate and
    `coverage --check-snapshot` exit non-zero on `ok:false`.
  - *Input digest pins* — committed `src/evemtbench/data/input_digests.json`
    (`--write`/`--verify`, self-hashing generator); `evemtbench.integrity` cross-checks every
    run record's loaded partitions against the pins.
  - *Golden fixture-metrics ratchet* — `tests/golden/fixture_metrics.json` pins exact
    deterministic-baseline aggregates; refresh only via `GOLDEN_UPDATE=1` in the same commit.
  - *Robustness page* — `audit/robustness.html` re-aggregates the committed headline numbers
    under alternative schemes (grid-equal mean, median, leave-one-grid-out, seed bootstrap CI)
    with a ranking-stability verdict; `significance.bootstrap_ci` added.
  - *Repro pack* — `bash hpc/make_repro_pack.sh`: clean `git archive` staging, live gate
    transcripts, and a leak-scan gate that refuses to write the zip on any forbidden string.

**Why:** the producer (`power-grid-and-ai-helper-functions`) renamed the in-distribution split
`fine_tune` → `adapt_grid` and now writes windows under `<root>/adapt_grid/windows/...`. The
benchmark adopts the same name so producer and consumer agree on the on-disk data contract.

- **Family relabel** `fine_tune` → `adapt_grid` across the API, protocol, splits, tasks,
  evaluation, HPC scripts, tests and docs. `resolve_data_root("fine_tune")` still resolves (read
  alias → `adapt_grid`) for backward compatibility.
- **Splits regenerated** as `src/evemtbench/splits/v0.2/` (same episode `sample_id`s, family
  prefix relabeled). `splits/v0.1/` retained read-only for historical attribution.
- **Version** bumped `BENCHMARK_VERSION` `0.1.0-dev` → `0.2.0-dev`, `SPLITS_VERSION` `0.1` → `0.2`.
  This is a **major** bump: **v0.1 scores are retired and NOT comparable**; v0.2 results pending
  re-run on the `adapt_grid` data.
- **Unchanged:** window contract (`W0p050`, step 5 ms, seq len 480), metrics, derived labels, and
  the task set.
- **Validity plumbing (M5, additive within 0.2.0-dev):**
  - `aggregate.<metric>.ci95` — 95% CI half-width across seeds (Student t, ddof=1; `null` below
    2 seeds) written by the runner next to the existing `mean`/`std` (std stays population/ddof=0).
  - New `evemtbench.evaluation.significance` (`ci95_halfwidth`, `paired_seed_test` paired **by
    seed id**, `compare_result_files` + `evemtbench-significance` CLI).
  - `RESULTS.md` and the leaderboard render **mean ± 95% CI** (derived from `per_seed` for
    artifacts written before `ci95` existed — no re-run needed).
  - `python -m evemtbench.coverage --check-floor` — majority-floor sanity gate: majority
    balanced accuracy ≈ the analytic `1/k` floor over classes present; no learned baseline below
    the majority floor on `test` (direction-aware; pending cells skip, never fail).
  - `run_record.json` gains the **no-tuning attestation** (fit on train, select on val, held-out
    sets predict-only), `splits_version`, and numpy/pandas/scikit-learn/scipy versions.
  - CI now runs a dedicated leakage/split-audit step; a new test validates the **committed
    v0.2 splits** (episode-disjoint, family-pure) directly from the packaged files.
  - `scipy` declared as a direct dependency (was transitive via scikit-learn);
    `requirements.lock` regenerated — future `run_record.lockfile_sha256` values change.
- **BYO enablement (M6, additive):** top-level `list_tasks`/`get_task`/`TaskDefinition`/
  `load_arrays` (dense arrays with a loud big-`global`-view guard); baseline **plugin registry**
  (`baselines.runner.register_baseline` on the `Baseline` protocol — third-party models run
  through the official runner with identical artifacts); `examples/byo_model.py`.
  `find_window_files` now narrows a **multi-geometry** windows dir by the pinned window
  contract (`W0p050_S0p005`) instead of failing "expected exactly one" — plain `load_task`
  works on production mounts that also carry other configs (single-config-read fix, an M7
  split-regeneration prerequisite; ambiguity without a contract still fails loud).
- **Committed splits regenerated STRATIFIED (M7, pre-release; supersedes all interim results):**
  `held_out` splits for `double_line`/`testgrid_110kv`/`cigre_mv` regenerated with
  `stratify_by="event_type"` at the same per-grid seeds; **`ieee39` too** once its `adapt_grid`
  family finished windowing (audit-clean, n=9999). `meta.json` records `stratify_by` and
  `leakage.verify_held_out_regenerates` now reproduces the recorded stratification
  (`evemtbench-splits --stratify-by`). All results produced on the uniform v0.2 splits are
  **retired** (archived locally under `outputs_archive/`, never rendered); the full matrix is
  re-swept on the stratified splits. Handled inside the `0.2.0-dev` pre-release — no released
  version ever carried the uniform v0.2 splits.
- **Docs reorg + design seams (M6):** `docs/README.md` audience index; `docs/DATA_LAYOUT.md`
  (the on-disk contract); ops/agent/history docs moved to `docs/internal/`. Open Challenges get
  frozen interfaces that raise until built: `tasks/vantage.py` (relay selectivity),
  `evaluation/robustness.py` (measurement-realism perturbation taxonomy), `baselines/gnn.py`
  (cross-topology GNN) + `docs/design/{selectivity,robustness,cross_topology}.md` and the
  `relay_selectivity` task card.

## 0.1.0-dev — multi-grid expansion (supersedes the DoubleLine-only scope below)

Within `0.1.0-dev` the benchmark grew from a DoubleLine pilot to a **four-grid** benchmark; the
"deliberately narrow" scope in the original entry below is **superseded**.

- **Four reference grids** across levels: CigreMV (20 kV MV), DoubleLine (110 kV HV),
  TestGrid110kV (110 kV HV), IEEE-39 (345 kV EHV, 60 Hz). Observability oracle now **derived from
  each grid's `feature_names`** (not DoubleLine-hardcoded); `global` width is grid-dependent
  (48–420 ch). IEEE-39 is wired in fine_tune-first (benchmark-not-yet-windowed support).
- **FC is now a 9-class taxonomy** — the 7 universal SHC/incipient subtypes **+ 2 MV-only HIF**
  subtypes (`flt_1phg_hif`, `flt_1phg_hif_w_arc`), appended so indices 0–6 are stable. (A label
  contract change; pre-release, so handled within `0.1.0-dev`. FC results re-run.)
- **`cnn` baseline** (1-D waveform) added; `gnn` planned for a future `cross_topology` protocol.
- **Protection-grade analysis** (`analysis.py`): detection latency + location difficulty curves.
- **Memory handling** for the wide `global` view: chunked projection + per-batch standardization.
- **Docs:** added `docs/{taxonomy,grids,protocol,metrics,submission,baselines,extending}.md`;
  un-staled README/datasheet/task-cards to the multi-grid scope.
- **Ops:** coverage manifest (`evemtbench.coverage`), grid-parametric `hpc/sweep.sh` + `hpc/submit.sh`,
  and the hands-off `hpc/autonomous_loop.sh`.

## 0.1.0-dev — full reset onto the reconfigured dataset (DoubleLine, FD/FC/FL)

**Why a fresh start:** the data simulation was reconfigured and the regenerated corpus **is now
the EvEMTBench** — it replaces the old dataset entirely. Everything prior (4 grids, 11 event/fault tasks, the in_grid /
cross_grid_zeroshot / fine_tune protocols, all v2.0 splits and results, the IEEE39 watcher) is
**retired and not comparable**. Version reset to `0.1.0-dev` / splits `0.1`.

### Scope (deliberately narrow)
- **DoubleLine only** (3-bus double-circuit, 110 kV). Of the 66 on-disk channels the benchmark
  uses **only the 4 main lines** (8 line cubicles, both ends) = **48 channels**; transformer and
  load cubicles are dropped.
- **9 tasks** = {**FD** fault detection (binary), **FC** fault classification (full `flt_*`
  taxonomy, 7 classes, multiclass), **FL** fault location (**regression** on `y_fault_location` %,
  line faults only)} × {**local** (1 cubicle, 6 ch), **line** (faulted line, both ends, 12 ch),
  **global** (all 4 main lines, 48 ch)}. Observability is a nested oracle keyed on the
  episode-constant `y_fault_line`; bus faults map to the incident line bays.
- FD is within-fault-episode onset detection (fault-active window vs. a pre-fault window of the
  same episode), so the local/line/global oracle is defined for every scored window.

### Protocol
- Single **`held_out`** protocol, 4 partitions: the `fine_tune` family split 70/15/15
  (train/val/**test**, in-distribution) plus the **whole `benchmark` family** as a second,
  never-trained-on held-out test. The runner fits once and evaluates BOTH sets, writing one
  `result.json` per `test_set`.

### Data / windows
- Window contract changed to **50 ms windows** (`W0p050`, seq len l=480), 5 ms step.

### Framework
- Added a **REGRESSION** label kind end-to-end: `mae` / `rmse` / `r2` / `median_abs_error` and a
  `METRIC_DIRECTION` table; regression baselines (mean predictor, `RandomForestRegressor`,
  MSE-head MLP with target standardization). The leaderboard picks a per-task headline metric
  (balanced_accuracy for classification, mae for regression) and orients bars/sorting by direction.
- Removed the retired grids/tasks/protocols and their split generators, label modules, and tests.

## 2.0.0-dev — event coverage + upstream label fix (major)

**Why a major bump:** an upstream label fix changed fault labels and the committed splits were
regenerated — results on v1.x are **not comparable** to v2.x.

### Data / labels (upstream `psp_helper`)
- Fixed `flt_1phg_hif_w_arc` being mislabeled `event_family="other"` (it fell through a
  hand-maintained map to `UNKNOWN`). Because window `status` keys off family, this had also forced
  `y_fault_category=None`, `y_is_grounded=None`, and `y_fault_present=0` for every HIF-with-arc
  window — corrupting fault labels on CIGRE MV. Hardened the resolvers with prefix fallbacks +
  guard tests so no `flt_*`/`switch_*` type can silently fall through again.
  (power-grid-and-ai-helper-functions MR !11.)
- **Re-windowed CIGRE MV** (the only grid emitting the type). This also **completed** the CIGRE
  fine_tune windowing, which had been incomplete: **341 → 993 episodes** (9,889 → 28,797 windows).
  All partitions verified: 0 `event_family=="other"`, complete manifests, no stale per-batch files.

### Splits (major — regenerated)
- `SPLITS_VERSION 1.0 → 2.0`. All committed splits regenerated from the corrected/complete data.
  CIGRE `in_grid` 341 → 993 episodes (now 695/149/149); `cross_grid_zeroshot` train sets for
  `double_line` and `testgrid_110kv` grew (CIGRE is a source). `v1.0` split files removed.

### Benchmark (new tasks)
- General event tasks beyond faults: `event_detection_{local,wide_area,topology}`,
  `fault_vs_event_local` (fault vs. benign switching event), `event_family_local` (normal/fault/
  switching), `event_type_local` (22-class). Labels consumed from the pipeline's per-window columns;
  frozen taxonomies; protection-named metrics (`missed_event_rate`, `false_event_trip_rate`).

### Notes
- Pre-v2.0 LME results in `RESULTS.md` were on `double_line` (unaffected by the CIGRE fix); CIGRE
  and cross-grid numbers should be (re-)run on v2.0 splits.

## 1.0.0-dev — initial benchmark (superseded)
- Three-layer package (data / tasks / splits / evaluation; import-isolated baselines), committed
  episode-level splits + leakage gates, `evaluate()` + CLI, baselines (majority / threshold /
  random_forest / mlp / deepsets), fault tasks (detection, type) across local / wide-area /
  topology input settings, lazy loader for cross-grid topology, internal leaderboard.
