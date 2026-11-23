# Changelog

Benchmark-contract versions (see [docs/versioning.md](docs/versioning.md)). The `BENCHMARK_VERSION`
in `evemtbench.version` is stamped into every result; a **major** bump means scores are not
comparable across versions.

## In the 1.3.0 tag — reproduction guides, paper builders and a cleanup (docs and tooling; no contract change)
- **Dataset DOI.** Dataset v0.1, the data behind every 1.x result, is registered as
  `10.48742/fau.abpd-r521`; the README and the datasheet name it.
- **Fault-versus-switching diagnostic files (SI Table S13).** `results/diagnostics/fault_vs_switching/`
  holds the output of `hpc/diag_fault_vs_switching.py` for the reference runs: per-seed and
  seed-summary rates and confusion shares, with supports and breakdowns by event type and switching
  category, the control and selection records, the input digests and the table. No result changes.
- **SI captions.** The captions of the split-composition and valid-count tables (SI Tables S17 and
  S18) follow the manuscript wording; the numbers are unchanged.
- **Fig. 3 builder.** `hpc/paper_artifacts/build_fig_transfer_overview.py` draws the matched
  target-only reference (dashed segment with its 95 % interval; arm C of the input-normalization
  control) behind each zero-shot/fine-tune pair, as in the paper's figure. It reads the control tree
  passed with `--abl`. The plotted transfer values are unchanged.
- **Tables 2-4 builders.** `build_tab_dataset_families.py`, `build_tab_task_spec.py` and
  `build_tab_baselines.py` emit the table environments of the revised manuscript verbatim (the
  baselines table now lists the impedance locator and no planned GNN).
- **Docs: no dedicated `no_event` episodes.** `docs/taxonomy.md`, the datasheet and three task cards
  said the two 110 kV grids contain quiescent `no_event` episodes. They do not (SI Table S15): the
  `no_event` / `normal` label comes from inactive windows of fault and switching episodes. The
  episode counts in `docs/taxonomy.md` are now the `adapt_grid` counts of the committed splits, and
  `docs/baselines.md` names the threshold statistic as it is computed (largest per-phase RMS current).
  The datasheet's operating-point note follows SI Table S14, and the audit's environment page names
  the A100 MIG slices the classical baselines ran on.

- **`reproduce/`.** One guide per group of paper results: where the numbers are, how to retrain
  them, and which builder rebuilds each table and figure. The README and the public docs are
  shorter and point there.
- **Normalisation control (SI Table S10).** The per-window control tree now holds the TST, so all
  three arms select among the same five deep baselines, and arm B's pre-training source is chosen
  on validation, never on the evaluation grid (`hpc/paper_artifacts/build_supp_tab_norm_control.py`).
- **Leaderboard.** `main`'s redesign is merged: `leaderboard.html` is the ranked page, the full
  tables are in `leaderboard_full.html`.
- **Cleanup.** One-off and superseded `hpc/` scripts, stale build outputs, the root
  `requirements.txt` and three unused helpers are removed; `tqdm`, `hydra-core` and `xgboost` are no
  longer dependencies, and the `baselines` extra now pulls `networkx` for the impedance locator.
  Stale lines in the public docs are fixed. No result, metric, split or task changes.
- **Release metadata for the software archive.** `CITATION.cff` (read by Zenodo) names the software
  "EvEMTBench Benchmark", lists all six paper authors with affiliations, dates the release
  2026-11-23, adds keywords and an abstract, and cites dataset v0.1 by its DOI and the
  data-generation code (doi:10.5281/zenodo.21624649); `pyproject.toml` matches.
- **Paper names.** The README and `reproduce/` map the paper's names (`eval_grid`, Protection Reference
  grid) to the identifiers used here (`benchmark`, `testgrid_110kv`). The scheduled `nightly` workflow is
  removed; the push-triggered `ci` runs the same checks.
- **Input and output guards (external code review; no result changes).** `load_task` rejects a single
  `directory` override for a task that reads more than one partition, and a split that lists an
  episode absent from the partition. The evaluator rejects null or duplicate reference keys.
  `input_digests --verify` checks the label and channel-descriptor files byte for byte (`--deep`:
  the waveform arrays too, where pinned). The runner stages predictions per attempt, so a failed
  re-run leaves the previous predictions, result and record intact. A non-finite validation mean
  blocks model selection. `make_results_bundle.py` refuses to write without result records. The
  evaluator JSON names the transfer source; the table preamble defines every macro the tables use;
  the quickstart activates its environment and has a data-free example.
  `check_results_fresh.py` reports a run without `outputs/` as SKIPPED, not as a pass (`--strict`:
  exit code 3).

## 1.3.0 — faulted phase of `flt_1phg_hif_w_arc` restored by a label patch (scoped revision, by cell)

- **The defect.** In the 1.x windows (`W0p050_S0p005`) every fault-active window of the
  high-impedance single-phase-to-ground fault with arc (`flt_1phg_hif_w_arc`) carries
  `y_phase_A = y_phase_B = y_phase_C = 0`, so the `fault_phase` valid-sample rule (at least one
  phase bit) dropped the whole class from training, validation selection and the test scores. The
  producer's one-phase rule (`psp_helper` `windows_helper._phases_from_select`) lists every other
  `flt_1phg*` type but not this one, and an unknown type silently falls through to "no phase".
  Affected, and nothing else (no other type, grid or family has a fault-active window without a
  phase bit):

  | partition | windows | episodes |
  |---|--:|--:|
  | CIGRE MV `adapt_grid` | 7,752 | 456 |
  | CIGRE MV `benchmark` | 1,392 | 87 |
  | `multi_grid` Template20kV (line view) | 4,318 | 254 line episodes |

  The source label `events/event_phase_select_1ph` names exactly one phase for every one of these
  episodes, and for every other fault type the window bits equal the source label on 100 % of
  episodes (`docs/known_issues.md` §4).
- **The fix: a committed label patch, not a re-window.** The windows (X, y parquet, manifests),
  splits, window contract and label rules are unchanged. `src/evemtbench/data/label_patches/`
  holds patch **LP-1**: an episode-keyed CSV (797 rows: grid, family, `sample_id`, the source
  selector and the three phase bits; a/b/c -> A/B/C, the producer's rule for the other 1phg types)
  and `index.json`, which pins the CSV sha256 and, per partition, the `manifest_sha256` it applies
  to, the episode count and the number of patched windows. `hpc/build_label_patch.py` builds both
  deterministically from the 1.x source labels (aux `settings_clean.csv` for CIGRE MV, the
  archive's `Template20kV/labels/settings_clean.csv` for the multi_grid corpus) and writes an audit
  table. `load_task` applies it only to `fault_phase` tasks, only for a v1.3.0 cell, and only on
  the exact partition it was built for; any target that is not a fault-active, phase-less window
  of the type, a count that differs from the pin, or an edited CSV raises `LabelPatchError`.
- **Valid samples gained** (fault-active windows that are `fault_phase`-valid):

  | grid / split | before | after | added |
  |---|--:|--:|--:|
  | CIGRE MV `adapt_grid` train / val / test | 45,275 / 9,671 / 9,671 | 50,698 / 10,827 / 10,844 | +12.0 % / +12.0 % / +12.1 % |
  | CIGRE MV `benchmark` | 67,212 | 68,604 | +2.1 % |
  | Template20kV train / val / test | 21,480 / 4,183 / 4,353 | 24,489 / 4,863 / 4,982 | +14.0 % / +16.3 % / +14.4 % |

  The restored windows fall into the single-phase classes A, B and C.
- **Scope of invalidation (per cell).** `results_version(protocol, task, *, grid=..., source=...)`
  returns `1.3.0` for exactly the 151 `fault_phase` cells that read a repaired label (755 seed
  runs), which are re-run into `outputs/1.3.0/`:
  - `held_out` on cigre_mv, `fault_phase_{local,line,global}` (21 cells);
  - `multi_grid_pretrain` on template_20kv, `fault_phase_{local,line}` (10);
  - `transfer_zeroshot` and `transfer_finetune`, `fault_phase_{local,line}`, targeting cigre_mv
    from every source (60) or targeting double_line / testgrid_110kv / ieee39 from the
    template_20kv source, whose source models were trained without the class (60).

  The other 203 `fault_phase` cells and every other task keep their trees and stamps (1.0.0 /
  1.1.0 / 1.2.0) and are not re-run. For a `fault_phase` task `results_version` raises without
  `grid`, so no caller can resolve the wrong tree. `CURRENT_RESULTS_VERSIONS` gains `1.3.0`; every
  reader of `outputs/` resolves a row's tree from its task, protocol, grid and source
  (`coverage.cell_results_version` / `cell_from_key`).
- **Re-evaluation stays exact.** The patch is off for any record stamped before 1.3.0, so the
  1.0.0 / 1.1.0 records re-evaluate on the labels they were computed with.
- **Provenance.** `task.provenance["label_patches"]` (patch id, CSV sha256, grid, family,
  manifest, episodes, patched windows) lands in `run_record.json["data"]` (schema: optional
  `data.label_patches`). `integrity` checks that every v1.3.0 `fault_phase` record carries exactly
  the patches its partitions call for (and older records none) and that the CSV matches its pin;
  `reevaluate` adds the patch list to its provenance-equality control.
- **Unchanged:** the projection cache (`DATA_CACHE_VERSION` stays `1.1.0`; the larger views get new
  content-addressed files), `input_digests.json`, the splits, `contract/contract.json`,
  `tasks/labels/fault_phase.py`, `tasks/valid_samples.py`.
- `coverage_snapshot.json`: every cell row carries its `results_version`; the `fault_phase` tasks
  are not listed in `results_versions_by_task` (their version depends on the cell).
- `evemtbench-evaluate` and `load_task` gain `pretrain_grid` (the transfer source, part of the
  per-cell version); `load_task` gains `label_patches` (None = resolved from the cell's version).
- Also documented, not fixed in 1.x: `multi_grid` switching and busbar-fault episodes appear only
  as `no_event` windows (`docs/known_issues.md` §5).

**Why:** a derived label changed for one task family on two grids, so the comparability of exactly
the cells that read it is invalidated (versioning policy, scoped revision). Repairing the labels
through a committed, pinned overlay keeps the windows byte-identical, keeps every older result
reproducible, and makes the repair reviewable in one small file; the producer is fixed for 2.0,
which deletes the overlay.

## In the 1.3.0 tag — conventional fault-locator baseline `impedance` (additive; no contract change)

A physical reference for the fault-location (FL) tasks, in answer to the review request for an
established fault-localisation method (EXP-06). No task, split, metric, valid-sample rule or
`result.json` field changes, and no existing result is re-scored.

- **The baseline.** `baselines/impedance.py` (signal layer `baselines/_phasor.py`): full-cycle DFT
  phasors on the last complete cycle of a window; the one-terminal reactance method at `local`
  (current-magnitude phase selector, zero-sequence compensation on ground loops) and the
  two-terminal positive-sequence method at `line`. Deterministic, nothing is learned. `global` is
  not applicable. See [docs/baselines.md](docs/baselines.md).
- **Inputs.** The line's series impedance and length come from the grid topology graph (aux
  overlay, `adapt_grid` family, the only one with a graph for all four reference grids), looked up
  by `y_fault_line`, the oracle the `local`/`line` views are already centred on. The baseline is
  the first to need the aux overlay at run time; `hpc/sweep.sh` now exports `EVEMTBENCH_AUX_ROOT`
  next to the source data root, because the staged data root has no such sibling.
- **No calibration by default.** The channels carry primary-level values, so the
  secondary-to-primary impedance scale is 1. `impedance_scale` remains an explicit parameter (a
  known VT/CT ratio, or `"fit"` on `train` only). The scale used, the line-parameter source, the
  nominal frequency, the measured voltage level and the lines with one measured terminal are
  written to the new, additive `run_record.json["calibration"]` field (present only for baselines
  that expose `calibration_`; older records stay valid).
- **Scope and registration.** `coverage.baseline_applies`: FL at `local` and `line`, `held_out`
  only, on the four reference grids (8 new cells). It is listed in the new
  `coverage.CONVENTIONAL_BASELINES` next to `threshold` and is not in `LEARNED_BASELINES`, so it
  is outside the validation-selected reference pool and the floor gate's dominance check.
  `TaskView.feature_axes()` is a new read-only accessor (which terminals a view measures).
- **Reported, not guessed.** A line without usable parameters stops the run with the reason. The
  one-ended CIGRE MV line 8-14 ([docs/known_issues.md](docs/known_issues.md) §1) falls back to the
  one-terminal method in the `line` view and is mirrored to the label's reference end.
- **Views.** `RESULTS.md`, the audit site, `coverage_snapshot.json` and the completed-runs
  overview enumerate the new baseline and need a rebuild once its cells have run. The manuscript
  builders (`hpc/paper_artifacts/`) know its label and n/a rule but leave it out of the manuscript
  baseline set until its place in the manuscript is decided, so every generated manuscript
  artifact is unchanged.

## In the 1.3.0 tag — transformer reference baseline `tst` (additive; no task, split or metric change)

- **New deep baseline `tst`** (`baselines/tst.py`): the Time Series Transformer encoder of Zerveas
  et al. (KDD 2021, arXiv:2010.02803), trained from scratch under the shared deep-baseline regime
  (`baselines/_train.py`: AdamW, cosine schedule, validation early stopping, 5 seeds). Patch
  embedding `Conv1d(C, 128, kernel 16, stride 8)` (59 tokens for the 480-sample window), learnable
  positional encoding, 3 encoder blocks (`d_model` 128, 8 heads, feed-forward 256, GELU, dropout
  0.1, batch normalisation in place of layer normalisation), linear head on the concatenated token
  outputs. The configuration is fixed a priori; nothing is tuned on `test` or `benchmark`.
- **Deviations from the published model:** the strided patch embedding (the source makes one token
  per time step), 8 attention heads (the source's generic configuration has 16), and a LayerNorm on
  the patch tokens after the `sqrt(d_model)`-scaled projection. The first sweep without the token
  norm diverged to NaN on the wide `global` views (27 seeds in 11 cells) and stopped early on every
  view; gradient clipping did not prevent it. Those outputs are archived and every `tst` cell was
  re-run with the token norm (issue #26). The shared
  optimiser and batch size are kept in place of the source repository's RAdam / batch 128.
- **Scope.** `tst` is appended to `coverage.ALL_BASELINES`, `LEARNED_BASELINES` and
  `DEEP_BASELINES`, so it applies to all 24 tasks under `held_out` and to the pre-training and
  transfer protocols, and it joins the validation-selection pool behind the headline tables. It is
  last in every tuple, so no tie-break among the existing baselines changes. The sweep defaults
  (`hpc/submit.sh`, `hpc/sweep.sh`, the loop and the campaign scripts) and the table, figure and
  audit builders enumerate it; the per-baseline SI tables gain a column.
- Existing results are untouched. The committed derived views (`RESULTS.md`, the leaderboard, the
  audit site, `coverage_snapshot.json`, `results/<version>/`, the generated paper tables) have to
  be regenerated once, because they enumerate baselines.

## In the 1.3.0 tag — run-machinery hardening (tooling only; no contract change)

Adapted from akmaier/Agent4CT (see `docs/EXTERNAL_REPOSITORY_REVIEW.md` §4). No task, split,
metric or `result.json` field changes, and no result is re-scored.

- **Installable package (roadmap W3).** `hpc/check_wheel.sh` builds the wheel and installs it into
  a fresh venv. It then checks that the package data is in the wheel (committed splits, contract,
  input digests, schemas, `py.typed`) and that no repo-only files are. Finally, from an empty
  directory, it confirms that the installed package reads a committed split and that
  `evemtbench-evaluate --self-test` passes.

  Both CIs run it in `--clean` mode, which installs the declared dependencies from PyPI and so
  proves that dependency list sufficient. `tests/unit/test_package_data.py` fails when a new data
  file under `src/evemtbench/` is missing from `package-data`. `pyproject.toml` gains trove
  classifiers, and the release checklist gains the PyPI step. Nothing is uploaded yet; publishing is
  the owner's call.

- **Failure records + retry cap.** The runner keeps `failure.json` in a cell's directory while a
  run is in flight. A crash closes it with the stage (`load`, `fit (seed n)`, `predict …`, …), the
  exception type and the traceback tail; SIGTERM (the SLURM time limit) is recorded the same way.
  A run that never closes its attempt (SIGKILL, OOM) reads as "no clean exit". Success deletes
  the file. `coverage --emit-pending`/`--emit-failed` skip cells with ≥ `--max-attempts`
  (default 3) failed attempts, `coverage --summary` groups not-done cells by failure reason, and
  `coverage --failures` lists them (`evemtbench.failures`). Failures are operational state: no
  builder reads them, and they are not part of `coverage_snapshot.json`.
- **Atomic run artifacts.** `result.json`, `run_record.json`, `predictions_seed*.parquet` and
  `failure.json` are written to a temporary sibling and renamed into place. `run_record.json` is
  now written **before** the `result.json` files, so a cell can no longer read as `done` without
  its run record.
- **Working-tree state in `run_record.json`.** New fields `git_dirty` and `git_diff_sha256`
  (SHA-256 of `git diff HEAD` when dirty; `null` when git is unavailable). They are additive, and
  older records stay valid.
- **Byte-identical view rebuilds.** `evemtbench-leaderboard` and `hpc/make_audit.py` leave a page
  untouched when only its `updated <ts>[ · git <sha>]` line would change
  (`evemtbench._io.write_view_if_changed`). A rebuild without a data change no longer rewrites
  every audit page. The freshness gate and the writer share one normaliser.
- `hpc/autonomous_loop.sh` reads the pending count from the line whose first field is `pending`
  (not any line containing the word).
- **Validation-selection reliability diagnostic.** A new section on `audit/robustness.html`
  grades the validation-selected baseline after the fact, per (task, grid) under `held_out`. It
  reports whether the pick is also best on `test` and `benchmark`, its regret (median and max, in
  metric units), and Kendall τ-b between the validation and held-out rankings
  (`evemtbench.evaluation.selection`; adapted from Agent4CT's board rank-correlation check). It
  is a diagnostic only: selection is unchanged.

- **Outside submissions (roadmap W4).** The new `submissions/<model>/` directory holds
  `metadata.json` plus `result.json` files in the runner's schema. `evemtbench-validate-submission`
  (`evemtbench.submission`) checks a submission without the dataset:
  - the metadata, including no declared tuning on or training with held-out data;
  - that every cell exists and applies to its grid;
  - at least 3 seeds and the official metric set;
  - that `aggregate` recomputes from `per_seed`;
  - value ranges;
  - `n_test` against the committed results bundle;
  - `full`-track completeness;
  - the run-record attestation, when one is included.

  CI validates every committed submission. The leaderboard lists valid submissions next to the
  reference baselines with a *self-reported* or *reproduced* badge, and keeps them out of the
  in-house coverage cards and the validation-selected transfer table. Documented in
  `docs/submission.md` §6.

- **Single-class binary breakdown warning (CODE-02).** For a binary task, `evaluate` now emits
  one `SingleClassBinaryBreakdownWarning` listing every breakdown group whose true labels hold a
  single class, where binary `macro_f1` depends on whether the absent class is ever predicted
  (`docs/known_issues.md` §3, which also now says which metrics to read there per task). No
  returned or stored value changes.

## 1.2.0 — fault-location valid samples restricted to line faults (scoped revision)

- **The defect.** The fault-location (FL) target is the distance along the faulted *line*. The
  v1.0/v1.1 valid-sample gate was "fault-active window with a non-null `y_fault_location`". In the
  `adapt_grid` family the producer also fills a location for **bus-target** fault episodes
  (`y_fault_line` = `MainBus*`), a value that is no position on any line, so those windows were
  trained on, used for validation selection and scored on the in-distribution `test` set
  (`docs/known_issues.md` §2 gives the measured share per grid). In the `benchmark` family
  bus-target locations are NaN, and the `multi_grid` corpora have no bus-target window in the
  gate, so neither was affected directly.
- **The fix.** The gate adds the missing condition: `y_fault_line` names a line (`MainLn*`)
  (`tasks/valid_samples.py`). `line_target_fault_location_mask`, formerly the opt-in line-only
  diagnostic, is kept as an alias of the gate.
- **Scope of invalidation (per-task versioning).** The FL tasks' `held_out`, `transfer_zeroshot`
  and `transfer_finetune` results are re-run under the corrected gate into `outputs/1.2.0/` — NOT
  comparable to their 1.0.0 / 1.1.0 counterparts (retained: `outputs/1.0.0/`, `outputs/1.1.0/`
  and the frozen `results/1.0.0/` / `results/1.1.0/` bundles). The FL `multi_grid_pretrain` source
  models are unaffected (same samples) and stay current in `outputs/1.1.0/`. Every other task is
  byte-identical under v1.2.0: its `held_out` results stay in `outputs/1.0.0/`, its
  pretrain/transfer results in `outputs/1.1.0/`, with their original stamps.
  `evemtbench.version.results_version(protocol, task)` encodes the mapping (it gained the `task`
  argument), and `CURRENT_RESULTS_VERSIONS` lists the three trees; every reader of `outputs/`
  resolves a row's tree from its protocol and task, so a superseded FL row in an older tree is
  never reported as current.
- **Projection cache unchanged.** The persistent cache root is keyed by the new
  `evemtbench.version.DATA_CACHE_VERSION` (still `1.1.0`) instead of `BENCHMARK_VERSION`: the
  inputs are unchanged, so the revision must not relocate the cache. The narrower FL views get
  new content-addressed files next to the existing ones.
- `coverage_snapshot.json` gains `results_versions_by_task`, the `<task>/<protocol>` pairs whose
  tree differs from the protocol default in `results_versions`.

**Why:** a valid-sample rule is a frozen artifact, and changing it invalidates comparability for
the tasks it selects samples for (versioning policy). Bus-target windows carried a target the task
does not define, in training, in validation selection and in the in-distribution test score.

## 1.1.0 — corrected (topology-grouped) multi_grid source-corpus splits (scoped revision)

- **The defect.** A `multi_grid` episode is one *line* of one simulation (`sample_id =
  sim_idx*10^4 + line_idx`), so the v1.0 episode-level 70/15/15 split scattered sibling lines of
  one simulation — and every generated topology — across train/val/test. Measured on the v1.0
  splits: ~99 % of val/test simulations also had sibling-line episodes in train, and all ~104
  topologies per corpus appeared in all three partitions. The scored target-grid evaluations were
  never contaminated (reference-grid splits are simulation-level), but source-domain model
  selection (pretrain early-stopping/`best_val` and the zero-shot checkpoint choice) read a leaky
  validation signal.
- **The fix.** `splits/v1.1/multi_grid_pretrain/` regenerates the three source-corpus splits
  **topology-grouped** (`graph_id`; group-aware iterative stratification by `event_type`,
  `splits.generator.split_grouped_episodes`): no topology — and therefore no simulation and no
  sibling line — crosses a partition boundary. A simulation-grouped (`result_id`) variant is
  implemented and audited as the fallback; topology grouping is the shipped split because the
  metadata identifies generated topologies reliably (0 inconsistencies across 431 072 episodes)
  and every event class spans 87–104 of ~104 topologies per corpus, so class support survives
  grouping. Each split dir carries a committed machine-readable `audit.json` (per-partition
  topology/simulation/episode/fault counts, class support, pairwise overlap counts — all zero —
  partition hashes, grouping + seed).
- **Scope of invalidation (per-protocol versioning).** `multi_grid_pretrain`,
  `transfer_zeroshot` and `transfer_finetune` results are re-run under the corrected splits into
  `outputs/1.1.0/` — NOT comparable to their 1.0.0 counterparts (retained: `outputs/1.0.0/`,
  frozen `results/1.0.0/` bundle). Every `held_out` artifact is byte-identical under v1.1.0, so
  the published held_out results remain current, stay in `outputs/1.0.0/` and keep their 1.0.0
  stamp (`evemtbench.version.results_version()` / `splits_version()` encode the mapping).
- **New leakage gates.** `splits.leakage.assert_group_disjoint` + `verify_protocol_regenerates`
  (grouped regeneration determinism), grouped split audits (`splits.audit.audit_multi_grid`),
  committed-artifact drift gates and a behavioral preprocessing-fit leakage test
  (`tests/integration/test_preprocessing_leakage.py`: channel stats / target scaling must be
  train-only).

**Why:** regenerating a committed split invalidates comparability for the protocols scored on it
(versioning policy); grouping the source corpus is the fix the v1.0 release documented as the
known split-granularity deviation.

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
