# Benchmark versioning

How a release actually ships (freeze → tag → repro pack → DOI): [docs/release.md](release.md).

The benchmark is versioned like an API. `BENCHMARK_VERSION` (in `evemtbench.version`) names the
current contract. Every `EvalResult` and `result.json` is stamped with the results version of its
cell (`results_version(protocol, task, grid=..., source=...)`: `held_out` results keep `1.0.0` and
pretrain/transfer results `1.1.0` wherever their inputs are unchanged; the fault-location tasks
re-run under the corrected valid-sample rule carry `1.2.0`; the faulted-phase cells re-run with
label patch LP-1 carry `1.3.0`), so every reported number is
attributable to a frozen contract. This is distinct from the Python package version in `pyproject.toml`.

## What each bump means

| Bump | Triggered by | Effect on scores |
|---|---|---|
| **major** | test set changes; a **metric** changes; a **derived label** changes; the **window contract** changes; any **split** is regenerated | invalidates score comparability — results on the old and new version are not comparable |
| **scoped revision** (minor, e.g. 1.0.0 → 1.1.0, 1.1.0 → 1.2.0, 1.2.0 → 1.3.0) | a frozen artifact changes for a **subset of the matrix** while every artifact the rest reads stays byte-identical. Three cases so far: a committed split is regenerated for a **subset of protocols**; a **valid-sample rule** is corrected for a **subset of tasks**; a **derived label** is repaired by a committed **label patch** for a **subset of cells** | invalidates comparability **only for the protocols / tasks / cells that read the changed artifact**; the untouched cells' results remain current and keep their original version stamp. The mapping is code, not prose: `evemtbench.version.results_version(protocol, task, grid=..., source=...)` / `splits_version(protocol)`; all output trees stay on disk (`CURRENT_RESULTS_VERSIONS` lists the ones that hold current results) and each superseded `results/<old>/` bundle is frozen for comparison |
| **minor** | a **task** or **baseline** is added without touching any frozen artifact | old scores remain valid |
| **patch** | bug fix that does not change any score | none |

**v1.1.0** is the first scoped revision: the `multi_grid_pretrain` source-corpus splits were
regenerated **topology-grouped** (`splits/v1.1/multi_grid_pretrain/`; the v1.0 episode-level split
scattered sibling lines of one simulation and every generated topology across train/val/test —
see `docs/protocol.md`). `multi_grid_pretrain` / `transfer_zeroshot` / `transfer_finetune` are
re-scored under `outputs/1.1.0/`; every `held_out` artifact is byte-identical, so held_out results
remain current at `outputs/1.0.0/` and are **not** re-run (re-running would only churn published
numbers without changing any input).

**v1.2.0** is the second scoped revision, and the first one scoped by **task**: the fault-location
valid-sample rule now requires a **line** target (`y_fault_line` = `MainLn*`) in addition to an
active fault and a non-null `y_fault_location`. The v1.0/v1.1 rule admitted the bus-target fault
episodes of the `adapt_grid` family, whose location value is no position on any line
([known_issues.md](known_issues.md) §2). The `fault_location_*` tasks' `held_out`,
`transfer_zeroshot` and `transfer_finetune` results are re-run under `outputs/1.2.0/` and are not
comparable to their earlier numbers. Their `multi_grid_pretrain` source models stay current at
`outputs/1.1.0/` (the multi_grid corpora have no bus-target window in the rule, so it selects the
same samples). Every other task keeps its results and stamps: `held_out` at `outputs/1.0.0/`,
pretrain/transfer at `outputs/1.1.0/`. A scoped revision is used instead of a major bump because
no other task's samples, labels, splits or metrics change; within the fault-location tasks the
effect is the same as a major bump.

**v1.3.0** is the third scoped revision, and the first one scoped by **cell**: the faulted-phase
labels of `flt_1phg_hif_w_arc` (high-impedance single-phase-to-ground fault with arc), which the
1.x windows carry with no phase bit so that `fault_phase` dropped the class
([known_issues.md](known_issues.md) §4), are restored by **label patch LP-1**
(`src/evemtbench/data/label_patches/`): a committed, sha256-pinned, episode-keyed overlay built
by `hpc/build_label_patch.py` from the producer's source labels. The windows, manifests, splits
and label rules are byte-identical. The patch is applied by `load_task` to `fault_phase` only, for
a v1.3.0 cell only, and only on the partition whose `manifest_sha256` it was built for. Exactly the
151 cells that read a repaired label move to `outputs/1.3.0/`: `held_out` on cigre_mv,
`multi_grid_pretrain` on template_20kv, and the transfer cells that target cigre_mv (any source) or
start from the template_20kv source (any target). The other 203 `fault_phase` cells keep their
trees, because re-running unaffected cells would only churn published numbers (GPU training is
not bit-identical, [determinism.md](determinism.md)). `results_version` therefore takes the
cell's `grid` and transfer `source` and raises for a `fault_phase` task without `grid`. Older
records re-evaluate exactly: the patch is off for a version before 1.3.0. A label patch is a 1.x
mechanism: it is inert on any other manifest, and the 2.0 producer fixes the label at the source.

## Frozen artifacts (within a major version)

- Official **metrics** (`evaluation/metrics.py`).
- **Derived labels** (`tasks/labels/`: FD / FC / FL) and the line-view oracle (`tasks/line_view.py`).
- The **valid-sample rules** (`tasks/valid_samples.py`), with the one scoped correction above
  (fault location, v1.2.0).
- The committed **label patches** (`data/label_patches/`, pinned by sha256 in its `index.json`;
  LP-1 since v1.3.0). A new or changed patch is a scoped revision for the cells that read it.
- The **window contract** (`WindowContract`: length 0.05 s, step 0.005 s — `W0p050`, seq len 480).
- The committed **split files** (`splits/v1.0/` for `held_out`, `splits/v1.1/` for
  `multi_grid_pretrain` — per-protocol via `evemtbench.version.splits_version()`; older
  `splits/v0.*` trees retained read-only for history).

Tests under `tests/` pin the label derivations and split guarantees; changing one fails CI until
the expectation and the version are bumped together. The committed splits' seeds are pinned in
CI; that they regenerate from the dataset is checked on the cluster
(`tests/cluster/test_committed_splits_regenerate.py`, marker `data_integrity`).

## Why this matters

Concrete example: switching the FD "active" rule from `overlap_ratio>0` to `y_fault_present`, or
regenerating the `held_out` splits, changes labels/partitions — results before and after must be
distinguishable forever. That is a **major** bump, and the `benchmark_version` stamped in every
result makes the distinction permanent and machine-checkable.

**v0.2** is exactly such a bump: the in-distribution family was relabeled `fine_tune` →
`adapt_grid` (matching the producer's on-disk split name), regenerating the committed splits as
`splits/v0.2/`. v0.1 results are retired and not comparable.
