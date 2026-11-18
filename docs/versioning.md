# Benchmark versioning

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

What each scoped revision changed (1.1.0 splits, 1.2.0 fault-location valid samples, 1.3.0 label
patch LP-1): [CHANGELOG](../CHANGELOG.md) and [known_issues.md](known_issues.md). A label patch is a
1.x mechanism; dataset V1 fixes the label at the source.

## Frozen artifacts (within a major version)

- Official **metrics** (`evaluation/metrics.py`).
- **Derived labels** (`tasks/labels/`, one deriver per task function) and the line-view oracle
  (`tasks/line_view.py`).
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
