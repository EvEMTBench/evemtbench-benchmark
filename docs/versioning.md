# Benchmark versioning

How a release actually ships (freeze → tag → repro pack → DOI): [docs/release.md](release.md).

The benchmark is versioned like an API. `BENCHMARK_VERSION` (in `evemtbench.version`) names the
current contract. Every `EvalResult` and `result.json` is stamped with its protocol's results
version (`results_version(protocol)`: `held_out` results keep `1.0.0`, whose inputs are unchanged
under 1.1.0), so every reported number is attributable to a frozen contract. This is distinct from the Python package version in `pyproject.toml`.

## What each bump means

| Bump | Triggered by | Effect on scores |
|---|---|---|
| **major** | test set changes; a **metric** changes; a **derived label** changes; the **window contract** changes; any **split** is regenerated | invalidates score comparability — results on the old and new version are not comparable |
| **scoped revision** (minor, e.g. 1.0.0 → 1.1.0) | a committed split is regenerated for a **subset of protocols** while every artifact the other protocols read stays byte-identical | invalidates comparability **only for the protocols scored on the regenerated split**; the untouched protocols' results remain current and keep their original version stamp. The per-protocol mapping is code, not prose: `evemtbench.version.results_version()` / `splits_version()`; both output trees stay on disk and the superseded `results/<old>/` bundle is frozen for comparison |
| **minor** | a **task** or **baseline** is added without touching any frozen artifact | old scores remain valid |
| **patch** | bug fix that does not change any score | none |

**v1.1.0** is the first scoped revision: the `multi_grid_pretrain` source-corpus splits were
regenerated **topology-grouped** (`splits/v1.1/multi_grid_pretrain/`; the v1.0 episode-level split
scattered sibling lines of one simulation and every generated topology across train/val/test —
see `docs/protocol.md`). `multi_grid_pretrain` / `transfer_zeroshot` / `transfer_finetune` are
re-scored under `outputs/1.1.0/`; every `held_out` artifact is byte-identical, so held_out results
remain current at `outputs/1.0.0/` and are **not** re-run (re-running would only churn published
numbers without changing any input).

## Frozen artifacts (within a major version)

- Official **metrics** (`evaluation/metrics.py`).
- **Derived labels** (`tasks/labels/`: FD / FC / FL) and the line-view oracle (`tasks/line_view.py`).
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
