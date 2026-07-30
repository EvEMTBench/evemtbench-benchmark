# Benchmark versioning

How a release actually ships (freeze → tag → repro pack → DOI): [docs/release.md](release.md).

The benchmark is versioned like an API. `BENCHMARK_VERSION` (in `evemtbench.version`) is
stamped into every `EvalResult` and `result.json`, so every reported number is attributable to
a frozen contract. This is distinct from the Python package version in `pyproject.toml`.

## What each bump means

| Bump | Triggered by | Effect on scores |
|---|---|---|
| **major** | test set changes; a **metric** changes; a **derived label** changes; the **window contract** changes; any **split** is regenerated | invalidates score comparability — results on the old and new version are not comparable |
| **minor** | a **task** or **baseline** is added without touching any frozen artifact | old scores remain valid |
| **patch** | bug fix that does not change any score | none |

## Frozen artifacts (within a major version)

- Official **metrics** (`evaluation/metrics.py`).
- **Derived labels** (`tasks/labels/`: FD / FC / FL) and the line-view oracle (`tasks/line_view.py`).
- The **window contract** (`WindowContract`: length 0.05 s, step 0.005 s — `W0p050`, seq len 480).
- The committed **split files** (`splits/v0.2/`; `splits/v0.1/` retained read-only for history).

Tests under `tests/` pin the label derivations and split guarantees; changing one fails CI until
the expectation and the version are bumped together. Leakage tests also verify the committed
splits regenerate from the dataset (`leakage.verify_held_out_regenerates`).

## Why this matters

Concrete example: switching the FD "active" rule from `overlap_ratio>0` to `y_fault_present`, or
regenerating the `held_out` splits, changes labels/partitions — results before and after must be
distinguishable forever. That is a **major** bump, and the `benchmark_version` stamped in every
result makes the distinction permanent and machine-checkable.

**v0.2** is exactly such a bump: the in-distribution family was relabeled `fine_tune` →
`adapt_grid` (matching the producer's on-disk split name), regenerating the committed splits as
`splits/v0.2/`. v0.1 results are retired and not comparable.
