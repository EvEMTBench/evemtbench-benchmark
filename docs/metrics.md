# EvEMTBench — metrics reference

Defines the official metrics, their orientation, how the leaderboard ranks, and the
protection-grade metrics. Frozen within a benchmark major version (see
[versioning.md](versioning.md)). Source: [`evaluation/metrics.py`](../src/evemtbench/evaluation/metrics.py),
[`leaderboard.py`](../src/evemtbench/leaderboard.py), [`analysis.py`](../src/evemtbench/analysis.py).

## Metric sets by label kind

| label kind | metrics |
|---|---|
| **binary** (FD) | `balanced_accuracy`, `macro_f1`, `missed_fault_rate`, `false_alarm_rate`, `accuracy` |
| **multiclass** (FC) | `balanced_accuracy`, `macro_f1`, `accuracy` (+ per-class F1 when the class vocab is supplied) |
| **regression** (FL) | `mae`, `rmse`, `r2`, `median_abs_error` |

- `missed_fault_rate` = FN/(positives) — **dependability** (missed faults); `false_alarm_rate` =
  FP/(negatives) — **security** (false trips). On a single-class subset, undefined metrics are
  `NaN` (recorded as `n_defined < n_seeds`, never silently 0).
  Binary `macro_f1` is the exception: it stays defined but unstable there (see
  [known_issues.md](known_issues.md#3-binary-macro_f1-on-single-class-breakdown-subsets)).
- `balanced_accuracy` = mean per-class recall → robust to the heavy class imbalance (negatives ≫
  positives for FD; SHC ≫ incipient/HIF for FC), which is why it's the headline classification metric.

## Direction (`METRIC_DIRECTION`)

| higher-is-better (+1) | lower-is-better (−1) |
|---|---|
| `balanced_accuracy`, `macro_f1`, `accuracy`, `r2` | `missed_fault_rate`, `false_alarm_rate`, `mae`, `rmse`, `median_abs_error` |

## Aggregation across seeds

Each cell runs ≥1 seed; per metric the result stores `{mean, std, ci95, n_defined, n_seeds}` —
mean/std over the **defined** seeds only (`n_defined`), so a metric undefined on some seeds is
distinguished from a genuine 0. `ci95` is the 95% CI half-width (Student t, ddof=1; `null` below
2 seeds — never a fake ± 0). Rendered tables report **mean ± 95% CI**; for baseline-vs-baseline
claims use the paired-by-seed test (`evemtbench-significance`).

## Leaderboard ranking

Per task section the **headline metric** is the first present of
`HEADLINE_PRIORITY = (balanced_accuracy, mae)` (i.e. balanced-accuracy for FD/FC, MAE for FL).
Metrics in `_ABSOLUTE_01 = {balanced_accuracy, accuracy, macro_f1}` render as absolute [0,1] bars
(higher = fuller); other headline metrics (mae, r2) get a section-relative bar oriented by
`METRIC_DIRECTION`. The board always exposes **per grid × per test_set** rows
(filterable by topology / test-set / model) — a single scalar is never the whole story.

## Protection-grade metrics (`analysis.py`)

Beyond static window accuracy, protection cares about **dependability, security, and speed**, and
how difficulty varies with the swept parameters. Computed post-hoc from saved predictions joined to
the windows `y` (never loading `X`):

- **Detection latency** (binary tasks): per fault episode, time from inception (`dt_start` = 0) to
  the first correctly-flagged active window → `detection_rate`, `median_latency_s`, `mean_latency_s`.
- **Location difficulty curve** (FL): MAE binned by distance-along-line %, exposing where
  localization degrades (the `benchmark` family sweeps location at discrete points).
- **Resistance-stratified curves**: planned — fault resistance is not yet in the windows `y` (lives
  in the simulation `settings.csv`); a dataset-side dependency.

Run: `python -m evemtbench.analysis --task <id> --grid <grid> --baseline <b> --test-set <test|benchmark>`.

## Reading transfer

Every metric is conditional on **(task × observability × grid × protocol × test_set)**. The
`benchmark` (transfer) column is the hard test: under the `adapt_grid`→`benchmark` operating-point
shift, magnitude-reliant `local`/`global` views can collapse while the differential `line` view
transfers — and this is grid-dependent. Always compare in-distribution (`test`) against
`benchmark`, per grid.
