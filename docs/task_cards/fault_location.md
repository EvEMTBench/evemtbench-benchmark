# Task card: `fault_location_{local,line,global}` (FL)

| | |
|---|---|
| **Objective** | Regression — distance along the faulted line (%), on fault-active windows of line-fault episodes |
| **Task ids** | `fault_location_local`, `fault_location_line`, `fault_location_global` |
| **Observability** | `local` (6 ch) ⊂ `line` (12 ch) ⊂ `global` (48 ch) — same nested oracle as FD |
| **Output** | Continuous: distance along the faulted line, ~1–99 % |
| **Label** | `y_fault_location` (% along the faulted line). **Episode-constant** and present only on **line faults** (NaN on bus faults and non-fault episodes). `tasks/labels/fault_location.py`. |
| **Valid samples** | fault-active windows of **line-fault** episodes only (`y_fault_location` not null) |
| **Protocol** | `held_out` |
| **Metrics** | MAE, RMSE, R², median absolute error (all in % distance; MAE/RMSE/median ↓, R² ↑) |
| **Baselines** | majority (train-mean predictor), random_forest (regressor), mlp + gru (MSE head, target-standardized) |

## Notes / claims

- This is the healthiest task end-to-end in [RESULTS.md](../../RESULTS.md) (sensible observability
  ordering, real R²). The **`line`** view (both terminals) is the most shift-robust — its
  differential signal beats both `local` and `global` on the `benchmark` family.
- FL is the first **REGRESSION** task in the benchmark (added in v0.1); the leaderboard orients
  bars/sorting by metric direction.
- Report both `held_out` test sets (`adapt_grid` `test` and `benchmark`).
