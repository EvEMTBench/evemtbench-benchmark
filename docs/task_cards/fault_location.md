# Task card: `fault_location_{local,line,global}` (FL)

| | |
|---|---|
| **Objective** | Regression — distance along the faulted line (%), on fault-active windows of line-fault episodes |
| **Task ids** | `fault_location_local`, `fault_location_line`, `fault_location_global` |
| **Observability** | `local` (6 ch) ⊂ `line` (12 ch) ⊂ `global` (48 ch) — same nested oracle as FD |
| **Output** | Continuous: distance along the faulted line, ~1–99 % |
| **Label** | `y_fault_location` (% along the faulted line). **Episode-constant**; NaN on non-fault episodes. Defined for **line faults** only; in `adapt_grid` bus-target fault episodes also carry a (non-physical) value, which the valid-sample rule excludes since v1.2.0 — see [known_issues.md](../known_issues.md) §2. `tasks/labels/fault_location.py`. |
| **Valid samples** | fault-active windows of a **line** fault (`y_fault_line` = `MainLn*`) with `y_fault_location` not null. The v1.0/v1.1 rule lacked the line condition and included the `adapt_grid` bus-target windows above; results under the two rules are not comparable ([versioning.md](../versioning.md)) |
| **Protocol** | `held_out` (all variants); the `local`/`line` variants also run `multi_grid_pretrain`, `transfer_zeroshot` and `transfer_finetune` (deep baselines only — see [protocol.md](../protocol.md)) |
| **Metrics** | MAE, RMSE, R², median absolute error (all in % distance; MAE/RMSE/median ↓, R² ↑) |
| **Baselines** | majority (train-mean predictor), impedance (conventional impedance-based fault locator; `local` and `line` only, `held_out`), random_forest (regressor), mlp / gru / cnn / resnet / tst (MSE head, target-standardized) |

## Notes / claims

- This is the healthiest task end-to-end in [RESULTS.md](../../RESULTS.md) (sensible observability
  ordering, real R²). The **`line`** view (both terminals) is the most shift-robust — its
  differential signal beats both `local` and `global` on the `benchmark` family.
- FL is the first **REGRESSION** task in the benchmark (added in v0.1); the leaderboard orients
  bars/sorting by metric direction.
- Report both `held_out` test sets (`adapt_grid` `test` and `benchmark`).
- **Conventional reference.** `impedance` is a deterministic impedance-based fault locator: the
  one-terminal reactance method at `local`, the two-terminal positive-sequence method at `line`
  (see [baselines.md](../baselines.md)). It reads the line's series impedance from the grid
  topology graph, selected by `y_fault_line`, the same oracle the two views are built on. It is
  a physical reference row, outside the learned-baseline selection pool, as `threshold` is for
  fault detection. It locates from one full cycle of fundamental-frequency phasors, so the valid
  samples it cannot serve by construction are part of its score: windows whose last cycle is not
  entirely in the fault state, and incipient faults. Read its error with the per-condition
  breakdown (`event_type`), not the aggregate alone.
