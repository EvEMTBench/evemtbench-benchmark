# Task card: `event_state_global` (ES) — **IMPLEMENTED**

> **Status:** in the registry and scored. The high-level "middle ground" between the fault triad's
> detection (FD) and classification (FC): a 3-class grid-state classifier over *every* window.
> Realizes the family granularity sketched in [event_classification.md](event_classification.md),
> with a precise activity×family label that makes the `normal` class available on **all four grids**.

| | |
|---|---|
| **Objective** | Classify the high-level state of the grid in each window: `normal` / `switching` / `fault` ("Normal / Event / Fault"). |
| **Task id** | `event_state_global` |
| **Output** | Multiclass (3 classes) |
| **Observability** | **`global`-only** (see below) |
| **Protocol** | `held_out` |
| **Metrics** | `balanced_accuracy`, `macro_f1`, `accuracy`; + per-class F1 ([metrics.md](../metrics.md)) |

## Label (activity × family)

Per window, from the required contract columns `overlap_ratio` and `event_family`
([data.schema](../../src/evemtbench/data/schema.py)); pure deriver in
[`tasks/labels/event_state.py`](../../src/evemtbench/tasks/labels/event_state.py):

- **`fault` (2)** iff `overlap_ratio > 0` and `event_family == "fault"`;
- **`switching` (1)** iff `overlap_ratio > 0` and `event_family == "switching"` — the user-facing
  "Event" class;
- **`normal` (0)** otherwise — every pre-/post-event (non-overlapping) window of *any* episode,
  plus the fully quiescent `no_event` (family `normal`) episodes.

Frozen class order `("normal", "switching", "fault")` (index = id); a change is a **major** bump.
Every window has a defined state, so **all windows are scored** (valid-sample mask is all-True) —
unlike the within-fault-episode masks of FD/FC/FL.

## Why `normal` is available on every grid

`no_event` episodes exist only on the HV grids ([taxonomy.md](../taxonomy.md)). But `normal` here is
**not** tied to `no_event`: it is any non-overlapping window. Because every fault and switching
episode contributes its own quiescent pre-event windows, `normal` has support on **all four grids**
(CigreMV, DoubleLine, TestGrid, IEEE-39) — the grid restriction that caveats ED/EC does not bind ES.

## Why `global`-only

`local`/`line` observability projects onto the faulted line via the `y_fault_line` oracle
([tasks/task.py](../../src/evemtbench/tasks/task.py) `_observe`). Switching and `normal` windows have
no faulted line, so a per-line view is undefined for them without ED's vantage machinery. ES is
therefore a single `global` task (matching the EC card's global-only decision).

## Relationship to FD / FC / EC

- **FD** (binary) only sees windows *inside fault episodes* (fault-active vs. pre-fault). ES is
  broader: it also sees switching and quiescent windows, and separates them.
- **FC** (9-class) subtypes faults *within* fault-active windows. ES is coarser: it stops at the
  family level. FC/FD are unchanged and complementary.
- **EC** (shipped as `event_classification_global`) is the full 23-`event_type` classification. ES is its family-level warm-up,
  now shipped as its own task.
