# Task card: `event_detection_global` (ED)

| | |
|---|---|
| **Objective** | Window-level detection of any grid event (fault or switching) vs. quiescent operation. |
| **Task id** | `event_detection_global` |
| **Output** | Binary: event active in window (1) vs. quiescent (0) |
| **Observability** | **`global`-only** (switching and quiescent windows have no faulted line for the `local`/`line` oracle) |
| **Label** | `1` iff `overlap_ratio > 0` and `event_family` is `fault` or `switching`; else `0`. Re-derived from `overlap_ratio` × `event_family` ([labels/event_detection.py](../../src/evemtbench/tasks/labels/event_detection.py)). |
| **Valid samples** | every window |
| **Protocol** | `held_out` |
| **Metrics** | `balanced_accuracy`, `macro_f1`, **`missed_event_rate`** (dependability), **`false_event_trip_rate`** (security), `accuracy` ([metrics.md](../metrics.md)) |

## Notes

- ED extends FD with **benign negatives and positives**: FD only sees windows inside fault
  episodes; ED sees every window, so switching transients count as events.
- Negatives are the inactive pre-/post-event windows of the fault and switching episodes. The
  reference grids contain no dedicated `no_event` episodes ([taxonomy.md](../taxonomy.md)). The
  pre/post mix differs by grid and by test set. Report per grid.
- ED = FD-positives ∪ switch-detection positives ([switch_tasks.md](switch_tasks.md)).
- A per-line relay framing (trip only for a fault on the protected line) is the
  [relay-selectivity](relay_selectivity.md) task, designed but not built.
