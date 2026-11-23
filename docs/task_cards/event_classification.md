# Task card: `event_classification_global` (EC)

| | |
|---|---|
| **Objective** | Classify *what kind* of event a window contains, across the full taxonomy. |
| **Task id** | `event_classification_global` |
| **Output** | Multiclass, **23 classes**: 9 `flt_*` + 13 `switch_*` + `no_event` |
| **Observability** | **`global`-only** (the kind of disturbance is a grid-wide property) |
| **Label** | `event_type` of an active window; non-active windows fold into `no_event` ([labels/event_type.py](../../src/evemtbench/tasks/labels/event_type.py)). Frozen union vocabulary, stable indices. |
| **Valid samples** | every window whose active type is in the vocabulary |
| **Protocol** | `held_out` |
| **Metrics** | `balanced_accuracy`, `macro_f1`, `accuracy`, **per-class F1** ([metrics.md](../metrics.md)) |

## Notes

- **Grid-restricted classes** (HIF / cable / motor MV-only, `gen_trip` EHV-only, `ibr_trip` on IBR
  grids; [taxonomy.md](../taxonomy.md)) have zero support where absent
  (NaN per-class F1). Report **per grid**, never pooled.
- Per-class F1 exposes the hard confusions (e.g. inrush vs. incipient fault).
- The 3-class family granularity (normal / switching / fault) is its own task,
  [event_state.md](event_state.md).
- FC (9-class) subtypes faults among fault-active windows; EC covers the whole event space. They
  are complementary.
