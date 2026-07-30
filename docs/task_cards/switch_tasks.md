# Task card: switching-event tasks (detection / type) — **SHIPPED**

The benign-disturbance counterparts of the fault tasks. Both are **global-only** (switching windows
have no faulted line for the local/line oracle).

| Task | id | source | kind | classes |
|---|---|---|---|---|
| Switch detection | `switch_detection_global` | `overlap_ratio` × `event_family` | binary | switching-active=1 / rest=0 |
| Switch-type | `switch_type_global` | `y_switch_category` | multiclass | 8: load, capacitor, ohl, inrush, ibr, cable, motor_start, generator |

Derivers: [switch_detection.py](../../src/evemtbench/tasks/labels/switch_detection.py),
[switch_type.py](../../src/evemtbench/tasks/labels/switch_type.py). Frozen vocabulary (index = id);
a change is a **major** version bump.

## Notes

- **Switch detection** positive = an active window whose `event_family == "switching"`; faults
  (active or pre-fault) and quiescent windows are negatives. Together with fault detection it
  partitions the positives of event detection: ED-positives = FD-positives ∪ SD-positives. Metrics
  include `missed_event_rate` / `false_event_trip_rate`.
- **Switch-type** classifies *what kind* of switching, on switching-active windows. Categories are
  grid-restricted (`cable`/`motor_start` MV-only, `generator` EHV-only, `ibr` on MV+TestGrid);
  universal categories (`load`, `capacitor`, `ohl`, `inrush`) come first so indices stay stable, and
  absent categories are zero-support (NaN per-class F1). Report per grid, never pooled.
- `y_switch_category` is a contract-*optional* column with **no contract-frozen enum** — the
  vocabulary is frozen in the deriver; it fails loud if the column is absent.
- Companion to [event classification](event_classification.md) (EC subtypes the whole event space;
  switch-type subtypes only the switching family).
