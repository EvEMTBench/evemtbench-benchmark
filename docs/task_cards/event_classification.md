# Task card: event classification (EC)

> **Status:** the **full event-type** classifier is **SHIPPED** as `event_classification_global` —
> 23-class (9 `flt_*` + 13 `switch_*` + `no_event`), global-only, over every window (non-active
> windows fold into `no_event`), label from [labels/event_type.py](../../src/evemtbench/tasks/labels/event_type.py).
> The **family granularity** (3-class normal/switching/fault) shipped separately as
> `event_state_global` ([event_state.md](event_state.md)) — ES families are the coarse grouping of
> EC's 23 types. Grid-restricted types are zero-support (NaN per-class F1); report per grid. The
> `type` framing below is what is implemented; a `local`/`line` ablation stays out of scope
> (global-only). See the [task-suite roadmap](../task_suite_roadmap.md).

| | |
|---|---|
| **Objective** | Classify *what kind* of event a window contains — beyond faults, across the full taxonomy. |
| **Output** | Multiclass |
| **Observability** | **`global`** (the kind of disturbance is a grid-wide property, not zone-local); a `local`/`line` ablation is optional. |
| **Protocol** | `held_out` (in-distribution primary; transfer caveated) |
| **Metrics** | `balanced_accuracy`, `macro_f1`, accuracy, **per-class F1** ([metrics.md](../metrics.md)) |

## Two granularities (family-first, then full)

- **`event_classification_family` — 3-class:** `fault` / `switching` / `normal` (from
  `event_family`). The interpretable warm-up; pairs with ED ("what benign disturbance is this?").
  **Shipped** as its own task `event_state_global` (see
  [event_state.md](event_state.md)) — a true state classifier over every window, where `normal`
  is any non-overlapping window (so it is available on all four grids, not just the HV `no_event`
  grids).
- **`event_classification_type` — full taxonomy:** the **23 `event_type`s** (9 fault + 13 switching
  + `no_event`). Use a **global union vocabulary** with stable indices (the FC/HIF pattern):
  grid-restricted types (HIF, cable, motor → MV; gen_trip → EHV; etc.) simply have **zero support**
  on grids that lack them, keeping the label space comparable across grids. Per-class F1 exposes the
  hard confusions (e.g. inrush vs. incipient fault).

## Labels & valid samples

Multiclass from `event_family` (family) or `event_type` (full); pure derivers in `tasks/labels/`
with a **frozen class vocabulary** + `evaluate._CLASS_VOCAB` entry (as FC has). Valid samples =
windows with a defined label for that granularity. Decide per design whether `no_event` is a class
(family: yes, as `normal`) or excluded (full taxonomy may treat it as the ED negative instead).

## Grid availability & transfer caveats (from [taxonomy.md](../taxonomy.md))

- **Grid-restricted classes:** HIF / cable / motor (MV-only), `gen_trip` (EHV-only), `ibr_trip`
  (IBR grids), `no_event` (HV-only). The union vocab handles this via zero-support classes; report
  **per grid**, never pooled.
- **`benchmark` transfer:** the `benchmark` families carry **no `no_event`** and (on some grids)
  sparse switching, so the `normal` class and some switching types are absent from transfer →
  in-distribution-primary, benchmark coverage-flagged.

## Relationship to FC

FC (9-class) classifies *fault subtype* among fault-active windows. EC is the broader task over the
**whole event space** (faults + switching + normal). They are complementary, additive tasks; FC is
unchanged.
