# Task card: event detection (ED)

> **Status (split):** the **global event-presence** detector is **SHIPPED** as `event_detection_global`
> — binary "any active event (fault or switching) vs. quiescent" over every window, global-only
> (switching/normal windows have no faulted line for the local/line oracle). Label re-derived from
> `overlap_ratio` × `event_family` ([labels/event_detection.py](../../src/evemtbench/tasks/labels/event_detection.py)).
> Metrics include the protection-framed `missed_event_rate` (dependability) and `false_event_trip_rate`
> (security). The **per-line relay-selectivity** framing below (`event_detection_{local,line}` —
> trip=1 iff a fault is on line L; benign switching/quiescent = no-trip) remains **PLANNED**: it needs
> the per-line *vantage expansion* mechanism. See also [switch detection](switch_tasks.md) and the
> [task-suite roadmap](../task_suite_roadmap.md).

| | |
|---|---|
| **Objective** | Window-level detection of grid events — distinguishing a disturbance from quiescent operation, framed as a protection relay's trip/no-trip decision. |
| **Task ids** | `event_detection_global` (shipped); `event_detection_{local,line}` planned (per-line framing below) |
| **Output** | Binary |
| **Protocol** | `held_out` (in-distribution primary; transfer caveated — see below) |
| **Metrics** | `balanced_accuracy`, **`missed_event_rate`** (dependability), **`false_event_trip_rate`** (security), `macro_f1`, accuracy; + episode-level **detection latency** ([metrics.md](../metrics.md)) |

## Two framings (by observability)

ED extends FD by giving it **realistic benign negatives** — switching transients and `no_event` —
which FD never sees (FD's negatives are only pre-fault windows *inside* fault episodes). This adds
the **security / false-trip** dimension a real relay is judged on.

- **`local` / `line` — per-line relay selectivity.** Each main line is a relay vantage `L`:
  **trip = 1 iff an active fault is on line `L`**; everything else is **no-trip** — out-of-zone
  faults (other lines, bus faults), switching, `no_event`, and pre-fault windows. Mirrors the
  observability oracle (single-end / differential; [grids.md](../grids.md)). Implementation note:
  this needs a per-line *vantage expansion* of the view (one row per `(window, vantage line)`), the
  one genuinely new mechanism vs. the fault tasks.
- **`global` — centralized detection.** Any active fault anywhere on the grid vs. benign
  (switching / `no_event`). The wide-area situational-awareness setting.

## Labels & valid samples

Binary, derived from `y_fault_present` / `y_event_present` / `y_fault_line` (pure deriver, to live
in `tasks/labels/`). Valid samples include **fault, switching, and `no_event` windows** (the whole
point — benign events must be seen as negatives), unlike FD's fault-episodes-only mask.

## Grid availability & transfer caveat

`no_event` exists **only in the HV grids' `adapt_grid`** (DoubleLine, TestGrid) and in **no**
`benchmark` family ([taxonomy.md](../taxonomy.md)). Consequences:
- An "event-vs-quiescent" detector has **no `benchmark` negative** → transfer is reported
  **in-distribution-primary**, benchmark only as a coverage-flagged partial.
- On **MV/EHV** grids there is no quiescent class at all; there, ED is meaningful only in the
  **relay-selectivity** framing (negatives = out-of-zone faults + switching, available everywhere),
  not as "event vs `no_event`".

## Relationship to FD

FD (frozen) = within-fault-episode onset detection (negatives = pre-fault windows). ED is a
**separate, additive** task that does not change FD; it answers the protection question FD cannot:
*does the relay correctly hold for benign switching/quiescent operation?*
