# Task card: relay selectivity (in-zone vs out-of-zone) — DESIGNED, NOT BUILT

**Status:** paper Open Challenge; interface frozen at `evemtbench.tasks.vantage`
(raises until built). Full design: [docs/design/selectivity.md](../design/selectivity.md).

| | |
|---|---|
| function | selectivity — trip only for in-zone faults from a fixed vantage |
| observability | `local` (the vantage's 6 channels; `line` variant = 12) |
| label | `in_zone` ∈ {0, 1} per window (binary) |
| tier | 3 — hard frontier (the oracle-free deployed question) |
| protocol | `held_out` episodes, unchanged; vantages multiply *evaluation* cells |

## Label semantics (pure deriver, reuse-only)

`in_zone = y_fault_present AND (y_fault_line == vantage.protected_line)`. Out-of-zone faults,
switching events and normal operation are all 0 — the negative class deliberately contains the
*hard* negatives (remote-fault voltage dips) the oracle tasks filter away. Bus faults are
in-zone for both incident-line bays (the `y_fault_line` convention).

## Metrics

Binary set + the selectivity pair:

- **`out_of_zone_trip_rate`** — FP among windows whose fault is outside the zone. *The*
  selectivity number; the false-positive mode motivating the task.
- in-zone `missed_fault_rate` — dependability must not be traded silently for security.

Report stratified by electrical distance vantage→fault (the difficulty axis) — the aux topology
graph (M2) provides the distance.

## Why it matters

Every oracle task overstates deployed performance: a model that trips on *any* grid disturbance
scores perfectly on `fault_detection_local` under the oracle (it only ever sees in-zone faults)
and would be unusable as a relay. Expectation: threshold overcurrent looks dependable but
unselective at low fault impedance; learned models must beat that gap.
