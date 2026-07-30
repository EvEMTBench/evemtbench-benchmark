# Design: relay selectivity (out-of-zone discrimination)

**Status: designed, not built** — a paper Open Challenge. Code seam: `evemtbench.tasks.vantage`.

## Why

Every current task either hands the model the faulted line (the observability **oracle** keyed on
`y_fault_line`) or all channels (`global`). A deployed relay has neither: it sits at **one fixed
vantage** (cubicle) and sees *every* event in the grid from there. The operational failure mode
this hides is **false tripping on out-of-zone faults** — a healthy-line relay seeing the voltage
dip of a remote fault must not trip. Protection engineering calls this *selectivity*; ML models
trained only on the oracle views have never been asked the question.

## Task shape

- **Input:** a fixed vantage `(grid, cubicle)` — its 6 local channels (optionally the line pair,
  12) — for *all* episodes, not just those whose fault touches the vantage.
- **Label:** `in_zone` ∈ {0, 1} per window: fault present **and** on the vantage's protected
  line ⇒ 1; fault elsewhere / benign event / normal ⇒ 0. Derivable today from
  `y_fault_present` × (`y_fault_line` == vantage's line) — reuse-only, no new data.
- **Metrics:** the binary set + two selectivity-specific rates: `out_of_zone_trip_rate`
  (FP among out-of-zone-fault windows — *the* selectivity number) and in-zone
  `missed_fault_rate`. Report per fault-distance bin (electrical distance vantage→fault) —
  the difficulty axis.
- **Splits/protocol:** unchanged `held_out` episodes; the vantage enumeration multiplies
  evaluation (not training) cells.

## Build plan (the seam to fill)

1. `tasks/vantage.py::iter_vantages(grid)` — enumerate cubicles + their protected line from the
   oracle's channel map (`tasks/line_view.py` already owns this mapping).
2. `vantage_view(task, cubicle)` — a `TaskView` of the vantage's channels over *all* valid
   windows, labels rewritten to `in_zone` (a pure deriver + golden tests, like every label).
3. Register `relay_selectivity_local` (tier 3) — the registry/coverage/leaderboard machinery
   picks it up with no further changes (all registry-driven since M1).
4. Baseline expectation: threshold overcurrent should look *good* on in-zone recall and *bad* on
   out-of-zone trips at low fault impedance — the gap is the benchmark's headroom.

## Open questions

- Vantage sampling: all cubicles (large eval matrix) vs. a frozen representative subset per grid
  (proposal: all line cubicles, both ends — matches the oracle's channel universe).
- Zone definition for bus faults: incident-line bays ⇒ in-zone for both adjacent vantages
  (matches how `y_fault_line` maps bus faults today).
- Whether transfer (`benchmark` family) selectivity is scored at v-next or deferred.

## Acceptance criteria

- [ ] `evemtbench.tasks.vantage` yields a fixed-relay view whose channel set provably excludes
      the fault-line oracle (leakage-style test).
- [ ] In-zone/out-of-zone labels derive from committed topology data with a frozen derivation
      (label test in `tests/`).
- [ ] The selectivity task registers with tier + metrics (incl. a false-trip rate) and appears
      in the coverage matrix; majority floor verified by `--check-floor`.
- [ ] `BENCHMARK_VERSION` minor bump + CHANGELOG + task card.
