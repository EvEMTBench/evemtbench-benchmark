# Datasheet — EvEMTBench

A short datasheet (after Gebru et al.) for the dataset this benchmark evaluates on. The dataset
is a separate artifact (see [LICENSES/DATA-LICENSE.md](../LICENSES/DATA-LICENSE.md)); this
documents how the benchmark uses it and its known limitations.

> **Scope.** The benchmark evaluates **all four reference grids** (CigreMV 20 kV MV, DoubleLine
> 110 kV HV, TestGrid110kV 110 kV HV, IEEE-39 345 kV EHV) on **24 task ids** — the fault triad
> FD / FC / FL and the fault attributes (grounded / phase / category) × {local, line, global}
> observability, plus six `global`-only event-scope tasks (event detection / classification /
> state, switch detection / type, fault origin) — under the `held_out` protocol. The three
> `multi_grid` pre-training templates (20 / 110 / 345 kV) feed the `multi_grid_pretrain`,
> `transfer_zeroshot` and `transfer_finetune` protocols at the portable `local` / `line` views (see
> [protocol.md](protocol.md), [grids.md](grids.md), [taxonomy.md](taxonomy.md)).

## Motivation
ML for power-system protection lacks shared datasets and reproducible evaluation. EvEMTBench
provides comprehensively labeled, grid-wide point-on-wave (PoW) voltage/current waveforms from
detailed EMT simulations across multiple grids and voltage levels, so methods can be compared
under fixed tasks, splits, and metrics.

## Composition
- **Signals:** three-phase V and I at each measurement cubicle, 9600 Hz; windows (window contract 1.0.0) are 0.05 s
  (480 samples) at 0.005 s stride. Per-cubicle channel order `[Iₐ,I_b,I_c,Vₐ,V_b,V_c]`.
  **Sign convention:** all current measurements are oriented **towards the branch element**
  (producer convention — a model that flips this sign is not evaluating the same quantity).
  Each exported simulation is 4801 samples long (0.5 s × 9600 Hz + 1; both endpoints included).
- **Unit of an episode.** For the reference grids one episode = one simulation (`sample_id` is the
  producer's `sim_idx`). For the `multi_grid` templates the windows are stored in the portable
  two-terminal view, so one episode = **one line of one simulation**
  (`sample_id = sim_idx × 10⁴ + line_idx`): ≈10 000 simulations per voltage level become
  119 153 / 138 512 / 173 407 episodes at 20 / 110 / 345 kV. Compare corpus sizes in the matching
  unit — the dataset paper counts simulations.
- **Grids:** CIGRE MV (20 kV/50 Hz), Double Line (110 kV/50 Hz), Protection Reference
  (110 kV/50 Hz), IEEE 39-Bus (345 kV/**60 Hz**, the only 60 Hz grid).
- **Event types (~23) in 3 active families** (`event_type` / `event_family` columns):
  - `fault` (9): `flt_1phg_shc`, `flt_2ph_shc`, `flt_2phg_shc`, `flt_3ph_shc`, `flt_1phg_hif`,
    `flt_1phg_incipient`, and `_w_arc` variants of 1PHG SHC / HIF / incipient.
  - `switching` (13): `switch_{load,cap,ohl,cable}_{on,off}`, `switch_inrush_{lv,hv}`,
    `switch_motor_start`, `switch_gen_trip`, `switch_ibr_trip`.
  - `normal` (1): `no_event` (steady state).
  Grid-restricted types: cable/motor/HIF are MV-only; `switch_gen_trip` is IEEE-39 only;
  `no_event` appears only in Double Line + Test Grid adapt_grid.
- **Dataset families** (orthogonal to event families): `benchmark` (held-out, discrete parameter
  sweep, lean schema, fault-heavy), `adapt_grid` (single grid, continuous random events + operating
  states, balanced), `train` (many synthetic topologies).
- **Identity:** one simulation = one `sim_idx` → one `result{sim_idx}.csv`; windows carry
  `sample_id` (= the episode). Settings schemas vary by grid (19–130 columns).

## Collection / provenance
Generated with DIgSILENT PowerFactory EMT simulations (10 µs step, 1.5 s; first 1.0 s discarded so
initialisation transients settle; last 0.5 s exported). 22 event types are simulated (13 operating
events + 9 fault types), each event type available at the voltage levels its grid elements support.
Simulations whose load flow or EMT run failed are dropped in post-processing, so the surviving label
file is `settings_clean.csv` (one row per `sim_idx`) — the per-episode counts below are therefore
slightly under the nominal 10 000 per dataset. Authoring paths in `general/result_file_path` are
Windows-style provenance metadata only — never resolved by the loader (files are located by
`sim_idx`).

## Preprocessing / labeling
Window labels are derived deterministically by the windowing pipeline from `settings.csv` event
fields; the benchmark's label derivers (`tasks/labels/`) then read the resulting per-window `y`
columns.
Fault resistance is selected by `event_type` (HIF vs SHC), not by null-check (both columns are
populated on fault rows). Splits are at the episode level so no window leaks across partitions.

## Auxiliary artifacts (topology graphs + per-episode settings)

Each per-grid archive additionally carries, beside the waveforms: a **networkx topology graph**
(`graphs/*.pickle` — nodes = buses with electrical attributes, edges = lines/transformers with
parameters) and the **per-episode `settings.csv`** (one row per `sim_idx`: event type, location,
phases, fault resistance `event_flt_shc_resistance`, switching parameters). The benchmark
extracts these into a separate additive overlay (`$EVEMTBENCH_AUX_ROOT`; the producer data root
is never written) via `python -m evemtbench.data.extract_aux`; they join windows by
`sample_id == general/sim_idx`. Consumers: `data/graphs.py` (graph/node/edge features — the
cross-topology direction), `data/aux_labels.py` (settings join — e.g. the `resistance_bin`
evaluation stratum). Layout details: [DATA_LAYOUT.md](DATA_LAYOUT.md).

## Known limitations (read before claiming results)
- **Synthetic only** — no guarantee of real-world deployment performance; a sim-to-real gap
  remains (noise, unknown controls, measurement error).
- **Balanced ≠ empirical** — `adapt_grid`/`train` class balance is for controlled comparison,
  not real event frequencies.
- **no_event asymmetry** — the steady-state `no_event` class exists only in some partitions, so
  the negative-class composition differs by grid. Always read the per-family breakdown. (Window-
  level family is derived from `y_event_present`/`y_fault_present`, not the episode-level
  `event_family` column.)
- **Operating-point / scale shift (adapt_grid → benchmark)** — the `benchmark` family sits at a
  different operating point (≈6–13× steady-state current on some grids) from `adapt_grid`. With
  per-channel standardization fit on train, magnitude-reliant view×model combos transfer poorly
  while the **differential `line` view** (which cancels the shared load term) transfers — a genuine
  domain-shift test, not a calibration artifact. Transfer behaviour is **grid-dependent**
  (e.g. CigreMV transfers well, TestGrid collapses); always read the per-grid `benchmark` column.
- **Grid-confounded grounding & events** — grounding types and HIF/cable/motor events are
  grid-restricted (IEEE-39 is solid-ground, no MV events), so grounding/event-stratified robustness
  is confounded with grid. The 9-class FC vocabulary includes HIF, which has **zero support** on
  HV/EHV grids.
- **Per-grid target counts differ** — metrics are not comparable when pooled across grids; the
  leaderboard reports **per grid** (and per test_set).
- **`no_event` asymmetry** — the steady-state `no_event` class exists only in the HV grids'
  `adapt_grid` (not in any `benchmark` family, nor on MV/EHV), so an event-vs-quiescent task has no
  benchmark negative. See [taxonomy.md](taxonomy.md).
- **Global-view memory** — the `global` view scales with topology (48→420 ch); the widest grids
  (IEEE-39) need careful host-memory handling (chunked projection, per-batch standardization).
- **Observability matters** — `line` / `global` models see more than a `local` single-end relay;
  never compare across observability settings or grids without stating which (fairness rule).

## Intended use
Controlled, reproducible, diagnostic comparison of ML methods for protection tasks under EMT
simulation. **Not** deployment certification, relay validation, or empirical-frequency modeling.
