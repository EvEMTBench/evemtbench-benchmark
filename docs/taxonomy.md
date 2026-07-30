# EvEMTBench — full fault & event taxonomy (all topologies & voltage levels)

Derived from **every** windowed label file on disk — all four reference grids × both families
(`adapt_grid` + `benchmark`), 8 `y_*.parquet` files. Window contract: 50 ms / 5 ms step (l=480),
9600 Hz. Episode counts below are **distinct simulations (`sample_id`) summed over both families**.

## Reference grids & levels

| grid key | topology id | level | voltage | frequency |
|---|---|---|---|---|
| `cigre_mv` | `mv_cigre_distribution_20kv` | **MV** | 20 kV | 50 Hz |
| `double_line` | `hv_double_line_110kv` | **HV** | 110 kV | 50 Hz |
| `testgrid_110kv` | `hv_testgrid_110kv` | **HV** | 110 kV | 50 Hz |
| `ieee39` | `ehv_ieee39_transmission_345kv` | **EHV** | 345 kV | 60 Hz (only 60 Hz grid) |

**23 distinct `event_type`s = 9 faults + 14 operating events**, in 3 families: `fault`,
`switching`, `normal`.

## Fault events (9) — `event_family = fault`, `event_type` prefix `flt_`

| event_type | category | grounded | faulted phases | levels present | episodes |
|---|---|---|---|---|---|
| `flt_1phg_shc` | shc | yes | A · B · C (single, per episode) | MV/HV/EHV | 5622 |
| `flt_1phg_shc_w_arc` | shc | yes | A · B · C | MV/HV/EHV | 5670 |
| `flt_2ph_shc` | shc | **no** | AB · AC · BC | MV/HV/EHV | 5547 |
| `flt_2phg_shc` | shc | yes | AB · AC · BC | MV/HV/EHV | 5539 |
| `flt_3ph_shc` | shc | **no** | ABC | MV/HV/EHV | 5597 |
| `flt_1phg_incipient` | incipient | yes | A · B · C | MV/HV/EHV | 2589 |
| `flt_1phg_incipient_w_arc` | incipient | yes | A · B · C | MV/HV/EHV | 2640 |
| `flt_1phg_hif` | **hif** | yes | A · B · C | **MV only** | 555 |
| `flt_1phg_hif_w_arc` | **hif** | yes | A · B · C | **MV only** | 560 |

Categories: **shc** = solid/low-impedance short circuit; **hif** = high-impedance ground fault;
**incipient** = self-extinguishing high-impedance fault. `_w_arc` = Kizilcay arc model on the
faulted phase. "1phg" is single-line-to-ground on **one** phase (which phase varies per episode
and is recorded in `y_phase_{A,B,C}`).

## Operating / non-fault events (14)

| event_type | family | switch category | action | levels present | episodes |
|---|---|---|---|---|---|
| `switch_load_on` / `switch_load_off` | switching | load | on/off | MV/HV/EHV | 2353 / 2315 |
| `switch_cap_on` / `switch_cap_off` | switching | capacitor | on/off | MV/HV/EHV | 2312 / 2319 |
| `switch_ohl_on` / `switch_ohl_off` | switching | ohl | on/off | MV/HV/EHV | 2365 / 2334 |
| `switch_inrush_lv` | switching | inrush | on (LV trafo) | MV/HV/EHV | 2305 |
| `switch_inrush_hv` | switching | inrush | on (HV trafo) | MV/HV (not EHV) | 1709 |
| `switch_ibr_trip` | switching | ibr | trip | MV + TestGrid HV | 1147 |
| `switch_cable_on` / `switch_cable_off` | switching | cable | on/off | **MV only** | 463 / 434 |
| `switch_motor_start` | switching | motor_start | start | **MV only** | 472 |
| `switch_gen_trip` | switching | generator | trip | **EHV only** | 675 |
| `no_event` | normal | — | — | **HV only** (DoubleLine, TestGrid) | 1841 |

## Grid × event availability matrix

`✓` = present in that grid (either family).

| event_type | CigreMV (MV) | DoubleLine (HV) | TestGrid110kV (HV) | IEEE-39 (EHV) |
|---|:---:|:---:|:---:|:---:|
| `flt_1phg_shc`(+`_w_arc`) | ✓ | ✓ | ✓ | ✓ |
| `flt_2ph_shc` / `flt_2phg_shc` / `flt_3ph_shc` | ✓ | ✓ | ✓ | ✓ |
| `flt_1phg_incipient`(+`_w_arc`) | ✓ | ✓ | ✓ | ✓ |
| `flt_1phg_hif`(+`_w_arc`) | ✓ | · | · | · |
| `no_event` | · | ✓ | ✓ | · |
| `switch_load_*` / `switch_cap_*` / `switch_ohl_*` / `switch_inrush_lv` | ✓ | ✓ | ✓ | ✓ |
| `switch_inrush_hv` | ✓ | ✓ | ✓ | · |
| `switch_ibr_trip` | ✓ | · | ✓ | · |
| `switch_cable_*` / `switch_motor_start` | ✓ | · | · | · |
| `switch_gen_trip` | · | · | · | ✓ |

## Why events are grid-restricted (physically motivated)

- **HIF, cable switching, motor start → MV only.** High-impedance ground faults are a MV
  phenomenon; underground cables and induction-motor starts are modeled only in the MV grid.
- **Generator trip → EHV only.** Conventional synchronous generation tripping is simulated on
  the 345 kV transmission grid (IEEE-39).
- **IBR trip → grids with inverter-based resources** (CigreMV + TestGrid110kV); DoubleLine and
  IEEE-39 (as configured) have none.
- **HV-trafo inrush → not on EHV** (the modeled EHV grid has no higher level to energize from).
- **`no_event` (quiescent background) → only the HV grids** (DoubleLine, TestGrid). CigreMV and
  IEEE-39 carry **no `no_event` episodes** — relevant for event-detection negatives and for any
  same-grid transfer of an "event vs. quiescent" task on those grids.

## Label fields in `y` (per window)

`event_type`, `event_family`, `status`, `overlap_ratio`; targets `y_event_present`,
`y_fault_present`, `y_switch_present`; fault descriptors `y_fault_line`, `y_fault_location`,
`y_fault_category`, `y_is_grounded`, `y_phase_{A,B,C}`; switching descriptors
`y_switch_category`, `y_switch_action`, `y_switch_target_bus`; timing `dt_start`,
`window_start_time`, `window_duration`.

## `adapt_grid` vs `benchmark` availability (per grid)

The two families share the **same event vocabulary on every grid**, with a single exception:
`no_event` is **`adapt_grid`-only** on the two HV grids (and absent entirely on MV/EHV). Nothing
is `benchmark`-only anywhere.

| grid | adapt_grid types | benchmark types | difference |
|---|:--:|:--:|---|
| DoubleLine (HV) | 16 | 15 | `no_event` adapt_grid-only (1218 ep) |
| TestGrid110kV (HV) | 17 | 16 | `no_event` adapt_grid-only (623 ep) |
| CigreMV (MV) | 21 | 21 | identical |
| IEEE-39 (EHV) | 15 | 15 | identical |

So fault and switching types transfer cleanly (every such type in `adapt_grid` is also in
`benchmark`); only an "event-vs-quiescent" task lacks a `benchmark` negative. The families still
differ by **construction** — `benchmark` is a systematic discrete-parameter sweep at a fixed
operating point (e.g. fault location ∈ {1,20,50,80,99}%), `adapt_grid` is randomly sampled — which
is the real transfer challenge. Approx. benchmark episode counts: DoubleLine 1101, TestGrid 2435,
CigreMV 4509, IEEE-39 9711.

## Mapping to benchmark tasks

- **Fault classification (FC)** uses the **9-class** fault taxonomy above (the 7 universal
  SHC/incipient subtypes + the 2 MV-only HIF subtypes); HV/EHV grids have zero HIF support.
- **Fault detection (FD)** / **fault location (FL)** key off `y_fault_present` / `y_fault_location`.
- **Event-state classification (ES, `event_state_global`)** — 3-class `normal` / `switching` /
  `fault` over **every** window, keyed off `overlap_ratio` × `event_family` (see
  [task_cards/event_state.md](task_cards/event_state.md)). The `normal` class comes from every
  non-overlapping (pre-/post-event) window, so it has support on all four grids — not just the
  HV grids that carry `no_event`. The high-level middle ground between FD and FC.
- **Event-scope tasks (global-only):** **event detection** (`event_detection_global`, any active
  event vs. quiescent), **switch detection** (`switch_detection_global`), **switch-type**
  (`switch_type_global`, 8 categories from `y_switch_category`), and **full event classification**
  (`event_classification_global`, 23-class from `event_type`, non-active → `no_event`).
- **Fault-attribute tasks (fault-active windows, all 3 observability):** **grounded vs ungrounded**
  (`fault_grounded_*`, `y_is_grounded`), **faulted-phase** (`fault_phase_*`, 7-class from
  `y_phase_*`), **fault-category** (`fault_category_*`, shc/incipient/hif from `y_fault_category`).
- **Fault origin (`fault_origin_global`):** binary line-vs-bus from the `y_fault_line` prefix
  (`MainLn*` vs `MainBus*`), **global-only** (the local/line oracle is keyed on `y_fault_line`, so it
  would leak). Line vs bus is *also* a leakage-free **stratification axis** (`origin`) reported for
  every fault task.
- See [task_suite_roadmap.md](task_suite_roadmap.md) for the full difficulty-tiered catalog and the
  deferred axes (noise-robustness harness, `cross_topology`, resistance stratification) and the
  mechanism-blocked tasks (faulted-line/zone ID; multi-label phase).
