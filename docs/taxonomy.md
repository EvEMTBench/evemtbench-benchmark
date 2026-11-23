# EvEMTBench — full fault & event taxonomy (all topologies & voltage levels)

Derived from **every** windowed label file on disk — all four reference grids × both families
(`adapt_grid` + `benchmark`), 8 `y_*.parquet` files. Window contract: 50 ms / 5 ms step (l=480),
9600 Hz. Episode counts below are **distinct simulations (`sample_id`) summed over both families**.

## Reference grids & levels

Grid keys, levels and frequencies: [grids.md](grids.md).

**23 distinct `event_type`s = 9 faults + 14 operating events**, in 3 families: `fault`,
`switching`, `normal`.

> **"levels present" = observed, not capability.** The dataset paper tabulates which events the
> *generator simulates per voltage level* (22 event types; `no_event` is not an event there). The
> column below reports what actually **occurs in the four reference grids** of this benchmark, which
> is narrower wherever a specific grid lacks the element: e.g. synchronous-machine trips are
> generated at HV *and* EHV, but among our reference grids only IEEE-39 contributes them; IBR trips
> are MV+HV by design and appear at CIGRÉ MV and TestGrid 110 kV only. Use this column to read the
> not-applicable cells in the result tables; use the dataset paper for what the
> generator can produce.

## Fault events (9) — `event_family = fault`, `event_type` prefix `flt_`

| event_type | category | grounded | faulted phases | levels present | `adapt_grid` episodes |
|---|---|---|---|---|---|
| `flt_1phg_shc` | shc | yes | A · B · C (single, per episode) | MV/HV/EHV | 2420 |
| `flt_1phg_shc_w_arc` | shc | yes | A · B · C | MV/HV/EHV | 2492 |
| `flt_2ph_shc` | shc | **no** | AB · AC · BC | MV/HV/EHV | 2466 |
| `flt_2phg_shc` | shc | yes | AB · AC · BC | MV/HV/EHV | 2416 |
| `flt_3ph_shc` | shc | **no** | ABC | MV/HV/EHV | 2527 |
| `flt_1phg_incipient` | incipient | yes | A · B · C | MV/HV/EHV | 2333 |
| `flt_1phg_incipient_w_arc` | incipient | yes | A · B · C | MV/HV/EHV | 2479 |
| `flt_1phg_hif` | **hif** | yes | A · B · C | **MV only** | 477 |
| `flt_1phg_hif_w_arc` | **hif** | yes | A · B · C | **MV only** | 456 |

Categories: **shc** = solid/low-impedance short circuit; **hif** = high-impedance ground fault;
**incipient** = self-extinguishing high-impedance fault. `_w_arc` = Kizilcay arc model on the
faulted phase. "1phg" is single-line-to-ground on **one** phase (which phase varies per episode
and is recorded in `y_phase_{A,B,C}`).

## Operating / non-fault events (14)

| event_type | family | switch category | action | levels present | `adapt_grid` episodes |
|---|---|---|---|---|---|
| `switch_load_on` / `switch_load_off` | switching | load | on/off | MV/HV/EHV | 2357 / 2408 |
| `switch_cap_on` / `switch_cap_off` | switching | capacitor | on/off | MV/HV/EHV | 2423 / 2363 |
| `switch_ohl_on` / `switch_ohl_off` | switching | ohl | on/off | MV/HV/EHV | 2423 / 2404 |
| `switch_inrush_lv` | switching | inrush | on (LV trafo) | MV/HV/EHV | 2344 |
| `switch_inrush_hv` | switching | inrush | on (HV trafo) | MV/HV (not EHV) | 1728 |
| `switch_ibr_trip` | switching | ibr | trip | MV + TestGrid HV | 1093 |
| `switch_cable_on` / `switch_cable_off` | switching | cable | on/off | **MV only** | 493 / 420 |
| `switch_motor_start` | switching | motor_start | start | **MV only** | 457 |
| `switch_gen_trip` | switching | generator | trip | **EHV only** | 665 |
| `no_event` | normal | — | — | all grids, as the label of inactive windows; no dedicated episodes | — |

## Grid × event availability matrix

`✓` = present in that grid (either family).

| event_type | CigreMV (MV) | DoubleLine (HV) | TestGrid110kV (HV) | IEEE-39 (EHV) |
|---|:---:|:---:|:---:|:---:|
| `flt_1phg_shc`(+`_w_arc`) | ✓ | ✓ | ✓ | ✓ |
| `flt_2ph_shc` / `flt_2phg_shc` / `flt_3ph_shc` | ✓ | ✓ | ✓ | ✓ |
| `flt_1phg_incipient`(+`_w_arc`) | ✓ | ✓ | ✓ | ✓ |
| `flt_1phg_hif`(+`_w_arc`) | ✓ | · | · | · |
| `no_event` (inactive windows; no dedicated episodes) | ✓ | ✓ | ✓ | ✓ |
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
- **`no_event` → no dedicated episodes.** Neither family of any reference grid contains quiescent
  (`no_event`) episodes. The `no_event` / `normal` label is carried by the inactive pre- and
  post-event windows of the fault and switching episodes, so it has support on all four grids.

## Label fields in `y` (per window)

`event_type`, `event_family`, `status`, `overlap_ratio`; targets `y_event_present`,
`y_fault_present`, `y_switch_present`; fault descriptors `y_fault_line`, `y_fault_location`,
`y_fault_category`, `y_is_grounded`, `y_phase_{A,B,C}`; switching descriptors
`y_switch_category`, `y_switch_action`, `y_switch_target_bus`; timing `dt_start`,
`window_start_time`, `window_duration`.

## `adapt_grid` vs `benchmark` availability (per grid)

The two families share the **same event vocabulary on every grid**; nothing is `adapt_grid`-only or
`benchmark`-only.

| grid | adapt_grid types | benchmark types | difference |
|---|:--:|:--:|---|
| DoubleLine (HV) | 15 | 15 | identical |
| TestGrid110kV (HV) | 16 | 16 | identical |
| CigreMV (MV) | 21 | 21 | identical |
| IEEE-39 (EHV) | 15 | 15 | identical |

Episode counts in the tables above are the `adapt_grid` episodes of the four reference grids
(train + validation + test), summed from the committed split metadata
(`src/evemtbench/splits/v1.0/held_out/<grid>/meta.json`). Window support per class in both families
is in the paper's Supplementary Tables S19 and S20.

So fault and switching types transfer cleanly (every such type in `adapt_grid` is also in
`benchmark`); only an "event-vs-quiescent" task lacks a `benchmark` negative. The families still
differ by **construction** — `benchmark` is a systematic discrete-parameter sweep at a fixed
operating point (e.g. fault location ∈ {1,20,50,80,99}%), `adapt_grid` is randomly sampled — which
is the real transfer challenge. Approx. benchmark episode counts: DoubleLine 1101, TestGrid 2435,
CigreMV 4509, IEEE-39 9711.

## Mapping to benchmark tasks

Which label field each task reads: the [task cards](task_cards/); the task list with tiers: the
[README](../README.md#tasks); deferred tasks and axes: [task_suite_roadmap.md](task_suite_roadmap.md).
