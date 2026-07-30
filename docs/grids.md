# EvEMTBench — reference grids & observability

Companion to [taxonomy.md](taxonomy.md): taxonomy.md documents **what events** occur; this
documents **on what topologies** and **how the model observes them**. All figures below are
derived from the on-disk windows `meta.json` (`feature_names`) via
[`tasks/line_view.py`](../src/evemtbench/tasks/line_view.py)’s `build_oracle` — the same path the
benchmark uses at runtime — not hand-written.

## The four reference grids

| key | topology id | level | voltage | freq | on-disk channels | cubicles | sources |
|---|---|---|---|---|---|---|---|
| `cigre_mv` | `mv_cigre_distribution_20kv` | MV | 20 kV | 50 Hz | 270 | 45 | IBR |
| `double_line` | `hv_double_line_110kv` | HV | 110 kV | 50 Hz | 66 | 11 | sync + ext-grid |
| `testgrid_110kv` | `hv_testgrid_110kv` | HV | 110 kV | 50 Hz | 150 | 25 | IBR |
| `ieee39` | `ehv_ieee39_transmission_345kv` | EHV | 345 kV | 60 Hz (only 60 Hz grid) | 594 | 99 | synchronous |

Each cubicle carries 6 channels — `[Isec, Usec] × [L1, L2, L3]` (3-phase current + voltage),
sampled point-on-wave at 9600 Hz; the window contract is 50 ms / 5 ms step (l = 480).

## Main-line universe & observability (per grid)

The benchmark observes only the **main-line bays** (cubicles named `Bus_<bus>_Line_<lineid>`),
dropping transformer / load / IBR / generator / cap-bank cubicles. That line universe is the
**`global`** view. Its width is **grid-dependent** — this variability across grids is exactly what
motivates topology-agnostic models (GNNs) over fixed-input models.

| grid | main lines | buses | line cubicles | **global** width | `line` width | `local` width |
|---|---|---|---|---|---|---|
| `double_line` | 4 | 3 | 8 | **48** ch | 12 ch | 6 ch |
| `cigre_mv` | 15 | 14 | 29 | **174** ch | 12 ch | 6 ch |
| `testgrid_110kv` | 9 | 6 | 18 | **108** ch | 12 ch | 6 ch |
| `ieee39` | 35 | 28 | 70 | **420** ch | 12 ch | 6 ch |

The three observability settings form a nested oracle centred on the faulted element
(`y_fault_line`, episode-constant). Naming the conventional-protection analogs:

- **`local`** = the **near terminal** of the faulted line (1 cubicle, 6 ch) — a *single-ended*
  view, the information set of overcurrent / distance relays.
- **`line`** = **both terminals** of the faulted line (2 cubicles, 12 ch) — a two-ended
  **current-differential** view. A line’s two terminals are the two cubicles that share a `lineid`
  at different buses; this is *the* setting that transfers under operating-point shift, because the
  differential cancels the shared load term.
- **`global`** = **all main-line cubicles** (grid-dependent width above) — a centralized /
  wide-area view.

`local`/`line` are always 6/12 ch regardless of grid (one line’s terminals); only `global` scales
with topology. A **bus fault** (`MainBus*`) has no single line; by convention it maps to the line
bays physically at that bus (the relays electrically adjacent to the fault) — revisable, defined in
`line_view.py`.

## Grid-restricted events (see taxonomy.md for the full matrix)

- **MV-only (`cigre_mv`):** high-impedance faults (`flt_1phg_hif*`), cable switching, motor start.
- **EHV-only (`ieee39`):** generator trip; and HV-transformer inrush is *absent* on EHV.
- **IBR trip:** only grids with inverter-based resources (`cigre_mv`, `testgrid_110kv`).
- **`no_event` (quiescent background):** only the HV grids (`double_line`, `testgrid_110kv`).

## How a new grid is added

Register a `GridSpec` in [`data/roots.py`](../src/evemtbench/data/roots.py) (key, topology id, raw
folder, voltage, frequency, level); the observability oracle then derives automatically from that
grid’s `feature_names` — no per-grid topology file. Generate its `held_out` split with
`python -m evemtbench.splits.generator --grids <key>` (a not-yet-windowed benchmark family yields
an empty benchmark partition → in-distribution test only). See [extending.md](extending.md).
