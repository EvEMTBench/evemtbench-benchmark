# EvEMTBench task-suite roadmap

EvEMTBench aims to be a *universal* power-grid protection/event benchmark with a task range that
spans **easy anchors** — near-solved on ideal (synthetic) data, so the community's frontier there is
adding realism (noise, sensor loss, sim-to-real) — through a **hard frontier** that is unsolved even
on ideal data, where the frontier is better *models*.

This document is the living catalog: what is scored, what is derivable-but-not-yet-scored, the
difficulty tiers (with empirical evidence), and the deferred difficulty *axes*. The mechanical
"how to add a task" checklist lives in [extending.md](extending.md); per-task contracts live in
[task_cards/](task_cards/).

## Difficulty comes from two things

1. **Task intrinsic difficulty** — separability of the label from a 50 ms / 480-sample window.
2. **Generalization axis** — the protocol: `held_out` scores an in-distribution `test` split
   (`adapt_grid`) *and* a cross-distribution `benchmark` split (a systematic operating-point sweep).
   Many tasks are near-solved in-distribution yet collapse under the operating-point shift; true
   cross-*topology* transfer is a further, not-yet-benchmarked step.

## Scored tasks (24 task ids)

Observability: `local ⊂ line ⊂ global` (oracle keyed on the faulted line). Objectives scored on
switching/normal windows are **global-only** (no faulted line to center the local/line view on).

### Tier 1 — easy anchors (near-solved on ideal data → community adds realism)
| Task | id(s) | kind | obs | evidence |
|---|---|---|---|---|
| Fault detection (FD) | `fault_detection_{local,line,global}` | binary | 3 | **0.89–0.91** bal-acc in-dist → **0.99** transfer (resnet) |
| Event detection (ED) | `event_detection_global` | binary | global | expected easy (broadest positive set) |
| Switch detection | `switch_detection_global` | binary | global | expected easy |
| Grounded vs ungrounded | `fault_grounded_{local,line,global}` | binary | 3 | strong residual-current signal |

### Tier 2 — medium (learnable in-dist on HV; harder on MV / under transfer)
| Task | id(s) | kind | obs | evidence |
|---|---|---|---|---|
| Event-state (ES) | `event_state_global` | 3-class | global | best `test` bal-acc **0.73** (gru), floor 0.33; transfer 0.80–0.84 |
| Fault classification (FC) | `fault_classification_{local,line,global}` | 9-class | 3 | HV **~0.86**, MV/HIF **0.59–0.67** |
| Faulted-phase | `fault_phase_{local,line,global}` | 7-class | 3 | phase selection; medium |
| Fault-category | `fault_category_{local,line,global}` | 3-class | 3 | shc/incipient/hif; HIF makes MV hard |
| Switch-type | `switch_type_global` | 8-class | global | benign-event subtyping |
| Fault origin | `fault_origin_global` | binary | global | line vs bus (busbar vs line protection); global-only |

### Tier 3 — hard frontier (hard even on ideal data → community finds better models)
| Task | id(s) | kind | obs | evidence |
|---|---|---|---|---|
| Full event classification (EC) | `event_classification_global` | 23-class | global | fine-grained + heavily imbalanced |
| Fault location (FL) | `fault_location_{local,line,global}` | regression | 3 | `line`/`global` MAE 16%→R²>0.9, but **`local` on MV/TestGrid ties the mean predictor** (unsolved) |

**HIF as a hard subproblem:** high-impedance faults exist only on the MV grid; `fault_category`
and FC both expose them (zero-support NaN per-class F1 elsewhere). HIF detection/discrimination is a
known-hard, safety-critical target that lives inside these tasks today; a dedicated MV HIF task is a
possible future addition.

## Difficulty axes (deferred — the roadmap's next phases)

- **Noise / perturbation robustness** (the "community adds noise" axis): a post-hoc evaluation that
  degrades the ideal inputs (additive noise / SNR sweep, channel dropout / sensor loss, amplitude
  quantization, timing jitter) and reports metric-vs-degradation curves per task on saved
  checkpoints — no retraining. Turns the Tier-1 anchors into live research targets.
- **`cross_topology` protocol + `gnn` baseline**: train on the diverse-topology `train` family
  (`multi_grid`, e.g. `Template20kV` — **105+ varied networkx topologies** already on disk under
  `multi_grid/.../graphs/`), test on each unseen grid — true topology transfer, beyond the current
  operating-point shift. The graph artifacts exist (see "Auxiliary artifacts" below); still needs the
  `train` family windowed + a variable-channel `gnn`.
- **Fault-resistance stratification / regression**: fault resistance is a physical difficulty knob
  and is **available per-episode** in each grid's `labels/settings.csv`
  (`events/event_flt_shc_resistance`, `event_flt_hif_resistance`; `grounding/*_resistance` on MV),
  joinable to windows by `general/sim_idx` ↔ `sample_id`. So it is a near-term data-plumbing task
  (surface it into `y`), not blocked — enables resistance-binned FD/HIF/FL difficulty curves and a
  resistance-regression task.

## Auxiliary artifacts (topology graphs + simulator settings)

Beyond the windowed `X`/`y`, every family archive carries two extra artifact types (confirmed in
`benchmark/DoubleLine_benchmark.tar.gz` → `DoubleLine/{graphs,labels}/`; `adapt_grid` symmetric;
`multi_grid` slices have them uncompressed):

- **`graphs/graph_<family>.pickle`** — a networkx `MultiGraph` of the grid. Nodes = buses/elements
  (ExtGrid/slack, loads, IBRs, switches) with per-element `param_dict`s; edges = transformers/lines
  with electrical `param_dict` + `typ_param_dict` (ratings, vector group, sequence impedances,
  saturation curves); graph-level attrs = neutral-grounding treatment (Petersen coil / resistive /
  solid). The **reference grids have ONE fixed topology** (the sweep varies the fault, not the
  topology); **`multi_grid` has many varied topologies** — the cross-topology training ground. Enables
  GNN node/edge features, and structural labels (IBR penetration/count, grid size, grounding scheme).
- **`labels/settings.csv`** — the per-episode simulator parameter table, keyed by `general/sim_idx`.
  Reference-grid schema (~19 cols): `event_type`, `event_target` (faulted element), fault
  `location`/`resistance`, phase-select, arc params. `multi_grid` schema (~90 cols) additionally
  carries `grounding/*` (type + resistances), `arc/*`, per-load P/Q operating point, `ExtGrid` V
  setpoint, and component types. Source of many candidate labels not yet in `y` (grounding-type
  classification, resistance/arc/duration regression, operating-point regression).

## Derivable but not scored (candidate future tasks)

All are supported by existing `y` columns; see [taxonomy.md](taxonomy.md) for the label fields.
- **Switch-action** (`y_switch_action`: on/off/trip/start) — small multiclass, global-only.
- **Onset-time / detection-latency regression** (`dt_start`) — currently a post-hoc metric on the
  binary detectors rather than its own task.

### Blocked on new machinery (not reuse-only)
- **Faulted-line / zone identification** (`y_fault_line`): two blockers — (1) **leakage**, because
  `y_fault_line` *is* the observability oracle, so at `local`/`line` the input is selected to sit on
  the answer → must be **global-only** and flagged; (2) a **per-grid variable-size class space**
  (line+bus set differs per grid), which breaks the single frozen-vocab / fixed-`num_classes`
  assumption. High protection value; needs a per-grid class-vocab mechanism. Its **coarse line-vs-bus
  version ships** as `fault_origin_global` (binary, global-only for the same leakage reason); the
  fine per-element identifier remains blocked on the per-grid class space.
- **Multi-label faulted-phase** (independent A/B/C bits): needs a new `LabelKind.MULTILABEL`
  threaded through the criterion (BCE), every baseline's `predict` (sigmoid threshold), predictions
  I/O (multi-column), and metrics (per-label / subset / Hamming). The shipped 7-class single-label
  `fault_phase` covers the common case without this.

## Principles for keeping the suite "universal"

- **Frozen vocabularies** with grid-restricted classes as zero-support (NaN per-class F1), so the
  label space is comparable across grids; report **per grid**, never pooled.
- **Re-derive labels** from the raw window fields (activity × family/type) rather than echoing a
  precomputed column, so the label contract is explicit and testable.
- **Fairness:** never compare across observability settings blindly; `benchmark` (transfer) is the
  hard column; fit only on `train`.
- **Stratified reporting:** every result is broken down by `grid`, `event_family`, `event_type`, and
  **`origin`** (line vs bus, derived from `y_fault_line`) — so any fault task can be read per fault
  origin. Stratifying *metrics* by `y_fault_line` is leakage-free (it is not a model input), unlike
  using it as a prediction label.
