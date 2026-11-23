# EvEMTBench task-suite roadmap

EvEMTBench aims to be a *universal* power-grid protection/event benchmark with a task range that
spans **easy anchors** — near-solved on ideal (synthetic) data, so the community's frontier there is
adding realism (noise, sensor loss, sim-to-real) — through a **hard frontier** that is unsolved even
on ideal data, where the frontier is better *models*.

This document is the living catalog: what is scored, what is derivable-but-not-yet-scored, the
difficulty tiers, and the deferred difficulty *axes*. The mechanical
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
Current numbers per task: [RESULTS.md](../RESULTS.md).

### Tier 1 — easy anchors (near-solved on ideal data → community adds realism)
| Task | id(s) | kind | obs | note |
|---|---|---|---|---|
| Fault detection (FD) | `fault_detection_{local,line,global}` | binary | 3 | near-solved in-distribution |
| Event detection (ED) | `event_detection_global` | binary | global | expected easy (broadest positive set) |
| Switch detection | `switch_detection_global` | binary | global | expected easy |
| Grounded vs ungrounded | `fault_grounded_{local,line,global}` | binary | 3 | strong residual-current signal |

### Tier 2 — medium (learnable in-dist on HV; harder on MV / under transfer)
| Task | id(s) | kind | obs | note |
|---|---|---|---|---|
| Event-state (ES) | `event_state_global` | 3-class | global | coarse grid state over every window |
| Fault classification (FC) | `fault_classification_{local,line,global}` | 9-class | 3 | HIF makes MV hard |
| Faulted-phase | `fault_phase_{local,line,global}` | 7-class | 3 | phase selection; medium |
| Fault-category | `fault_category_{local,line,global}` | 3-class | 3 | shc/incipient/hif; HIF makes MV hard |
| Switch-type | `switch_type_global` | 8-class | global | benign-event subtyping |
| Fault origin | `fault_origin_global` | binary | global | line vs bus (busbar vs line protection); global-only |

### Tier 3 — hard frontier (hard even on ideal data → community finds better models)
| Task | id(s) | kind | obs | note |
|---|---|---|---|---|
| Full event classification (EC) | `event_classification_global` | 23-class | global | fine-grained + heavily imbalanced |
| Fault location (FL) | `fault_location_{local,line,global}` | regression | 3 | `local` view is the hard case |

**HIF as a hard subproblem:** high-impedance faults exist only on the MV grid; `fault_category`
and FC both expose them (zero-support NaN per-class F1 elsewhere). HIF detection/discrimination is a
known-hard, safety-critical target that lives inside these tasks today; a dedicated MV HIF task is a
possible future addition.

## Difficulty axes (deferred — the roadmap's next phases)

The next benchmark version, **2.0.0 on dataset V1**, is planned separately (development roadmap; paused until
after the 1.x paper revision). It re-runs the existing suite on the new data and changes how the active-window
label is derived ([known_issues.md](known_issues.md) §6); the axes below come after it.

- **Relay selectivity** *(designed, not built)*: a fixed-relay task with out-of-zone faults, so that zone
  selection and selectivity are scored instead of assumed ([design/selectivity.md](design/selectivity.md)).

- **Noise / perturbation robustness** (the "community adds noise" axis): a post-hoc evaluation that
  degrades the ideal inputs with the benchmark-owned perturbation taxonomy
  ([design/robustness.md](design/robustness.md): noise/SNR, harmonics, CT saturation, sensor
  dropout, time skew, sampling-rate mismatch) and reports metric-vs-degradation curves per task on
  saved checkpoints — no retraining. Turns the Tier-1 anchors into live research targets.
- **`cross_topology` protocol + `gnn` baseline** *(planned, not implemented)*: train on the
  diverse-topology `multi_grid` family, test on each unseen grid — true topology transfer, beyond
  the current operating-point shift. The graph artifacts exist, and `multi_grid` is already used
  for pre-training, but only in the portable 12-channel view (`local` / `line` inputs;
  [design/pretrain_transfer.md](design/pretrain_transfer.md)). Still needed: the `cross_topology`
  protocol itself, a variable-channel (cubicle-set) view, and the `gnn`
  ([design/cross_topology.md](design/cross_topology.md)).
- **Fault-resistance regression**: the per-episode fault resistance is in each grid's
  `settings.csv` and already feeds the `resistance_bin` breakdown. A resistance-regression task is
  not built.

## Auxiliary artifacts (topology graphs + simulator settings)

Layout and join keys: [DATA_LAYOUT.md](DATA_LAYOUT.md#auxiliary-overlay-additive-never-writes-the-data-root).
The `multi_grid` settings (~90 columns) carry many candidate labels not yet in `y`: grounding type,
arc parameters, operating point.

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
