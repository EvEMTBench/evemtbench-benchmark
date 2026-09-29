# Task card: `fault_detection_{local,line,global}` (FD)

| | |
|---|---|
| **Objective** | Window-level fault detection — is a fault active in this window? (Benchmark Plan §4.1) |
| **Task ids** | `fault_detection_local`, `fault_detection_line`, `fault_detection_global` |
| **Observability** | `local` (near terminal — 1 cubicle, 6 ch) ⊂ `line` (both terminals — 2 cubicles, 12 ch) ⊂ `global` (all main-line cubicles, **grid-dependent** width: 48 ch DoubleLine … 420 ch IEEE-39). Nested oracle keyed by `y_fault_line`; conventional-protection analogs single-end / **differential** / centralized. See [grids.md](../grids.md). |
| **Output** | Binary: fault active in window (1) vs. pre-fault (0) |
| **Label rule** | `1` iff `event_type` is `flt_*` **and** the window overlaps the event (`overlap_ratio > 0`); else `0`. The negatives are the **pre-fault windows of the same fault episode**. Pure + test-pinned (`tasks/labels/event_present.py`). |
| **Valid samples** | All windows of **fault episodes** (`y_fault_line` not null), so the local/line/global oracle is defined for every scored window. |
| **Protocol** | `held_out` (all variants); the `local`/`line` variants also run `multi_grid_pretrain`, `transfer_zeroshot` and `transfer_finetune` (deep baselines only — see [protocol.md](../protocol.md)) |
| **Metrics** | balanced accuracy, macro F1, **missed-fault rate**, **false-alarm rate**, accuracy |
| **Baselines** | majority, threshold (overcurrent; `local` only), random_forest, mlp, gru, cnn, resnet |

## Notes / claims

- **What this task is, and is not.** FD is **line-conditioned onset detection**, not end-to-end
  grid-wide fault detection or relay selection. Two design choices make that so, and both are
  deliberate (the *oracle* framing, framework §2.4):
  1. **Only fault episodes are scored** (`tasks/valid_samples.py`: `y_fault_line` not null). No
     switching or normal episode ever enters FD, so the negatives are exclusively the **pre-onset
     windows of episodes that do go on to fault**. The question is therefore "has the fault started
     yet?", not "is anything wrong on this grid?".
  2. **At `local`/`line`, the input window is selected by the ground-truth faulted line**
     (`tasks/task.py::_observe`, keyed on the episode-constant `y_fault_line`). The model is shown
     the right terminals; it is not asked to find them.
  Consequence to keep in mind when reading the leaderboard: a model that trips on *any* disturbance
  scores near-perfectly on `fault_detection_local` under this oracle — it only ever sees in-zone
  faults — and would be unusable as a relay. Deployed performance is **overstated** by construction;
  the numbers compare *representations under matched information*, which is what the benchmark is
  for. See [relay_selectivity.md](relay_selectivity.md) for the un-oracled formulation.
- `held_out` scores **two** test sets: the in-distribution `adapt_grid` `test` split and the
  never-trained-on `benchmark` family. Report both — they can diverge sharply.
- Never compare across observability settings without stating which (fairness rule, framework §2.4).
- See [RESULTS.md](../../RESULTS.md): FD learns well in-distribution; on the `benchmark` family the
  magnitude-reliant `local`/`global` views collapse under the `adapt_grid`→`benchmark` current-scale
  shift while the **differential `line` view transfers** — and this is **grid-dependent** (CigreMV
  transfers, TestGrid collapses). Threshold (a pure overcurrent rule) collapses the same way, so it
  is a shift diagnostic, not just a weak baseline. Multi-grid; baselines incl. `cnn`.
