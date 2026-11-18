# Task card: `fault_detection_{local,line,global}` (FD)

| | |
|---|---|
| **Objective** | Window-level fault detection — is a fault active in this window? |
| **Task ids** | `fault_detection_local`, `fault_detection_line`, `fault_detection_global` |
| **Observability** | `local` (near terminal — 1 cubicle, 6 ch) ⊂ `line` (both terminals — 2 cubicles, 12 ch) ⊂ `global` (all main-line cubicles, **grid-dependent** width: 48 ch DoubleLine … 420 ch IEEE-39). Nested oracle keyed by `y_fault_line`; conventional-protection analogs single-end / **differential** / centralized. See [grids.md](../grids.md). |
| **Output** | Binary: fault active in window (1) vs. pre-fault (0) |
| **Label rule** | `1` iff `event_type` is `flt_*` **and** the window overlaps the event (`overlap_ratio > 0`); else `0`. The negatives are the **pre-fault windows of the same fault episode**. Pure + test-pinned (`tasks/labels/event_present.py`). |
| **Valid samples** | All windows of **fault episodes** (`y_fault_line` not null), so the local/line/global oracle is defined for every scored window. |
| **Protocol** | `held_out` (all variants); the `local`/`line` variants also run `multi_grid_pretrain`, `transfer_zeroshot` and `transfer_finetune` (deep baselines only — see [protocol.md](../protocol.md)) |
| **Metrics** | balanced accuracy, macro F1, **missed-fault rate**, **false-alarm rate**, accuracy |
| **Baselines** | majority, threshold (overcurrent; `local` only), random_forest, mlp, gru, cnn, resnet, tst |

## Notes / claims

- **Oracle task.** FD is **line-conditioned onset detection**, not grid-wide detection: only fault
  episodes are scored, and at `local`/`line` the input is chosen by the true faulted line. Deployed
  performance is overstated by construction ([protocol.md](../protocol.md#task-matrix)). The
  un-oracled formulation is [relay_selectivity.md](relay_selectivity.md).
- `held_out` scores **two** test sets: the in-distribution `adapt_grid` `test` split and the
  never-trained-on `benchmark` family. Report both — they can diverge sharply.
- Never compare across observability settings without stating which.
- `threshold` (a pure overcurrent rule) is the conventional anchor: where it collapses on
  `benchmark`, that diagnoses the operating-point shift. Current numbers: [RESULTS.md](../../RESULTS.md).
