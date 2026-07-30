# Task card: `fault_classification_{local,line,global}` (FC)

| | |
|---|---|
| **Objective** | Classify the fault type, on fault-active windows only |
| **Task ids** | `fault_classification_local`, `fault_classification_line`, `fault_classification_global` |
| **Observability** | `local` (6 ch) ⊂ `line` (12 ch) ⊂ `global` (grid-dependent, 48–420 ch) — same nested oracle as FD (see [grids.md](../grids.md)) |
| **Output** | Multiclass over the **9-class fault taxonomy** (7 universal SHC/incipient subtypes + 2 MV-only HIF subtypes) |
| **Classes** (frozen, index = label id) | 0–6: `flt_1phg_shc`, `flt_1phg_shc_w_arc`, `flt_2ph_shc`, `flt_2phg_shc`, `flt_3ph_shc`, `flt_1phg_incipient`, `flt_1phg_incipient_w_arc`; **7–8 (MV-only): `flt_1phg_hif`, `flt_1phg_hif_w_arc`**. HV/EHV grids have zero HIF support. |
| **Label** | `event_type` → frozen `FAULT_CLASS_CLASSES` index. Arc (`_w_arc`) variants kept distinct. HIF appended last so indices 0–6 are stable. `tasks/labels/fault_class.py`. |
| **Valid samples** | fault-active windows whose fault type is **in the 9-class vocab** (a type outside it is excluded, never mapped to −1) |
| **Protocol** | `held_out` |
| **Metrics** | balanced accuracy, macro F1, accuracy (+ per-class F1) |
| **Baselines** | majority, random_forest, mlp, gru, cnn (threshold N/A — binary only) |

## Notes / claims

- The `benchmark` family is **deliberately imbalanced**: ~16× fewer incipient windows than
  short-circuit windows, which depresses benchmark balanced/macro scores (the incipient classes
  are barely represented). See [RESULTS.md](../../RESULTS.md). The frozen class order keeps indices
  stable so per-class F1 is comparable across runs.
- Report both `held_out` test sets (`adapt_grid` `test` and `benchmark`).
