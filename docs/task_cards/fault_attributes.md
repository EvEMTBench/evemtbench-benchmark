# Task card: fault-attribute tasks (grounded / phase / category) — **SHIPPED**

Three protection-relevant attributes of an *active fault*, complementary to the fault triad
(FD/FC/FL). All three are scored on **fault-active windows only** (where `y_fault_line` is defined),
so all three observability settings apply — registered as `{local,line,global}` variants like FD/FC/FL.

| Task | id(s) | source | kind | classes |
|---|---|---|---|---|
| Grounded vs ungrounded | `fault_grounded_{local,line,global}` | `y_is_grounded` | binary | grounded=1 / ungrounded=0 |
| Faulted-phase | `fault_phase_{local,line,global}` | `y_phase_{A,B,C}` | multiclass | 7: A,B,C,AB,AC,BC,ABC |
| Fault-category | `fault_category_{local,line,global}` | `y_fault_category` | multiclass | 3: shc, incipient, hif |
| Fault-origin | `fault_origin_global` | `y_fault_line` prefix | binary | bus=1 / line=0 |

**Fault-origin is global-only** (unlike the other three): `y_fault_line` *is* the observability
oracle, so at local/line the input window is selected to sit on the faulted element → the origin
would leak. See [fault_origin.py](../../src/evemtbench/tasks/labels/fault_origin.py). The line-vs-bus
distinction is *also* a leakage-free **stratification axis** (`origin`) reported for every fault
task — busbar vs line protection are distinct relay schemes, so per-origin FD/FC/FL scores matter.

Metrics: `balanced_accuracy`, `macro_f1`, `accuracy` (+ per-class F1 for the multiclass ones).
Derivers: [fault_grounded.py](../../src/evemtbench/tasks/labels/fault_grounded.py),
[fault_phase.py](../../src/evemtbench/tasks/labels/fault_phase.py),
[fault_category.py](../../src/evemtbench/tasks/labels/fault_category.py). Frozen vocabularies (index =
id); a change is a **major** version bump.

## Notes

- **Grounded** engages ground/residual-current relay elements; ungrounded = the phase faults
  `flt_2ph_shc` / `flt_3ph_shc`. Physically strong signal → expected to be an easy anchor.
- **Faulted-phase** is single-label over the 7 phase combinations (relay phase selection). A true
  independent-per-phase (multi-label) variant is deferred (needs a new `LabelKind`; see the
  [roadmap](../task_suite_roadmap.md)).
- **Fault-category** separates the mechanism: `shc` (solid short circuit), `incipient`
  (self-extinguishing HIF), `hif` (high-impedance ground fault). **HIF occurs only on the MV grid**
  (CigreMV) → zero support on HV/EHV (NaN per-class F1, like FC's HIF). It is the notoriously hard,
  safety-critical class — a Tier-3 signal riding inside a Tier-2 task.
- `y_fault_category` is a contract-*optional* column; its deriver fails loud if the column is absent.
