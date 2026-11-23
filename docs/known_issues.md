# Known issues (benchmark contract v1.3.0)

Dataset and scoring properties that affect published numbers. Each sits in an artifact that is
frozen within the major version (the valid-sample rules, the line-view oracle and the metric
definitions, see [versioning.md](versioning.md)).
Issues 1, 3, 5 and 6 are flagged only: v1.x keeps the published behaviour, and a behavioural fix
belongs to the next major contract version. Issue 2 was corrected by the scoped revision v1.2.0,
for the fault-location tasks alone, and issue 4 by the scoped revision v1.3.0, for the
faulted-phase cells that read the repaired labels alone. Each is documented here so a method author can account for them.
Numbers below were measured on the released windows on 2026-10-01 (issues 1–3) and 2026-11-13
(issues 4–6).

## 1. One-ended line in the CIGRE MV `line` view

**What.** The line-view oracle pairs every main line with a (near, far) terminal cubicle
(`src/evemtbench/tasks/line_view.py`). CIGRE MV line **8-14** has only one terminal in the export:
the producer exports `Bus_MainBus14_Line_8-14`, but not the Bus 8 end. The grid model has that
cubicle (`Cub_4`, from the topology graph), but it is not measured. For such a line the `line` view
fills the far slot with the near terminal, so its 12 channels are the same 6 channels twice.

**Scope.** CIGRE MV only, in both `adapt_grid` and `benchmark`; the other three reference grids
have no one-ended lines. Line 8-14 accounts for about 3.5 % of the valid `fault_*_line` samples in
`adapt_grid` and 5.4–6.7 % in `benchmark`.

**Reference end.** `y_fault_location` is measured from the **lower-numbered bus** of the line. A
voltage-dip test on every two-ended CIGRE MV line confirms it (14 of 14 lines): the difference
between the two terminals' fault-time residual voltages correlates positively with the location
in every one (Spearman ρ = 0.41–0.92, within fault type, high-impedance faults excluded). That is
also the terminal the oracle uses as *near*. Line 8-14's single terminal cannot confirm it by
itself (ρ = 0.09), but by the same convention its origin is **Bus 8, the end that is not
measured**. So on this line the `local` view and the `line` view's near slot both show the far end
relative to the label, and a one-ended estimate from them must be mirrored (100 − location) to
match it.

**What the code does.** `GridOracle.one_ended_lines` lists the affected lines (`("MainLn8-14",)` on
CIGRE MV), and building the oracle emits one `OneEndedLineWarning` (a `UserWarning` subclass in
`evemtbench.tasks.line_view`) naming them; filter that class if you run with warnings as errors. A two-ended method
should treat these lines as one-ended rather than use the duplicated far terminal; the `impedance`
reference baseline does (it falls back to its one-terminal method there, mirrors the estimate to
the label's reference end and lists the line in its run record). The same
padding applies to a bus fault at a bus with a single line bay (`MainBus*` pairs); that case is a
labelling convention for bus faults and is not listed in `one_ended_lines`.

## 2. Bus-target fault episodes inside the fault-location target (corrected in v1.2.0)

**Status.** Corrected in contract v1.2.0: the fault-location (FL) valid samples now require a line
target. Everything below describes the **v1.0.0 / v1.1.0 results**, which stay on disk and in the
frozen `results/1.0.0/` and `results/1.1.0/` bundles. The FL results of v1.2.0 are re-run without
the bus-target windows and are not comparable to them (see [versioning.md](versioning.md)).

**What.** Under v1.0/v1.1 the FL valid samples were the fault-active windows with a non-null
`y_fault_location` (`src/evemtbench/tasks/valid_samples.py`). The intent was line faults only. But
in the `adapt_grid` family the producer also fills a location for **bus-target** fault episodes
(`y_fault_line` = `MainBus*`): one value per episode, spread over 1–99 %, describing no position on
any line. Those windows passed the gate.

**Scope.** Bus-target windows made up 38.9–46.6 % of the v1.0/v1.1 FL samples in `adapt_grid`:

| Grid | Bus-target share of FL samples (`adapt_grid`, v1.0/v1.1 gate) |
|---|---|
| CIGRE MV | 46.6 % |
| Double Line | 43.5 % |
| TestGrid110kV | 38.9 % |
| IEEE-39 | 43.9 % |

In the `benchmark` family bus-target locations are NaN, and the `multi_grid` pre-training corpora
have no bus-target windows in the gate, so **no** bus-target window was scored on `benchmark`. The
effect was on FL training and validation and on the in-distribution `test` scores, for `held_out`
and the transfer protocols that fine-tune or test on a reference grid's `adapt_grid`.

**What the code does.** Since v1.2.0 the gate is fault-active AND non-null `y_fault_location` AND
`y_fault_line` starts with `MainLn`. `evemtbench.tasks.valid_samples.line_target_fault_location_mask(y)`,
the former opt-in line-only diagnostic, now returns the same mask as the gate. The FL
`multi_grid_pretrain` source models are unchanged by the correction and were not re-trained.

## 3. Binary `macro_f1` on single-class breakdown subsets

**What.** For binary tasks, `macro_f1` averages F1 over the classes that occur in `y_true` *or*
`y_pred` (`src/evemtbench/evaluation/metrics.py`); unlike the multiclass path it is not pinned to a
fixed label set. On a subset that contains only one true class, the score therefore depends on
whether the model ever predicts the other class: all-correct gives 1.0, while a single wrong
prediction adds the absent class with F1 = 0 and roughly halves it. `balanced_accuracy` is `NaN`
there, as [metrics.md](metrics.md) describes; `macro_f1` is not.

**Scope.** Only the stratified **breakdowns** that are single-class by construction, for example
`fault_origin_global` by origin, `fault_grounded_*` by event type, or the `normal` event family of
the detection tasks. The headline numbers are computed over the whole test set, which
holds both classes, and are unaffected.

**How to read it.** Treat binary `macro_f1` in a single-class breakdown row as undefined. What to
read instead depends on the task's metric set (`src/evemtbench/tasks/registry.py`):

- Every binary task emits `accuracy` and `macro_f1_supported`. In a single-class row `accuracy` is
  the recall of the one class present, and `macro_f1_supported` is that class's F1.
- The detection tasks also emit class-conditional rates: `missed_fault_rate` / `false_alarm_rate`
  for `fault_detection_*`, and `missed_event_rate` / `false_event_trip_rate` for
  `event_detection_global` and `switch_detection_global`. In a single-class row only the rate of
  the class present is defined (the missed rate in an all-positive row, the false-alarm rate in an
  all-negative row); the other has a zero denominator and is `null`, not 0.
- `fault_origin_global` and `fault_grounded_*` emit no class-conditional rate (their metric set is
  `balanced_accuracy`, `macro_f1`, `accuracy`), so use `accuracy` and `macro_f1_supported` there.

The values are stored unchanged. `evaluate` emits one `SingleClassBinaryBreakdownWarning` (a
`UserWarning` subclass in `evemtbench.evaluation.stratify`, filterable on its own) per evaluation of
a binary task that lists every single-class breakdown group with its size and the class present. It
fires on the true labels alone, whether the predictions in that group are right or not, and never
for multiclass or regression tasks. Results stored before the warning existed carry no marker.

## 4. High-impedance faults with arc had no faulted phase (corrected in v1.3.0)

**Status.** Corrected in contract v1.3.0 by label patch LP-1 (`src/evemtbench/data/label_patches/`),
for the `fault_phase` cells that read a repaired label. Everything below describes the **v1.0.0 /
v1.1.0 `fault_phase` results** of those cells, which stay on disk and in the frozen bundles; the
v1.3.0 results are re-run with the class and are not comparable to them (see
[versioning.md](versioning.md)).

**What.** In the 1.x windows every fault-active window of `flt_1phg_hif_w_arc` (high-impedance
single-phase-to-ground fault with arc) has `y_phase_A = y_phase_B = y_phase_C = 0`. The
`fault_phase` valid-sample rule keeps fault-active windows with at least one phase bit, so it
silently dropped the whole class: the published faulted-phase numbers on CIGRE MV and the
Template20kV corpus measured the task without it. The producer's one-phase rule
lists every other `flt_1phg*` type but not this one, and an unlisted type falls through to "no phase" without an error. The source label
`events/event_phase_select_1ph` names exactly one phase for every affected episode.

**Scope.** Only this event type, only on the MV grids that simulate it:

| partition | fault-active windows without a phase bit | episodes |
|---|--:|--:|
| CIGRE MV `adapt_grid` | 7,752 | 456 |
| CIGRE MV `benchmark` | 1,392 | 87 |
| `multi_grid` Template20kV (line view) | 4,318 | 254 |

No other type, grid or family has a fault-active window without a phase bit, and for every other
fault type the window bits equal the source label on 100 % of episodes. Phase bits are constant
within an episode, so an episode-level repair is exact. Only `fault_phase_*` reads the phase bits
(`fault_grounded`, `fault_category` and the classification tasks read other columns, which are
correct for this type), and the splits do not change (they stratify by event type).

The repair adds 12.0 / 12.0 / 12.1 % valid samples to the CIGRE MV `adapt_grid` train / val /
test, 2.1 % to its `benchmark`, and 14.0 / 16.3 / 14.4 % to the Template20kV train / val / test, all
in the single-phase classes A, B and C.

**What the code does.** Since v1.3.0, `load_task` applies LP-1 to the `fault_phase` tasks of the
v1.3.0 cells (`version.results_version(protocol, task, grid=..., source=...)`): `held_out` on
cigre_mv, `multi_grid_pretrain` on template_20kv, and the transfer cells that target cigre_mv or
start from the template_20kv source (count: `coverage_snapshot.json`). The patch restores the bit from the source
selector (a/b/c -> A/B/C, the producer's rule for the other 1phg types), only on the partition
whose `manifest_sha256` it was built for, and refuses to run if the data does not look exactly
like the defect it repairs. The windows themselves are unchanged. Records stamped before v1.3.0 are
re-evaluated without the patch. Dataset V1 fixes the rule at the source, so benchmark 2.0.0 will
need no patch.

## 5. `multi_grid` switching and busbar-fault episodes appear only as `no_event` windows

**What.** The `multi_grid` pre-training corpora use the portable double-line view: per simulation
one row per line, and only the faulted line is the target line (`is_target_line = 1`, event
carried). Every other row is `no_event` / `normal`. Episodes that have no target line — all
switching events and all faults on a busbar — therefore appear **only** as `no_event` windows,
although their waveforms contain the switching or fault transient.

**Scope.** No current task scores them: the `multi_grid` corpora feed `multi_grid_pretrain` and,
through it, the transfer protocols, whose `local` / `line` tasks gate on the faulted line
(`y_fault_line`), so no switching or busbar-fault window enters a valid sample with a wrong label.
The hazard is outside the task definitions: a method that uses all `multi_grid` windows — as
negatives, or for self-supervised or contrastive pre-training on every window — would treat
switching transients as quiescent signal.

**What to do.** In 1.x, do not use `multi_grid` `no_event` windows as clean negatives. This is a
property of the 1.x corpora and is not changed in 1.x. The producer of dataset V1 (the data for
benchmark 2.0.0) writes switching and busbar-fault episodes with their event labels.

## 6. The window just before an off-grid event onset counts as event-active (open in 1.x; to be fixed in 2.0.0)

**What.** The benchmark derives "fault active" and "switching active" as `overlap_ratio > 0`. The
producer computes `overlap_ratio` on the continuous interval `[window_start, window_start + 0.05 s]`,
which is one sample (1/9600 s) longer than the 480 samples a window holds. When an event onset lies
between two samples, the window whose end falls in that last sample gap contains **no post-onset
sample** (the producer's sample-based `y_fault_present` / `y_switch_present` are 0 there), but the
benchmark scores it as fault or switching active.

**Where.** Every `adapt_grid` and `multi_grid` episode whose onset is off the 9.6 kHz grid (onsets are
sampled continuously). The `benchmark` families are not affected: their onsets lie on the
grid (1.1 s). Measured on the DoubleLine `adapt_grid` windows: 4,667 of the fault-active and 5,332 of the
switch-active windows of 1.x (about 6 % and 9 %); the same rates in the dataset V1 windows.

**Effect.** One window per affected episode, directly before the onset, carries an active label with a
pre-event signal: label noise in the training, validation and in-distribution test partitions of every
task that uses the fault-active or switch-active rule (FD, FC, FL, FG, FP, FCat, FO, ES, ED, SD, ST, EC).
It applies to all baselines alike. Results on the `benchmark` families are unaffected.

**Status.** Kept as is for the 1.x releases (changing it is a label-definition change that would require
re-running nearly every learned cell). Planned for benchmark 2.0.0: active windows will be derived from the
producer's sample-based labels (at least one post-onset sample). Found while validating the dataset V1
windows (2026-11-13).
