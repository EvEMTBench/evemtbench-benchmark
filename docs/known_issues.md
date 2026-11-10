# Known issues (benchmark contract v1.1.0)

Dataset and scoring properties that affect published numbers. Each sits in an artifact that is
frozen within the major version (the valid-sample rules, the line-view oracle and the metric
definitions, see [versioning.md](versioning.md)),
so v1.x keeps the published behaviour and only flags it. A behavioural fix belongs to the next
major contract version, and each is documented here so a method author can account for them.
Numbers below were measured on the released windows on 2026-10-01.

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
should treat these lines as one-ended rather than use the duplicated far terminal. The same
padding applies to a bus fault at a bus with a single line bay (`MainBus*` pairs); that case is a
labelling convention for bus faults and is not listed in `one_ended_lines`.

## 2. Bus-target fault episodes inside the fault-location target

**What.** The fault-location (FL) valid samples are the fault-active windows with a non-null
`y_fault_location` (`src/evemtbench/tasks/valid_samples.py`). The intent is line faults only. But in
the `adapt_grid` family the producer also fills a location for **bus-target** fault episodes
(`y_fault_line` = `MainBus*`): one value per episode, spread over 1–99 %, describing no position on
any line. Those windows pass the gate.

**Scope.** Bus-target windows make up 38.9–46.6 % of the FL samples in `adapt_grid`:

| Grid | Bus-target share of FL samples (`adapt_grid`) |
|---|---|
| CIGRE MV | 46.6 % |
| Double Line | 43.5 % |
| TestGrid110kV | 38.9 % |
| IEEE-39 | 43.9 % |

In the `benchmark` family bus-target locations are NaN, and the `multi_grid` pre-training corpora
have no bus-target windows in the gate, so **no** bus-target window is scored on `benchmark`. The
effect is on FL training and validation and on the in-distribution `test` scores, for `held_out`
and the transfer protocols that fine-tune or test on a reference grid's `adapt_grid`.

**What the code does.** The official gate is unchanged. For diagnostics,
`evemtbench.tasks.valid_samples.line_target_fault_location_mask(y)` returns the line-targets-only
subset, so FL can be trained or scored on line faults alone outside the official leaderboard.

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

**How to read it.** Treat binary `macro_f1` in a single-class breakdown row as undefined; use
`missed_fault_rate` / `false_alarm_rate` there, which are defined on one class.
