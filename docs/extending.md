# EvEMTBench — extending the benchmark

How to add grids, tasks, baselines and protocols. Public-facing usage is in
[submission.md](submission.md).

## Add a grid

1. Register a `GridSpec` in [`data/roots.py`](../src/evemtbench/data/roots.py) `GRIDS`
   (key, topology id, raw folder, voltage, frequency, level).
2. Generate its split: `python -m evemtbench.splits.generator --grids <key>` — writes
   `splits/v<ver>/held_out/<key>/`. If the `benchmark` family isn't windowed yet, the generator
   writes an **empty** benchmark partition (in-distribution `test` only) and the runner/coverage
   handle it; re-run the generator when the benchmark lands.
3. **Nothing else** — the observability oracle derives automatically from that grid's
   `feature_names` ([`tasks/line_view.py`](../src/evemtbench/tasks/line_view.py) `build_oracle`:
   keep `Bus_<bus>_Line_<id>` cubicles, pair terminals sharing a `lineid`). Verify with a quick
   `build_oracle(key, feature_names)` check (line count, all `y_fault_line` labels resolve).

## Add a task

`taxonomy.py` (new `Function`) → `tasks/labels/<x>.py` (pure deriver + frozen class vocab) →
`valid_samples.py` (admissible-window mask) → `task.py::_LABEL_DERIVERS` → `registry.py` (`_make`)
→ `evaluation/evaluate.py::_CLASS_VOCAB` (multiclass). Keep a window's label `-1` only for
windows the valid mask excludes; never feed `-1` to a CrossEntropy head (a fault type outside the
vocab must be excluded by `valid_samples`).

## Add a baseline

Implement `fit(task)` / `predict(view) -> predictions_frame` (mirror `baselines/cnn.py`); register
its factory in `runner.py::BASELINE_FACTORIES` (in-tree) or via `runner.register_baseline()`
(third-party, before `runner.main()`; see `examples/byo_model.py`) — `_make_baseline` and the
`--baseline` choices both read that registry, so nothing else in the runner changes — and add it to
`coverage.py::ALL_BASELINES` (+ `baseline_applies` if it's restricted, like `threshold` and
`impedance`). A baseline that calibrates anything on `train` exposes it as a JSON-serialisable
`calibration_` attribute after `fit`; the runner copies it to `run_record.json["calibration"]`.
The builders that enumerate baselines are `hpc/make_results.py::BASELINES` (RESULTS.md),
`hpc/audit_manifest.py::BASELINE_ABOUT`/`NA_REASONS` (audit site) and
`hpc/paper_artifacts/paper_lib.py` (manuscript tables; labels, n/a rule and the table set). See
[baselines.md](baselines.md).

## Add a protocol

`splits/protocols.py` (`Protocol` entry) → `task.py::_protocol_triples` (which `(family, grid)` each
split draws from) → the task's `protocols` tuple in `registry.py` → `coverage.py`
(`protocol_grid_applies` / `baseline_applies` / `expected_test_sets` for grid & baseline scope). If
the protocol needs committed split files, add it to `generator.py::_GENERATED_PROTOCOLS` and commit
`splits/v<ver>/<protocol>/<grid>/`. The pre-training/transfer protocols
([design/pretrain_transfer.md](design/pretrain_transfer.md)) are the worked example: the `multi_grid`
double-line view (`tasks/line_view.py`), per-window normalization (`task.py::build_input_transform`),
and checkpoint transfer (`baselines/_transfer.py`). Adding a protocol is a **minor** bump (additive;
existing `held_out` results stay comparable).

How the reference sweep runs on the cluster is documented internally, in the development repository.
