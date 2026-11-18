# Quickstart

Evaluate your own model on an official EvEMTBench task in three calls — using only the
`data` / `tasks` / `splits` / `evaluation` layers (never `baselines`).

## Install

```bash
uv venv && uv pip install -e ".[dev]"      # core (evaluation layer, no torch)
# for the deep-learning baselines / runner:
uv pip install -e ".[baselines]"
```

Point the package at the dataset (never hardcoded):

```bash
export EVEMTBENCH_DATA_ROOT=/path/to/EvEMTBench   # contains benchmark/ adapt_grid/ multi_grid/
```

The root must hold the **windowed** partitions ([DATA_LAYOUT.md](DATA_LAYOUT.md)). The public
download currently contains only the raw simulation exports, and the producer windowing pipeline
is not public yet — see the note under *Dataset* in the [README](../README.md#dataset) for how to
build and verify windows against the pinned contract.

## Evaluate a model (library)

```python
from evemtbench import load_task, evaluate

task = load_task("fault_detection_local", protocol="held_out", grid="double_line")
# task.train / task.val / task.test : TaskViews of (waveform, label, metadata).

preds = my_model.predict(task.test)          # -> DataFrame[sample_id, window_idx, prediction]
result = evaluate(task, preds)               # official metrics + stratified breakdowns
print(result.metrics, result.benchmark_version)
```

`evaluate` aligns predictions to the test set by `(sample_id, window_idx)` — never by row
order — and errors on any missing/extra/duplicate window.

## Reproduce a baseline (runner)

```bash
evemtbench-run --task fault_detection_local --protocol held_out --grid double_line \
    --baseline majority --seeds 0 1 2 --out ./outputs
```

This fits once and evaluates both held-out sets, writing under
`outputs/<version>/<task>/<protocol>/<grid>/<baseline>/`: a `run_record.json` (seeds, git sha,
lockfile hash, env) and, per `test_set` (`test` and `benchmark`) subdir,
`predictions_seed{n}.parquet` + `result.json` (per-seed + aggregate `mean/std/ci95`).

## Audit a predictions file (CLI)

```bash
evemtbench-evaluate --task fault_detection_local --protocol held_out --grid double_line \
    --predictions preds.parquet --out result.json
```

Anyone can recompute the official metrics from a predictions file (`--test-set benchmark` for the
benchmark family), independent of how it was produced — this is what makes third-party results comparable.

## Protocols

Four protocols: `held_out`, `multi_grid_pretrain`, `transfer_zeroshot`, `transfer_finetune`.
What each trains and tests on: [protocol.md](protocol.md#protocols).
