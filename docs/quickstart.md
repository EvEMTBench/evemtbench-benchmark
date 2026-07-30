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
export EVEMTBENCH_DATA_ROOT=/path/to/EvEMTBench   # contains benchmark/ adapt_grid/ train/
```

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

Anyone can recompute the official metrics from a predictions file, independent of how it was
produced — this is what makes third-party results comparable.

## Protocol

v0.2 has a single protocol:

| Protocol | Trains on | Tests on | Claim |
|---|---|---|---|
| `held_out` | `adapt_grid` family, episode train split (70%) | `adapt_grid` `test` split (15%, in-distribution) **and** the whole `benchmark` family (never trained on) | in-distribution generalization + held-out benchmark test |

The runner fits once and evaluates **both** test sets, writing one `result.json` per `test_set`.
Splits are committed under `src/evemtbench/splits/v0.2/` and are the source of truth (see
[versioning.md](versioning.md)). On HPC (NHR@FAU Alex), run a narrow slice via the sweep worker
(`GRID=double_line BASELINES=mlp TASKS=fault_detection_local sbatch hpc/sweep.sh`), or fill the
matrix via `hpc/submit.sh` (one GPU job per grid, all baselines); see those scripts. A single
cell also runs anywhere via the runner directly:
`PYTHONPATH=src python -m evemtbench.baselines.runner --task fault_detection_local
--protocol held_out --grid double_line --baseline mlp`.
