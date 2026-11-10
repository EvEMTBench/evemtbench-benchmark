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

## Protocol

There are four protocols (`src/evemtbench/splits/protocols.py`):

| Protocol | Trains on | Tests on | Claim |
|---|---|---|---|
| `held_out` | reference grid's `adapt_grid` family, episode train split (70%) | `adapt_grid` `test` split (15%, in-distribution) **and** the whole `benchmark` family (never trained on) | in-distribution generalization + held-out benchmark test |
| `multi_grid_pretrain` | `multi_grid` pre-training corpus (Template20kV / 110kV / 345kV), train split | the corpus' own `test` split (in-corpus) | in-corpus pre-training; produces the transfer checkpoints |
| `transfer_zeroshot` | nothing on the target (initialised from a `multi_grid_pretrain` checkpoint) | reference grid's `test` and `benchmark` | zero-shot cross-grid generalization |
| `transfer_finetune` | pre-trained checkpoint, fine-tuned on the reference grid's `adapt_grid` train split | reference grid's `test` and `benchmark` | pre-train, then adapt to the target |

`held_out` covers all 24 tasks and every applicable baseline; the three pre-training/transfer protocols run
only on the portable `local`/`line` task variants and only with the deep baselines
(`mlp`, `gru`, `cnn`, `resnet`), and each transfer cell is scored once per pre-training source.
The runner fits once and evaluates every applicable test set, writing one `result.json` per
`test_set`. Splits are committed under `src/evemtbench/splits/` and are the source of truth (see
[versioning.md](versioning.md)): `splits/v1.0/held_out/` for `held_out` (the transfer protocols
reuse the target grid's held_out split) and the topology-grouped `splits/v1.1/multi_grid_pretrain/`
for pre-training. On HPC (NHR@FAU Alex), run a narrow slice via the sweep worker
(`GRID=double_line BASELINES=mlp TASKS=fault_detection_local sbatch hpc/sweep.sh`), or fill the
matrix via `hpc/submit.sh` (one GPU job per grid, all baselines); see those scripts. A single
cell also runs anywhere via the runner directly:
`PYTHONPATH=src python -m evemtbench.baselines.runner --task fault_detection_local
--protocol held_out --grid double_line --baseline mlp`.
