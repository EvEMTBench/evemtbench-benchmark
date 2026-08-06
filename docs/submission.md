# EvEMTBench — submission & results format

How to produce, evaluate, and submit results for the challenge. The evaluation layer
(`data`/`tasks`/`splits`/`evaluation`) is usable **without importing `baselines`** — you only need
it to score your own model (see [CONTRIBUTING.md](../CONTRIBUTING.md)). Protocol & tasks:
[protocol.md](protocol.md); metrics: [metrics.md](metrics.md).

## 1. Produce predictions

Load a task, train on `task.train` **only**, predict each held-out view, and emit a predictions
frame per `(task, grid, protocol, test_set, seed)`:

```python
from evemtbench import load_task
from evemtbench.evaluation.predictions_io import predictions_frame
task = load_task("fault_detection_line", protocol="held_out", grid="cigre_mv")
model.fit(task.train)                      # never touch test/benchmark during fit
preds = predictions_frame(task.test.metadata["sample_id"],
                          task.test.metadata["window_idx"],
                          model.predict(task.test))   # -> columns: sample_id, window_idx, prediction
```

**Predictions schema** (`predictions_seed{n}.parquet`): one row per scored window —

| column | meaning |
|---|---|
| `sample_id` | episode id (the split key) |
| `window_idx` | window index within the episode |
| `prediction` | class index (FD/FC) or float distance-% (FL) |

Predictions must cover exactly the task's valid windows for that view (the runner/evaluator align
on `(sample_id, window_idx)`).

## 2. Evaluate

Programmatic: `from evemtbench import evaluate; evaluate(task, preds, test_set="test")`.
Audit a saved predictions file: `evemtbench-evaluate --task ... --grid ... --protocol held_out
--predictions preds.parquet`. The reference runner (`evemtbench-run`) does fit→predict→evaluate over
seeds and writes the artifacts below.

## 3. `result.json` schema (one per task × protocol × grid × baseline × test_set)

Path: `outputs/<benchmark_version>/<task>/<protocol>/<grid>/<baseline>/<test_set>/result.json`.

```json
{
  "benchmark_version": "0.2.0-dev",
  "task": "fault_detection_line", "protocol": "held_out", "grid": "cigre_mv",
  "baseline": "mlp", "test_set": "test",
  "n_test": 12345,
  "seeds": [0, 1, 2, 3, 4],
  "n_params": 116736,
  "compute": {"n_params": 116736, "n_seeds": 5, "fit_seconds_mean": 74.2, "effective_epochs_mean": 21.0, "peak_mem_bytes_max": 812000000},
  "per_seed": [{"seed": 0, "metrics": {"balanced_accuracy": 0.91, "...": 0.0}}],
  "aggregate": {"balanced_accuracy": {"mean": 0.90, "std": 0.01, "ci95": 0.012, "n_defined": 5, "n_seeds": 5}},
  "breakdowns": {
    "event_type": {
      "flt_1phg_shc": {
        "n": 1207, "n_seeds": 5,
        "metrics": {"balanced_accuracy": {"mean": 0.74, "std": 0.002, "ci95": 0.003, "n_defined": 5, "n_seeds": 5}}
      }
    }
  }
}
```

`test_set` ∈ {`test` (in-distribution), `benchmark` (transfer)}; a grid whose benchmark family is
not yet windowed emits only `test`. `aggregate` is mean ± std over the **defined** seeds
(`n_defined`), plus `ci95` — the 95% CI half-width (Student t, ddof=1; `null` below 2 seeds);
see [metrics.md](metrics.md). `n_params` (trainable-parameter count) and `compute`
(per-cell training cost) are recorded for capacity/efficiency transparency — capacity is reported,
not equalized; both are optional for third-party submissions.

`breakdowns` is **seed-aggregated over the same seeds as `aggregate`**, with each entry under
`metrics` shaped exactly like a top-level `aggregate` entry, so a stratum metric and a headline
metric read identically. `n` is the stratum's support (a property of the eval set, so equal across
seeds) and `n_seeds` counts the seeds that reported the group. Strata are `grid`, `event_family`,
`event_type`, `origin` (and `resistance_bin` when aux settings are attached).

> **Changed 2026-08 (v1.0.0 artifacts regenerated).** Before this, `breakdowns` held the values of
> whichever seed ran *last*, sitting next to a 5-seed `aggregate`, with nothing recording that
> mismatch. Headline numbers were never affected. The metrics themselves are unchanged; the field was
> recomputed from the stored per-seed predictions by `evemtbench-reevaluate` — no model was retrained.

## 4. Reproducibility requirements (for a valid submission)

- **Harness self-test:** `evemtbench-evaluate --self-test` must pass in your environment first
  — it is dataless (synthetic tasks through the real `evaluate()`), so it proves the scoring
  harness before you trust any number it emits.
- **Splits:** use the committed `splits/v<SPLITS_VERSION>/held_out/<grid>/` (regenerate bit-for-bit
  with `python -m evemtbench.splits.generator`). No training on `test`/`benchmark`.
- **Seeds:** report ≥3 seeds; submit `mean ± std` + the `ci95` half-width. (The reference
  baselines report **5**; rendered tables show mean ± 95% CI.)
- **Environment:** record yours the way we record ours — as a capture, not an aspiration. Our
  results environment is `env/results-v1.0.0.lock` (and the CI gate is pinned separately in
  `requirements-ci-py3.*.lock`; the two are not the same thing — see
  [determinism.md](determinism.md)). Every run emits `run_record.json` (seeds, git sha, executing
  host/SLURM job, input split-file + window-manifest hashes, lockfile + results-env hashes, the
  captured `numpy`/`pandas`/`scikit-learn`/`scipy`/torch/CUDA versions, and the **no-tuning
  attestation** — fit on train, select on val, held-out sets predict-only) — include it.
- **Per-grid, per-test_set reporting:** never pool metrics across grids; report each
  (grid × test_set) cell. The leaderboard ([metrics.md](metrics.md)) ranks per task section.

## 5. Coverage / leaderboard

`python -m evemtbench.coverage --summary` shows the expected matrix vs. produced `result.json`
(done / pending / failed). `evemtbench-leaderboard --outputs ./outputs --out leaderboard.html`
renders the sortable, filterable board (topology / test-set / model / version).
