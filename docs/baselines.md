# EvEMTBench — reference baselines (model cards)

The benchmark ships reference baselines spanning trivial floors → conventional protection →
classical ML → deep learning. They are **demos**, import-isolated from the evaluation layer, and
exist to (a) anchor the leaderboard and (b) give the challenge a bar to beat. All learned baselines
**branch on the task label kind** — CrossEntropy (+ inverse-frequency class weights) for
classification, MSE (+ target standardization) for regression — and fit per-channel standardization
**on `task.train` only** (no leakage), applied **per batch** on-device (memory-lean for the wide
`global` view). Source: [`src/evemtbench/baselines/`](../src/evemtbench/baselines/). Run via
`evemtbench-run --baseline <name> ...`.

| baseline | class | device | applies to |
|---|---|---|---|
| `majority` | trivial floor | CPU | all tasks |
| `threshold` | conventional (overcurrent) | CPU | FD, **binary + `local` only** |
| `random_forest` | classical ML | CPU | all tasks |
| `mlp` | deep (flattened window) | GPU | all tasks |
| `gru` | deep (sequence) | GPU | all tasks |
| `cnn` | deep (1-D waveform) | GPU | all tasks |
| `resnet` | deep (1-D residual CNN) | GPU | all tasks |
| `gnn` | deep (topology-aware) | GPU | **planned** (cross-topology) |

## Cards

**`majority`** — predicts the most-frequent training class (classification) or the training mean
(regression). The trivial floor: any useful model must beat it; on balanced_accuracy it sits at
~1/num_classes.

**`threshold`** — a single-end **overcurrent** rule (peak |current| > learned threshold). Restricted
to **binary detection at `local`** observability — the information set of a classical overcurrent
relay. It is the *conventional-protection anchor*: where it collapses on `benchmark` transfer, that
is a diagnostic of the operating-point shift, not merely a weak model.

**`random_forest`** — extracts per-channel **summary features** over the time axis (mean, std, min,
max, peak-to-peak, RMS, peak → c×7 features) and fits a class-balanced `RandomForest{Classifier,
Regressor}`. Tree-based ⇒ scale-invariant, sklearn-only, runs anywhere (CPU). The classical-ML
reference.

**`mlp`** — flattens the window `(l, c)` to a vector → MLP (`hidden=(512,256,128)`, BatchNorm,
dropout 0.3), AdamW + weight decay, cosine LR, **validation-split early stopping** (best-val
checkpoint). Deep-learning reference on hand-free features.

**`gru`** — consumes the window as a time series `(l, c)`; 1-layer GRU, final-timestep hidden → head
(`hidden=128`). Sequence model.

**`cnn`** — 1-D dilated convolutions over the 480-sample window (3 blocks, width 64, growing
receptive field) → global average pool → head. The de-facto standard for point-on-wave protection
ML.

**`resnet`** — the canonical time-series ResNet (Wang et al. 2017): three residual blocks
(filters `[64, 128, 128]`, each three BN+ReLU conv layers with kernels 8/5/3 and a 1×1 shortcut) →
global average pool → head. A **modern, stronger learned reference** than the plain `cnn` (~0.5–0.6 M
params). Shares the identical training regime as mlp/gru/cnn — only the architecture differs.

**`gnn`** *(planned)* — message-passing over the grid graph (`grid_graph.pickle`), the natural
contender for the **`cross_topology`** protocol (train on diverse topologies → test on unseen
grids), where fixed-input models (MLP/CNN) cannot ingest a different channel count. Pending the
`train` family + graph artifacts.

## Hyperparameters & determinism

Defaults: `--epochs 60 --batch-size 256 --lr 1e-3 --seeds 0 1 2 3 4`. The four deep baselines
(mlp/gru/cnn/resnet) share **one** training regime via `baselines/_train.py` — AdamW + weight decay
`1e-4`, `CosineAnnealingLR`, and **validation early-stopping** (patience 12, best-val checkpoint,
restored before predict) — so only the architecture differs (a cross-baseline fairness requirement).
GPU training is **not** bit-identical run-to-run (reported as mean ± std over 5 seeds); splits,
labels, and metrics are deterministic (see [determinism.md](determinism.md)). Deep baselines require
an sm_70+ GPU (v100 / Quadro RTX / A6000); sm_61 (GTX/TitanXP) nodes are excluded.

## Model capacity & fairness

The deep baselines are compared at **idiomatic reference sizes**, **not** parameter-matched: their
natural capacities differ by orders of magnitude — the flatten-`mlp` is `~1.6 M` params on `local`
tasks and `~43 M` on the wide `global` view (the first Linear is `480·C·512`), while `gru`/`cnn` are
`~50–125 k` and `resnet` is `~0.5–0.6 M`. "Fair" here means a **shared protocol, training regime, and
budget** — not equal capacity; forcing parity would distort each architecture away from its reference
form. Capacity is instead made **transparent**: every deep cell records its trainable-parameter count
(`n_params` in `result.json`) and per-seed training cost (`compute`: wall-time, effective epochs, peak
GPU mem), surfaced as `params`/`train` columns in RESULTS and the leaderboard.

## Checkpoints, roadmap & limitations

The full sweep saves each deep baseline's **best-val checkpoint per seed** to
`outputs/.../checkpoints/seedN.pt` (sha-recorded in `run_record.json`; `outputs/` is git-ignored, so
these are release artifacts, not committed). This is deliberate insurance: several planned analyses
become **post-hoc, no re-training required**:

- **Noise / measurement-realism robustness** — evaluate the saved checkpoints on perturbed inputs
  (sensor noise, quantization, CT/PT saturation); the sim-to-real gap is a known limitation addressed
  this way rather than in-sweep.
- **Detection latency (time-to-trip)** — derivable from the stored per-window predictions
  (`predictions_seedN.parquet`, keyed by `sample_id, window_idx`): the FD valid set includes the
  **pre-fault** windows, so first-correct-flag-after-inception is a pure post-processing metric.
- **Dataset artifacts** (datasheet, license, DOI/Zenodo hosting, Croissant metadata) are tracked
  separately from this sweep for a NeurIPS D&B-style submission.

## Adding a baseline

Implement the `fit(task)` / `predict(view) -> predictions_frame` interface (mirror `cnn.py`),
register it in `baselines/runner.py::_make_baseline` and the `--baseline` choices + `coverage.py`
`ALL_BASELINES`. See [extending.md](extending.md). Third-party submissions need **not** add a
baseline — score directly via the evaluation layer ([submission.md](submission.md)).
