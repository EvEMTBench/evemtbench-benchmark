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
| `impedance` | conventional (impedance-based fault locator) | CPU | FL, **`local` + `line` only**, `held_out` |
| `random_forest` | classical ML | CPU | all tasks |
| `mlp` | deep (flattened window) | GPU | all tasks |
| `gru` | deep (sequence) | GPU | all tasks |
| `cnn` | deep (1-D waveform) | GPU | all tasks |
| `resnet` | deep (1-D residual CNN) | GPU | all tasks |
| `tst` | deep (patch Transformer) | GPU | all tasks |
| `gnn` | deep (topology-aware) | GPU | **planned** (cross-topology) |

## Cards

**`majority`** — predicts the most-frequent training class (classification) or the training mean
(regression). The trivial floor: any useful model must beat it; on balanced_accuracy it sits at
~1/num_classes.

**`threshold`** — a single-end **overcurrent** rule (largest per-phase RMS current in the window > learned threshold). Restricted
to **binary detection at `local`** observability — the information set of a classical overcurrent
relay. It is the *conventional-protection anchor*: where it collapses on `benchmark` transfer, that
is a diagnostic of the operating-point shift, not merely a weak model.

**`impedance`** — a conventional **impedance-based fault locator**, the physical reference for the
fault-location tasks ([`baselines/impedance.py`](../src/evemtbench/baselines/impedance.py), signal
layer in `baselines/_phasor.py`). It is deterministic and not learned, and it is reported next to
the learned baselines, never selected among them (`coverage.CONVENTIONAL_BASELINES`).

- *Phasors*: full-cycle DFT at the grid's nominal frequency (grid registry, cross-checked against
  the topology graph) on the **last complete cycle** of each window.
- *`local` view, one-terminal reactance method*: a current-magnitude phase selector picks the fault
  loop (residual current for ground involvement, phase currents above half the largest for the
  involved phases); ground loops use zero-sequence compensation `k0 = (Z0 − Z1) / (3 Z1)`;
  location = `Im(Z_loop) / (x1 · length)`.
- *`line` view, two-terminal method*: positive-sequence phasors of both terminals,
  `m = (V_S − V_R + Z_L I_R) / (Z_L (I_S + I_R))`, both currents counted into the line. No loop
  selection, and no dependence on the fault resistance for a series-impedance line.
- *`global` view*: not applicable.
- *Inputs beyond the waveforms*: the line's series impedance and length from the grid topology
  graph (aux overlay, `adapt_grid` family; assumed ohm/km and km), looked up by `y_fault_line`.
  That is the oracle the `local`/`line` views are already centred on. No other label column is
  read at prediction time.
- *Calibration*: none by default. The channels carry primary-level values (checked on every run:
  the phase-voltage level against the grid's nominal voltage, recorded in the run record), so the
  secondary-to-primary impedance scale is `1`. `impedance_scale` is an explicit parameter for data
  with true secondary quantities: a known ratio, or `"fit"` (one factor per grid or per line,
  least trimmed squares on `train` only). Whatever is used is written to
  `run_record.json["calibration"]`.
- *Reported, not guessed*: a line without usable parameters stops the run with the reason. A line
  with one measured terminal ([known_issues.md](known_issues.md) §1) falls back to the
  one-terminal method in the `line` view and is mirrored to the label's reference end; both are
  listed in the run record.
- *Known limits* (by construction, not tuned away): a window whose last cycle is not entirely in
  the fault state (inception or clearing inside it), faults without a sustained fundamental
  component (incipient faults), fault resistance with infeed from the far end (one-terminal
  method), shunt capacitance and mutual coupling (not modelled).

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

**`tst`** — the Time Series Transformer encoder (Zerveas et al., KDD 2021, arXiv:2010.02803),
trained from scratch as the **transformer reference**: patch embedding `Conv1d(C, 128, kernel 16,
stride 8)` over the `(C, 480)` window → 59 tokens, scaled by `√128`; learnable positional encoding;
3 post-norm encoder blocks (`d_model=128`, 8 heads, feed-forward 256, GELU, dropout 0.1) with
**batch normalisation in place of layer normalisation**, as in the source; a linear head on the
concatenation of all token outputs. Only the embedding depends on the channel count, so the encoder
cost is the same on the 6-channel `local` view and the wide `global` view. The configuration is
fixed a priori (the source's generic width, depth and dropout), not tuned. **Three deviations from
the published model:** (1) the strided patch embedding — the source makes one token per time step,
which is 480 tokens per window; patching follows Nie et al. (ICLR 2023, arXiv:2211.14730), but
channel-mixing rather than channel-independent; (2) **8 attention heads** instead of the generic
configuration's 16; (3) a **LayerNorm on the patch tokens** after the scaled projection — without
it training diverged to NaN on the wide `global` views and stopped early on all views. It also keeps the benchmark's shared optimiser and batch size (AdamW, cosine
schedule, batch 256) rather than the RAdam and batch 128 of the source repository's examples, so
that only the architecture differs from the other deep baselines; no masked pre-training is used.

**`gnn`** *(planned)* — message-passing over the grid graph (`graph.pickle`), the natural
contender for the **`cross_topology`** protocol (train on diverse topologies → test on unseen
grids), where fixed-input models (MLP/CNN) cannot ingest a different channel count. Pending the
`cross_topology` protocol and a variable-channel (cubicle-set) view
([design/cross_topology.md](design/cross_topology.md)).

## Hyperparameters & determinism

Defaults: `--epochs 60 --batch-size 256 --lr 1e-3`; the runner's own `--seeds` default is `0` (a
single seed), while the official sweep (`hpc/sweep.sh`, `SEEDS`) runs seeds `0 1 2 3 4`. The five
deep baselines (mlp/gru/cnn/resnet/tst) share **one** training regime via `baselines/_train.py` —
AdamW + weight decay `1e-4`, `CosineAnnealingLR`, and **validation early-stopping** (patience 12, best-val checkpoint,
restored before predict) — so only the architecture differs (a cross-baseline fairness requirement).
GPU training is **not** bit-identical run-to-run (reported as mean ± std over 5 seeds); splits,
labels, and metrics are deterministic (see [determinism.md](determinism.md)).

## Model capacity & fairness

The deep baselines are compared at **idiomatic reference sizes**, **not** parameter-matched: their
natural capacities differ by orders of magnitude — the flatten-`mlp` is `~1.6 M` params on `local`
tasks and `~43 M` on the wide `global` view (the first Linear is `480·C·512`), while `gru`/`cnn` are
`~50–125 k`, `resnet` is `~0.5–0.6 M`, and `tst` is `~0.43 M` on `local` (its patch embedding adds
`16·C·128` weights per input channel `C`). "Fair" here means a **shared protocol, training regime, and
budget** — not equal capacity; forcing parity would distort each architecture away from its reference
form. Capacity is instead made **transparent**: every deep cell records its trainable-parameter count
(`n_params` in `result.json`) and per-seed training cost (`compute`: wall-time, effective epochs, peak
GPU mem), surfaced as `params`/`train` columns in RESULTS and the leaderboard.

## Checkpoints

The full sweep saves each deep baseline's **best-val checkpoint per seed** to
`outputs/.../checkpoints/seedN.pt` (sha-recorded in `run_record.json`; release artifacts, not
committed). Noise-robustness and detection-latency analyses can run on them post hoc, without
re-training.

## Adding a baseline

See [extending.md](extending.md#add-a-baseline). Third-party submissions need **not** add a
baseline — score directly via the evaluation layer ([submission.md](submission.md)).
