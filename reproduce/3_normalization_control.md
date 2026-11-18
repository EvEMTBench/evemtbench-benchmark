# 3. Input-normalization control

Covers **SI Table S10**: does pre-training help, or only the per-window normalization the transfer
protocols use? Three arms for FD, FC and FL on the `local` and `line` views:

| Arm | Run |
|---|---|
| A | `held_out`, no normalization (the reference results, guide 1) |
| B | `transfer_finetune` (guide 2); the pre-training grid is chosen on validation |
| C | `held_out` with per-window normalization, in a separate output tree |

Each arm reports the deep baseline (`mlp`, `gru`, `cnn`, `resnet`, `tst`) that is best on
validation.

## Numbers

Table S10 is not in the results bundle: arm C is a control, not a benchmark result.

## Train

Arm C, per task, view, grid and deep baseline, into its own tree:

```bash
evemtbench-run --task fault_detection_line --protocol held_out --grid double_line \
    --baseline cnn --normalize per_window --seeds 0 1 2 3 4 --epochs 60 \
    --out control/outputs
```

## Rebuild

```bash
make norm-control ABL=control       # reads outputs/ (arms A, B) and control/outputs/ (arm C)
```
