# 2. Pre-training and transfer

Covers **Fig. 3** and **SI Tables S6–S9**: zero-shot and fine-tuned transfer from the `multi_grid`
template grids to the four reference grids, for FD, FC and FL on the `local` and `line` views,
deep baselines only.

## Numbers

| Paper | File in `results/1.3.0/tables/` |
|---|---|
| Table S6 (FD) | `tab_transfer_fd.tex` |
| Table S7 (FC) | `tab_transfer_fc.tex` |
| Table S8 (FL) | `tab_transfer_fl.tex` |
| Table S9 (target-only training) | `supp_tab_transfer_scratch.tex` |

In `aggregated_results.csv`, filter `protocol` in `multi_grid_pretrain`, `transfer_zeroshot`,
`transfer_finetune`; `source` is the pre-training grid.

## Train

1. Pre-train on one template grid (`template_20kv`, `template_110kv` or `template_345kv`):

   ```bash
   evemtbench-run --task fault_detection_line --protocol multi_grid_pretrain \
       --grid template_20kv --baseline cnn --seeds 0 1 2 3 4 --epochs 60 --out outputs
   ```

2. Transfer to a reference grid, zero-shot or fine-tuned:

   ```bash
   evemtbench-run --task fault_detection_line --protocol transfer_finetune \
       --grid double_line --pretrain-grid template_20kv \
       --baseline cnn --seeds 0 1 2 3 4 --epochs 60 --out outputs
   ```

   Use `--protocol transfer_zeroshot` for the zero-shot cells. Both read the checkpoint of step 1
   from `outputs/`.

Baselines: `mlp`, `gru`, `cnn`, `resnet`, `tst`. These protocols normalize each window
(`--normalize auto` resolves to `per_window`). Table S9 compares with the `held_out` cells of
guide 1.

## Rebuild

```bash
PYTHONPATH=src python hpc/paper_artifacts/build_tab_transfer_fd.py        # Table S6; likewise fc, fl
PYTHONPATH=src python hpc/paper_artifacts/build_supp_tab_transfer_scratch.py
PYTHONPATH=src python hpc/paper_artifacts/build_fig_transfer_overview.py --abl control  # Fig. 3
```

Fig. 3 also draws the matched target-only reference (dashed, with its 95 % interval): arm C of the
input-normalization control. `--abl` names that control tree, the directory containing its
`outputs/` ([guide 3](3_normalization_control.md)).
