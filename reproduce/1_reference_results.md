# 1. Reference results (`held_out`)

Covers **Fig. 2** and **SI Tables S1–S5** and **Figs. S1–S5** (FD, FC, FL, ED, EC).

## Numbers

| Paper | File in `results/1.3.0/tables/` |
|---|---|
| Table S1 (FD) | `tab_res_fd.tex` |
| Table S2 (FC) | `tab_res_fc.tex` |
| Table S3 (FL) | `tab_res_fl.tex` |
| Table S4 (ED) | `tab_res_ed.tex` |
| Table S5 (EC) | `tab_res_ec.tex` |

Every baseline, not only the selected one: `supp_tab_<task>_{test,benchmark}.tex`.
In `aggregated_results.csv`, filter `protocol == held_out`.

## Train

One cell = one task, grid and baseline, five seeds, both test sets (`test`, `benchmark`):

```bash
evemtbench-run --task fault_detection_local --protocol held_out --grid double_line \
    --baseline cnn --seeds 0 1 2 3 4 --epochs 60 --out outputs
```

- Tasks: the 24 task ids ([task table](../README.md#tasks)).
- Grids: `double_line`, `cigre_mv`, `testgrid_110kv`, `ieee39`.
- Baselines: `majority`, `threshold`, `impedance`, `random_forest`, `mlp`, `gru`, `cnn`,
  `resnet`, `tst`.
- Training settings of every baseline: SI Table S22.

The headline tables report, per task and grid, the baseline selected on the validation split
(SI Table S24); `test` and `benchmark` are never used for selection. To run the full matrix on
SLURM, see `hpc/submit.sh`.

## Rebuild

```bash
PYTHONPATH=src python hpc/paper_artifacts/build_tab_res_fd.py      # Table S1; likewise fc, fl, ed, ec
PYTHONPATH=src python hpc/paper_artifacts/build_fig_supp_fd.py     # Fig. S1; likewise fc, fl, ed, ec
PYTHONPATH=src python hpc/paper_artifacts/build_fig_synthesis.py   # Fig. 2
```

The scripts read `outputs/` and write tables to `paper/manuscript/generated/` and figures to
`paper/manuscript/figures/` (created on first run). Compare a rebuilt table with its saved copy
in `results/1.3.0/tables/`. `make paper-artifacts` builds every table (the dataset-backed ones
need the data). Figures need matplotlib.
