# EvEMTBench v1.1 — results (1.1.0)

The `held_out` results below were produced under contract v1.0.0 and remain current under
v1.1.0: the v1.1 revision regenerated only the `multi_grid` source-corpus splits (pre-training /
transfer protocols — see the leaderboard and `results/1.1.0/`), leaving every `held_out` artifact
byte-identical.

Protocol `held_out`: train/val/test on the `adapt_grid` family; **benchmark** = the held-out
`benchmark` family (never trained on). 5 seeds — cells show **mean ± 95% CI half-width** (Student
t across seeds; single-seed cells show the mean alone). Deep baselines trained to a 60-epoch
cap with validation early-stopping. Baselines: majority, threshold (FD-local only), random_forest,
mlp, gru, cnn, resnet (1D-ResNet). Deep baselines share one training regime (only the architecture
differs); `params` = trainable-parameter count (capacity is reported, **not** equalized). Grids are
reported **per grid** (never pooled — class spaces differ, e.g. HIF is MV-only).

Tasks are grouped by **difficulty tier** (see `docs/task_suite_roadmap.md`):
- **Tier 1 — easy anchors:** near-solved on ideal data; the community frontier is added realism.
- **Tier 2 — medium:** learnable in-distribution, harder on MV / under transfer.
- **Tier 3 — hard frontier:** hard even on ideal data; the community frontier is better models.

> **Status**: tables are accumulated as jobs complete; `—` marks a pending cell. This file is
> auto-generated from `outputs/` by `hpc/make_results.py` — do not hand-edit.

> **Known issues**: dataset properties that affect some of these numbers are listed, with scope,
> in [`docs/known_issues.md`](docs/known_issues.md) (flagged; the published behaviour is unchanged).


## Results by grid


### DoubleLine (110 kV HV)


#### Tier 1 — easy anchors


**ED — event detection** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| global | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.405 ± 0.000 | — | 0s |
| global | random_forest | 0.804 ± 0.000 | 0.947 ± 0.004 | 0.923 ± 0.006 | — | 1.8m |
| global | mlp | 0.786 ± 0.001 | 0.984 ± 0.000 | 0.975 ± 0.000 | 12.0M | 8.9m |
| global | gru | 0.911 ± 0.012 | 0.990 ± 0.003 | 0.990 ± 0.003 | 69k | 8.9m |
| global | cnn | 0.797 ± 0.007 | 0.983 ± 0.000 | 0.975 ± 0.000 | 68k | 9.0m |
| global | resnet | 0.866 ± 0.033 | 0.952 ± 0.111 | 0.956 ± 0.093 | 530k | 24.0m |

**FD — fault detection** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| local | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.407 ± 0.000 | — | 0s |
| local | threshold | 0.729 ± 0.000 | 0.916 ± 0.000 | 0.877 ± 0.000 | — | 2s |
| local | random_forest | 0.881 ± 0.001 | 0.499 ± 0.000 | 0.406 ± 0.000 | — | 16s |
| local | mlp | 0.850 ± 0.001 | 0.981 ± 0.000 | 0.970 ± 0.000 | 1.6M | 1.1m |
| local | gru | 0.864 ± 0.017 | 0.984 ± 0.004 | 0.975 ± 0.007 | 52k | 2.2m |
| local | cnn | 0.872 ± 0.014 | 0.987 ± 0.001 | 0.979 ± 0.002 | 49k | 2.0m |
| local | resnet | 0.883 ± 0.030 | 0.989 ± 0.008 | 0.984 ± 0.015 | 506k | 7.5m |
| line | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.407 ± 0.000 | — | 0s |
| line | random_forest | 0.884 ± 0.001 | 0.500 ± 0.000 | 0.407 ± 0.000 | — | 21s |
| line | mlp | 0.854 ± 0.001 | 0.987 ± 0.000 | 0.980 ± 0.001 | 3.1M | 1.4m |
| line | gru | 0.870 ± 0.004 | 0.955 ± 0.089 | 0.955 ± 0.071 | 55k | 3.0m |
| line | cnn | 0.877 ± 0.011 | 0.987 ± 0.001 | 0.980 ± 0.002 | 51k | 2.4m |
| line | resnet | 0.906 ± 0.051 | 0.991 ± 0.008 | 0.987 ± 0.014 | 509k | 8.6m |
| global | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.407 ± 0.000 | — | 0s |
| global | random_forest | 0.890 ± 0.001 | 0.500 ± 0.000 | 0.407 ± 0.000 | — | 40s |
| global | mlp | 0.849 ± 0.001 | 0.986 ± 0.000 | 0.978 ± 0.001 | 12.0M | 4.4m |
| global | gru | 0.858 ± 0.021 | 0.943 ± 0.114 | 0.941 ± 0.093 | 69k | 2.6m |
| global | cnn | 0.867 ± 0.017 | 0.987 ± 0.001 | 0.979 ± 0.001 | 68k | 3.8m |
| global | resnet | 0.863 ± 0.016 | 0.987 ± 0.001 | 0.980 ± 0.001 | 530k | 8.2m |

**grounded vs ungrounded** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| local | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.379 ± 0.000 | — | 0s |
| local | random_forest | 0.922 ± 0.002 | 0.678 ± 0.005 | 0.678 ± 0.007 | — | 9s |
| local | mlp | 0.968 ± 0.000 | 0.991 ± 0.001 | 0.991 ± 0.000 | 1.6M | 1.2m |
| local | gru | 0.967 ± 0.003 | 0.992 ± 0.004 | 0.993 ± 0.004 | 52k | 1.8m |
| local | cnn | 0.970 ± 0.001 | 0.998 ± 0.002 | 0.998 ± 0.003 | 49k | 2.1m |
| local | resnet | 0.970 ± 0.002 | 0.997 ± 0.004 | 0.997 ± 0.004 | 506k | 5.0m |
| line | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.379 ± 0.000 | — | 0s |
| line | random_forest | 0.940 ± 0.001 | 0.686 ± 0.009 | 0.687 ± 0.011 | — | 11s |
| line | mlp | 0.970 ± 0.000 | 0.996 ± 0.000 | 0.995 ± 0.000 | 3.1M | 1.4m |
| line | gru | 0.970 ± 0.001 | 0.998 ± 0.000 | 0.997 ± 0.000 | 55k | 1.7m |
| line | cnn | 0.970 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 51k | 2.1m |
| line | resnet | 0.971 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 509k | 7.5m |
| global | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.379 ± 0.000 | — | 0s |
| global | random_forest | 0.955 ± 0.000 | 0.653 ± 0.007 | 0.644 ± 0.010 | — | 21s |
| global | mlp | 0.969 ± 0.000 | 0.998 ± 0.001 | 0.998 ± 0.001 | 12.0M | 3.3m |
| global | gru | 0.970 ± 0.000 | 0.999 ± 0.001 | 0.998 ± 0.001 | 69k | 2.5m |
| global | cnn | 0.971 ± 0.000 | 1.000 ± 0.001 | 1.000 ± 0.001 | 68k | 3.2m |
| global | resnet | 0.970 ± 0.000 | 0.999 ± 0.001 | 0.999 ± 0.001 | 530k | 6.1m |

**SD — switch detection** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| global | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.498 ± 0.000 | — | 0s |
| global | random_forest | 0.721 ± 0.001 | 0.630 ± 0.016 | 0.559 ± 0.042 | — | 1.5m |
| global | mlp | 0.701 ± 0.000 | 0.730 ± 0.002 | 0.804 ± 0.004 | 12.0M | 10.3m |
| global | gru | 0.769 ± 0.029 | 0.826 ± 0.032 | 0.795 ± 0.144 | 69k | 9.3m |
| global | cnn | 0.729 ± 0.025 | 0.816 ± 0.017 | 0.624 ± 0.235 | 68k | 10.6m |
| global | resnet | 0.780 ± 0.004 | 0.808 ± 0.017 | 0.712 ± 0.265 | 530k | 26.4m |

#### Tier 2 — medium


**ES — event state (3-class)** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| global | majority | 0.333 ± 0.000 | 0.333 ± 0.000 | 0.162 ± 0.000 | — | 0s |
| global | random_forest | 0.723 ± 0.001 | 0.703 ± 0.008 | 0.641 ± 0.004 | — | 1.9m |
| global | mlp | 0.702 ± 0.001 | 0.808 ± 0.000 | 0.856 ± 0.002 | 12.0M | 8.8m |
| global | gru | 0.743 ± 0.023 | 0.833 ± 0.024 | 0.824 ± 0.127 | 69k | 8.9m |
| global | cnn | 0.695 ± 0.004 | 0.798 ± 0.009 | 0.830 ± 0.011 | 68k | 8.9m |
| global | resnet | 0.777 ± 0.011 | 0.870 ± 0.016 | 0.895 ± 0.006 | 530k | 19.8m |

**fault category (shc/incipient/hif)** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| local | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.329 ± 0.000 | — | 0s |
| local | random_forest | 0.807 ± 0.005 | 0.606 ± 0.192 | 0.382 ± 0.091 | — | 11s |
| local | mlp | 0.951 ± 0.001 | 0.985 ± 0.001 | 0.536 ± 0.004 | 1.6M | 52s |
| local | gru | 0.959 ± 0.002 | 0.992 ± 0.003 | 0.586 ± 0.022 | 53k | 1.7m |
| local | cnn | 0.963 ± 0.005 | 0.994 ± 0.005 | 0.607 ± 0.043 | 49k | 1.7m |
| local | resnet | 0.966 ± 0.002 | 0.998 ± 0.001 | 0.647 ± 0.011 | 506k | 7.7m |
| line | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.329 ± 0.000 | — | 0s |
| line | random_forest | 0.827 ± 0.007 | 0.683 ± 0.094 | 0.459 ± 0.054 | — | 14s |
| line | mlp | 0.964 ± 0.000 | 0.998 ± 0.001 | 0.640 ± 0.006 | 3.1M | 1.3m |
| line | gru | 0.967 ± 0.001 | 1.000 ± 0.000 | 0.661 ± 0.005 | 55k | 1.4m |
| line | cnn | 0.964 ± 0.003 | 0.999 ± 0.001 | 0.650 ± 0.007 | 52k | 1.5m |
| line | resnet | 0.965 ± 0.004 | 0.999 ± 0.001 | 0.651 ± 0.018 | 509k | 5.1m |
| global | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.329 ± 0.000 | — | 0s |
| global | random_forest | 0.884 ± 0.004 | 0.500 ± 0.000 | 0.329 ± 0.000 | — | 29s |
| global | mlp | 0.962 ± 0.001 | 0.995 ± 0.001 | 0.610 ± 0.007 | 12.0M | 3.3m |
| global | gru | 0.965 ± 0.003 | 0.998 ± 0.001 | 0.648 ± 0.010 | 69k | 2.1m |
| global | cnn | 0.960 ± 0.001 | 0.995 ± 0.002 | 0.607 ± 0.017 | 68k | 1.0m |
| global | resnet | 0.958 ± 0.006 | 0.996 ± 0.004 | 0.619 ± 0.035 | 530k | 3.6m |

**FC — fault classification** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| local | majority | 0.143 ± 0.000 | 0.143 ± 0.000 | 0.036 ± 0.000 | — | 0s |
| local | random_forest | 0.606 ± 0.005 | 0.539 ± 0.002 | 0.312 ± 0.002 | — | 10s |
| local | mlp | 0.679 ± 0.004 | 0.704 ± 0.001 | 0.516 ± 0.014 | 1.6M | 55s |
| local | gru | 0.679 ± 0.010 | 0.710 ± 0.002 | 0.529 ± 0.007 | 53k | 1.9m |
| local | cnn | 0.795 ± 0.012 | 0.812 ± 0.010 | 0.616 ± 0.012 | 50k | 2.0m |
| local | resnet | 0.855 ± 0.007 | 0.896 ± 0.045 | 0.693 ± 0.034 | 507k | 7.5m |
| line | majority | 0.143 ± 0.000 | 0.143 ± 0.000 | 0.036 ± 0.000 | — | 0s |
| line | random_forest | 0.661 ± 0.004 | 0.555 ± 0.003 | 0.322 ± 0.003 | — | 13s |
| line | mlp | 0.704 ± 0.005 | 0.716 ± 0.001 | 0.549 ± 0.003 | 3.1M | 1.1m |
| line | gru | 0.697 ± 0.006 | 0.719 ± 0.005 | 0.546 ± 0.019 | 56k | 2.0m |
| line | cnn | 0.792 ± 0.019 | 0.811 ± 0.016 | 0.600 ± 0.009 | 52k | 2.1m |
| line | resnet | 0.863 ± 0.007 | 0.873 ± 0.023 | 0.671 ± 0.020 | 510k | 6.3m |
| global | majority | 0.143 ± 0.000 | 0.143 ± 0.000 | 0.036 ± 0.000 | — | 0s |
| global | random_forest | 0.679 ± 0.003 | 0.581 ± 0.004 | 0.374 ± 0.004 | — | 23s |
| global | mlp | 0.676 ± 0.007 | 0.717 ± 0.002 | 0.519 ± 0.001 | 12.0M | 3.2m |
| global | gru | 0.669 ± 0.006 | 0.709 ± 0.004 | 0.537 ± 0.016 | 70k | 1.3m |
| global | cnn | 0.675 ± 0.013 | 0.709 ± 0.003 | 0.513 ± 0.003 | 68k | 1.1m |
| global | resnet | 0.818 ± 0.087 | 0.854 ± 0.097 | 0.653 ± 0.095 | 531k | 6.6m |

**fault origin (line vs bus)** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| global | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.463 ± 0.000 | — | 0s |
| global | random_forest | 0.842 ± 0.002 | 0.917 ± 0.003 | 0.799 ± 0.005 | — | 26s |
| global | mlp | 0.865 ± 0.003 | 0.990 ± 0.005 | 0.970 ± 0.005 | 12.0M | 3.0m |
| global | gru | 0.863 ± 0.007 | 0.969 ± 0.017 | 0.975 ± 0.003 | 69k | 2.1m |
| global | cnn | 0.867 ± 0.010 | 0.976 ± 0.019 | 0.971 ± 0.004 | 68k | 2.6m |
| global | resnet | 0.881 ± 0.012 | 0.975 ± 0.020 | 0.975 ± 0.006 | 530k | 6.9m |

**faulted phase (7-class)** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| local | majority | 0.143 ± 0.000 | 0.143 ± 0.000 | 0.035 ± 0.000 | — | 0s |
| local | random_forest | 0.799 ± 0.002 | 0.919 ± 0.002 | 0.919 ± 0.002 | — | 10s |
| local | mlp | 0.850 ± 0.005 | 0.979 ± 0.001 | 0.979 ± 0.001 | 1.6M | 56s |
| local | gru | 0.851 ± 0.012 | 0.975 ± 0.002 | 0.975 ± 0.002 | 53k | 2.2m |
| local | cnn | 0.881 ± 0.003 | 0.984 ± 0.002 | 0.984 ± 0.002 | 49k | 1.9m |
| local | resnet | 0.897 ± 0.029 | 0.990 ± 0.009 | 0.990 ± 0.009 | 506k | 6.8m |
| line | majority | 0.143 ± 0.000 | 0.143 ± 0.000 | 0.035 ± 0.000 | — | 0s |
| line | random_forest | 0.828 ± 0.003 | 0.961 ± 0.001 | 0.960 ± 0.001 | — | 12s |
| line | mlp | 0.849 ± 0.002 | 0.983 ± 0.001 | 0.983 ± 0.001 | 3.1M | 1.3m |
| line | gru | 0.865 ± 0.034 | 0.983 ± 0.002 | 0.983 ± 0.002 | 55k | 2.9m |
| line | cnn | 0.878 ± 0.002 | 0.986 ± 0.001 | 0.986 ± 0.001 | 52k | 2.1m |
| line | resnet | 0.894 ± 0.026 | 0.991 ± 0.007 | 0.990 ± 0.008 | 510k | 10.1m |
| global | majority | 0.143 ± 0.000 | 0.143 ± 0.000 | 0.035 ± 0.000 | — | 0s |
| global | random_forest | 0.839 ± 0.002 | 0.961 ± 0.005 | 0.958 ± 0.005 | — | 22s |
| global | mlp | 0.852 ± 0.003 | 0.983 ± 0.001 | 0.983 ± 0.001 | 12.0M | 8.6m |
| global | gru | 0.857 ± 0.015 | 0.982 ± 0.001 | 0.982 ± 0.001 | 69k | 2.5m |
| global | cnn | 0.865 ± 0.003 | 0.984 ± 0.002 | 0.983 ± 0.002 | 68k | 3.3m |
| global | resnet | 0.878 ± 0.034 | 0.988 ± 0.011 | 0.987 ± 0.011 | 531k | 5.8m |

**ST — switch type** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| global | majority | 0.250 ± 0.000 | 0.250 ± 0.000 | 0.050 ± 0.000 | — | 0s |
| global | random_forest | 0.729 ± 0.002 | 0.304 ± 0.000 | 0.097 ± 0.000 | — | 19s |
| global | mlp | 0.705 ± 0.009 | 0.732 ± 0.011 | 0.358 ± 0.024 | 12.0M | 4.5m |
| global | gru | 0.696 ± 0.014 | 0.720 ± 0.012 | 0.343 ± 0.026 | 69k | 2.1m |
| global | cnn | 0.710 ± 0.006 | 0.678 ± 0.036 | 0.298 ± 0.019 | 68k | 3.9m |
| global | resnet | 0.791 ± 0.050 | 0.738 ± 0.007 | 0.330 ± 0.003 | 531k | 7.2m |

#### Tier 3 — hard frontier


**EC — event classification (23-class)** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| global | majority | 0.062 ± 0.000 | 0.062 ± 0.000 | 0.021 ± 0.000 | — | 0s |
| global | random_forest | 0.568 ± 0.002 | 0.220 ± 0.009 | 0.138 ± 0.005 | — | 1.3m |
| global | mlp | 0.536 ± 0.004 | 0.573 ± 0.013 | 0.323 ± 0.017 | 12.0M | 9.7m |
| global | gru | 0.676 ± 0.014 | 0.751 ± 0.042 | 0.452 ± 0.034 | 71k | 10.0m |
| global | cnn | 0.630 ± 0.033 | 0.669 ± 0.063 | 0.382 ± 0.034 | 70k | 9.4m |
| global | resnet | 0.761 ± 0.006 | 0.833 ± 0.016 | 0.518 ± 0.012 | 533k | 20.5m |

**FL — fault location (regression)** (mae ↓)

| observability | baseline | mae (test) | mae (bench) | rmse (bench) | params | train |
|---|---|---|---|---|---|---|
| local | majority | 24.416 ± 0.000 | 31.094 ± 0.000 | 36.038 ± 0.000 | — | 0s |
| local | random_forest | 22.441 ± 0.079 | 26.961 ± 0.054 | 32.042 ± 0.075 | — | 36s |
| local | mlp | 19.690 ± 0.037 | 15.204 ± 0.163 | 18.944 ± 0.181 | 1.6M | 55s |
| local | gru | 18.600 ± 0.213 | 11.695 ± 0.530 | 15.704 ± 0.702 | 52k | 1.5m |
| local | cnn | 19.303 ± 0.179 | 15.649 ± 0.432 | 19.958 ± 0.403 | 49k | 1.6m |
| local | resnet | 19.357 ± 0.459 | 15.434 ± 1.558 | 19.592 ± 1.402 | 506k | 3.4m |
| line | majority | 24.416 ± 0.000 | 31.094 ± 0.000 | 36.038 ± 0.000 | — | 0s |
| line | random_forest | 19.120 ± 0.036 | 14.992 ± 0.059 | 18.906 ± 0.074 | — | 1.2m |
| line | mlp | 16.095 ± 0.030 | 5.179 ± 0.243 | 6.698 ± 0.339 | 3.1M | 1.1m |
| line | gru | 16.546 ± 0.127 | 6.248 ± 0.522 | 8.243 ± 0.798 | 55k | 46s |
| line | cnn | 17.841 ± 0.524 | 10.421 ± 1.154 | 14.692 ± 1.222 | 51k | 1.1m |
| line | resnet | 17.199 ± 0.263 | 8.809 ± 1.116 | 11.984 ± 1.616 | 509k | 3.7m |
| global | majority | 24.416 ± 0.000 | 31.094 ± 0.000 | 36.038 ± 0.000 | — | 0s |
| global | random_forest | 19.776 ± 0.039 | 17.824 ± 0.092 | 23.086 ± 0.114 | — | 4.8m |
| global | mlp | 16.338 ± 0.125 | 6.296 ± 0.269 | 8.701 ± 0.374 | 12.0M | 3.0m |
| global | gru | 15.869 ± 0.359 | 5.102 ± 0.715 | 7.080 ± 0.895 | 68k | 1.4m |
| global | cnn | 17.771 ± 0.093 | 10.106 ± 0.327 | 14.818 ± 0.488 | 67k | 1.7m |
| global | resnet | 17.445 ± 0.256 | 8.267 ± 1.416 | 11.622 ± 2.101 | 530k | 3.8m |

### CigreMV (20 kV MV)


#### Tier 1 — easy anchors


**ED — event detection** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| global | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.404 ± 0.000 | — | 0s |
| global | random_forest | 0.810 ± 0.000 | 0.984 ± 0.000 | 0.982 ± 0.001 | — | 7.9m |
| global | mlp | 0.748 ± 0.002 | 0.957 ± 0.000 | 0.937 ± 0.001 | 42.9M | 36.5m |
| global | gru | 0.785 ± 0.083 | 0.966 ± 0.015 | 0.950 ± 0.021 | 117k | 38.3m |
| global | cnn | 0.752 ± 0.015 | 0.966 ± 0.006 | 0.950 ± 0.009 | 124k | 36.8m |
| global | resnet | 0.884 ± 0.019 | 0.989 ± 0.007 | 0.987 ± 0.006 | 602k | 36.9m |

**FD — fault detection** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| local | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.407 ± 0.000 | — | 0s |
| local | threshold | 0.660 ± 0.000 | 0.810 ± 0.000 | 0.820 ± 0.000 | — | 2s |
| local | random_forest | 0.836 ± 0.002 | 0.983 ± 0.001 | 0.980 ± 0.001 | — | 17s |
| local | mlp | 0.772 ± 0.002 | 0.958 ± 0.003 | 0.936 ± 0.004 | 1.6M | 2.9m |
| local | gru | 0.825 ± 0.036 | 0.975 ± 0.013 | 0.962 ± 0.019 | 52k | 2.7m |
| local | cnn | 0.826 ± 0.006 | 0.970 ± 0.004 | 0.954 ± 0.006 | 49k | 2.4m |
| local | resnet | 0.831 ± 0.043 | 0.969 ± 0.008 | 0.955 ± 0.015 | 506k | 8.2m |
| line | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.407 ± 0.000 | — | 0s |
| line | random_forest | 0.838 ± 0.002 | 0.980 ± 0.003 | 0.978 ± 0.002 | — | 23s |
| line | mlp | 0.776 ± 0.002 | 0.964 ± 0.000 | 0.945 ± 0.001 | 3.1M | 1.3m |
| line | gru | 0.792 ± 0.020 | 0.965 ± 0.005 | 0.947 ± 0.006 | 55k | 2.7m |
| line | cnn | 0.808 ± 0.034 | 0.965 ± 0.008 | 0.948 ± 0.006 | 51k | 2.2m |
| line | resnet | 0.797 ± 0.039 | 0.963 ± 0.010 | 0.944 ± 0.015 | 509k | 6.8m |
| global | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.407 ± 0.000 | — | 0s |
| global | random_forest | 0.827 ± 0.003 | 0.985 ± 0.001 | 0.980 ± 0.002 | — | 3.0m |
| global | mlp | 0.763 ± 0.001 | 0.959 ± 0.000 | 0.937 ± 0.000 | 42.9M | 14.2m |
| global | gru | 0.786 ± 0.027 | 0.963 ± 0.006 | 0.944 ± 0.008 | 117k | 12.9m |
| global | cnn | 0.799 ± 0.022 | 0.961 ± 0.008 | 0.941 ± 0.011 | 124k | 12.6m |
| global | resnet | 0.799 ± 0.037 | 0.964 ± 0.015 | 0.945 ± 0.021 | 602k | 9.9m |

**grounded vs ungrounded** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| local | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.385 ± 0.000 | — | 0s |
| local | random_forest | 0.936 ± 0.002 | 0.956 ± 0.001 | 0.951 ± 0.001 | — | 9s |
| local | mlp | 0.948 ± 0.001 | 0.983 ± 0.001 | 0.982 ± 0.001 | 1.6M | 31s |
| local | gru | 0.940 ± 0.007 | 0.983 ± 0.006 | 0.982 ± 0.006 | 52k | 1.3m |
| local | cnn | 0.956 ± 0.005 | 0.993 ± 0.005 | 0.994 ± 0.005 | 49k | 1.1m |
| local | resnet | 0.966 ± 0.001 | 0.997 ± 0.001 | 0.998 ± 0.001 | 506k | 6.3m |
| line | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.385 ± 0.000 | — | 0s |
| line | random_forest | 0.945 ± 0.001 | 0.982 ± 0.000 | 0.979 ± 0.000 | — | 12s |
| line | mlp | 0.947 ± 0.001 | 0.987 ± 0.001 | 0.985 ± 0.001 | 3.1M | 41s |
| line | gru | 0.946 ± 0.010 | 0.985 ± 0.007 | 0.983 ± 0.008 | 55k | 1.4m |
| line | cnn | 0.960 ± 0.008 | 0.994 ± 0.004 | 0.994 ± 0.004 | 51k | 1.2m |
| line | resnet | 0.963 ± 0.007 | 0.996 ± 0.001 | 0.995 ± 0.001 | 509k | 5.6m |
| global | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.385 ± 0.000 | — | 0s |
| global | random_forest | 0.935 ± 0.002 | 0.989 ± 0.001 | 0.987 ± 0.001 | — | 1.1m |
| global | mlp | 0.928 ± 0.013 | 0.975 ± 0.008 | 0.973 ± 0.007 | 42.9M | 7.0m |
| global | gru | 0.944 ± 0.012 | 0.986 ± 0.005 | 0.985 ± 0.005 | 117k | 5.9m |
| global | cnn | 0.946 ± 0.007 | 0.988 ± 0.005 | 0.988 ± 0.005 | 124k | 6.8m |
| global | resnet | 0.953 ± 0.008 | 0.994 ± 0.002 | 0.993 ± 0.003 | 602k | 6.7m |

**SD — switch detection** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| global | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.496 ± 0.000 | — | 0s |
| global | random_forest | 0.754 ± 0.002 | 0.816 ± 0.005 | 0.819 ± 0.013 | — | 6.7m |
| global | mlp | 0.688 ± 0.002 | 0.710 ± 0.001 | 0.734 ± 0.006 | 42.9M | 36.6m |
| global | gru | 0.718 ± 0.127 | 0.758 ± 0.102 | 0.625 ± 0.158 | 117k | 28.3m |
| global | cnn | 0.781 ± 0.012 | 0.822 ± 0.014 | 0.761 ± 0.045 | 124k | 37.9m |
| global | resnet | 0.865 ± 0.011 | 0.914 ± 0.014 | 0.902 ± 0.014 | 602k | 37.2m |

#### Tier 2 — medium


**ES — event state (3-class)** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| global | majority | 0.333 ± 0.000 | 0.333 ± 0.000 | 0.163 ± 0.000 | — | 0s |
| global | random_forest | 0.724 ± 0.001 | 0.866 ± 0.002 | 0.851 ± 0.003 | — | 7.8m |
| global | mlp | 0.659 ± 0.001 | 0.798 ± 0.002 | 0.804 ± 0.003 | 42.9M | 38.6m |
| global | gru | 0.740 ± 0.030 | 0.848 ± 0.025 | 0.810 ± 0.015 | 117k | 39.2m |
| global | cnn | 0.733 ± 0.027 | 0.865 ± 0.024 | 0.871 ± 0.019 | 124k | 38.5m |
| global | resnet | 0.696 ± 0.111 | 0.845 ± 0.069 | 0.812 ± 0.075 | 603k | 20.1m |

**fault category (shc/incipient/hif)** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| local | majority | 0.333 ± 0.000 | 0.333 ± 0.000 | 0.322 ± 0.000 | — | 0s |
| local | random_forest | 0.597 ± 0.005 | 0.631 ± 0.009 | 0.642 ± 0.008 | — | 9s |
| local | mlp | 0.636 ± 0.004 | 0.655 ± 0.002 | 0.557 ± 0.004 | 1.6M | 32s |
| local | gru | 0.646 ± 0.006 | 0.664 ± 0.002 | 0.608 ± 0.018 | 53k | 1.8m |
| local | cnn | 0.645 ± 0.016 | 0.660 ± 0.004 | 0.593 ± 0.024 | 49k | 1.0m |
| local | resnet | 0.651 ± 0.016 | 0.680 ± 0.039 | 0.634 ± 0.076 | 506k | 5.7m |
| line | majority | 0.333 ± 0.000 | 0.333 ± 0.000 | 0.322 ± 0.000 | — | 0s |
| line | random_forest | 0.595 ± 0.007 | 0.643 ± 0.006 | 0.650 ± 0.004 | — | 12s |
| line | mlp | 0.649 ± 0.003 | 0.662 ± 0.001 | 0.588 ± 0.008 | 3.1M | 39s |
| line | gru | 0.656 ± 0.015 | 0.670 ± 0.012 | 0.622 ± 0.039 | 55k | 1.8m |
| line | cnn | 0.670 ± 0.003 | 0.673 ± 0.015 | 0.637 ± 0.028 | 52k | 1.5m |
| line | resnet | 0.658 ± 0.027 | 0.674 ± 0.032 | 0.610 ± 0.080 | 509k | 4.7m |
| global | majority | 0.333 ± 0.000 | 0.333 ± 0.000 | 0.322 ± 0.000 | — | 0s |
| global | random_forest | 0.569 ± 0.007 | 0.665 ± 0.001 | 0.579 ± 0.001 | — | 59s |
| global | mlp | 0.634 ± 0.009 | 0.668 ± 0.043 | 0.548 ± 0.070 | 42.9M | 10.1m |
| global | gru | 0.638 ± 0.024 | 0.674 ± 0.032 | 0.627 ± 0.037 | 117k | 5.5m |
| global | cnn | 0.654 ± 0.022 | 0.656 ± 0.007 | 0.542 ± 0.063 | 124k | 5.3m |
| global | resnet | 0.635 ± 0.017 | 0.657 ± 0.003 | 0.505 ± 0.058 | 603k | 6.3m |

**FC — fault classification** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| local | majority | 0.111 ± 0.000 | 0.111 ± 0.000 | 0.035 ± 0.000 | — | 0s |
| local | random_forest | 0.477 ± 0.005 | 0.526 ± 0.004 | 0.521 ± 0.004 | — | 9s |
| local | mlp | 0.527 ± 0.005 | 0.546 ± 0.002 | 0.504 ± 0.006 | 1.6M | 54s |
| local | gru | 0.519 ± 0.009 | 0.542 ± 0.001 | 0.510 ± 0.006 | 53k | 2.9m |
| local | cnn | 0.566 ± 0.028 | 0.589 ± 0.031 | 0.567 ± 0.032 | 50k | 1.9m |
| local | resnet | 0.656 ± 0.011 | 0.701 ± 0.011 | 0.696 ± 0.012 | 507k | 6.6m |
| line | majority | 0.111 ± 0.000 | 0.111 ± 0.000 | 0.035 ± 0.000 | — | 0s |
| line | random_forest | 0.500 ± 0.005 | 0.531 ± 0.003 | 0.529 ± 0.003 | — | 12s |
| line | mlp | 0.523 ± 0.002 | 0.549 ± 0.002 | 0.509 ± 0.002 | 3.1M | 1.0m |
| line | gru | 0.533 ± 0.009 | 0.552 ± 0.005 | 0.518 ± 0.012 | 56k | 2.2m |
| line | cnn | 0.576 ± 0.022 | 0.589 ± 0.028 | 0.568 ± 0.030 | 52k | 2.0m |
| line | resnet | 0.665 ± 0.010 | 0.700 ± 0.010 | 0.694 ± 0.011 | 510k | 7.0m |
| global | majority | 0.111 ± 0.000 | 0.111 ± 0.000 | 0.035 ± 0.000 | — | 0s |
| global | random_forest | 0.524 ± 0.004 | 0.548 ± 0.002 | 0.497 ± 0.007 | — | 57s |
| global | mlp | 0.523 ± 0.003 | 0.542 ± 0.001 | 0.465 ± 0.001 | 42.9M | 9.8m |
| global | gru | 0.517 ± 0.019 | 0.540 ± 0.011 | 0.515 ± 0.011 | 118k | 5.1m |
| global | cnn | 0.510 ± 0.012 | 0.534 ± 0.006 | 0.479 ± 0.020 | 125k | 4.6m |
| global | resnet | 0.520 ± 0.024 | 0.548 ± 0.007 | 0.481 ± 0.007 | 603k | 5.5m |

**fault origin (line vs bus)** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| global | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.451 ± 0.000 | — | 0s |
| global | random_forest | 0.581 ± 0.009 | 0.574 ± 0.003 | 0.519 ± 0.005 | — | 44s |
| global | mlp | 0.774 ± 0.020 | 0.969 ± 0.021 | 0.935 ± 0.015 | 42.9M | 9.0m |
| global | gru | 0.802 ± 0.027 | 0.940 ± 0.031 | 0.945 ± 0.019 | 117k | 7.6m |
| global | cnn | 0.811 ± 0.020 | 0.919 ± 0.035 | 0.930 ± 0.010 | 124k | 8.6m |
| global | resnet | 0.778 ± 0.018 | 0.944 ± 0.042 | 0.927 ± 0.013 | 602k | 3.1m |

**faulted phase (7-class)** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| local | majority | 0.143 ± 0.000 | 0.143 ± 0.000 | 0.036 ± 0.000 | — | 0s |
| local | random_forest | 0.783 ± 0.002 | 0.935 ± 0.000 | 0.935 ± 0.000 | — | 7s |
| local | mlp | 0.810 ± 0.004 | 0.963 ± 0.001 | 0.963 ± 0.001 | 1.6M | 29s |
| local | gru | 0.809 ± 0.008 | 0.954 ± 0.003 | 0.953 ± 0.004 | 53k | 1.7m |
| local | cnn | 0.854 ± 0.002 | 0.972 ± 0.003 | 0.972 ± 0.003 | 49k | 1.4m |
| local | resnet | 0.843 ± 0.023 | 0.970 ± 0.008 | 0.970 ± 0.008 | 506k | 4.0m |
| line | majority | 0.143 ± 0.000 | 0.143 ± 0.000 | 0.036 ± 0.000 | — | 0s |
| line | random_forest | 0.797 ± 0.002 | 0.954 ± 0.001 | 0.955 ± 0.001 | — | 11s |
| line | mlp | 0.805 ± 0.002 | 0.966 ± 0.001 | 0.966 ± 0.001 | 3.1M | 35s |
| line | gru | 0.811 ± 0.031 | 0.962 ± 0.012 | 0.962 ± 0.012 | 55k | 1.4m |
| line | cnn | 0.850 ± 0.003 | 0.975 ± 0.001 | 0.975 ± 0.001 | 52k | 1.4m |
| line | resnet | 0.858 ± 0.036 | 0.977 ± 0.010 | 0.977 ± 0.010 | 510k | 5.2m |
| global | majority | 0.143 ± 0.000 | 0.143 ± 0.000 | 0.036 ± 0.000 | — | 0s |
| global | random_forest | 0.794 ± 0.003 | 0.949 ± 0.004 | 0.951 ± 0.004 | — | 51s |
| global | mlp | 0.807 ± 0.003 | 0.964 ± 0.003 | 0.964 ± 0.003 | 42.9M | 8.9m |
| global | gru | 0.828 ± 0.019 | 0.966 ± 0.006 | 0.966 ± 0.007 | 118k | 8.1m |
| global | cnn | 0.837 ± 0.003 | 0.977 ± 0.004 | 0.977 ± 0.004 | 125k | 9.0m |
| global | resnet | 0.800 ± 0.022 | 0.954 ± 0.009 | 0.954 ± 0.009 | 603k | 3.6m |

**ST — switch type** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| global | majority | 0.143 ± 0.000 | 0.143 ± 0.000 | 0.038 ± 0.000 | — | 0s |
| global | random_forest | 0.605 ± 0.005 | 0.751 ± 0.003 | 0.638 ± 0.003 | — | 54s |
| global | mlp | 0.592 ± 0.002 | 0.687 ± 0.013 | 0.609 ± 0.012 | 42.9M | 10.7m |
| global | gru | 0.739 ± 0.012 | 0.781 ± 0.020 | 0.690 ± 0.018 | 118k | 26.6m |
| global | cnn | 0.726 ± 0.005 | 0.782 ± 0.016 | 0.692 ± 0.013 | 125k | 12.4m |
| global | resnet | 0.628 ± 0.064 | 0.683 ± 0.074 | 0.590 ± 0.075 | 603k | 3.3m |

#### Tier 3 — hard frontier


**EC — event classification (23-class)** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| global | majority | 0.045 ± 0.000 | 0.045 ± 0.000 | 0.021 ± 0.000 | — | 0s |
| global | random_forest | 0.455 ± 0.005 | 0.520 ± 0.006 | 0.409 ± 0.005 | — | 6.1m |
| global | mlp | 0.521 ± 0.002 | 0.542 ± 0.006 | 0.377 ± 0.008 | 42.9M | 43.0m |
| global | gru | 0.640 ± 0.009 | 0.698 ± 0.031 | 0.517 ± 0.030 | 120k | 40.6m |
| global | cnn | 0.632 ± 0.011 | 0.702 ± 0.011 | 0.558 ± 0.018 | 127k | 37.9m |
| global | resnet | 0.613 ± 0.091 | 0.670 ± 0.092 | 0.553 ± 0.112 | 605k | 15.6m |

**FL — fault location (regression)** (mae ↓)

| observability | baseline | mae (test) | mae (bench) | rmse (bench) | params | train |
|---|---|---|---|---|---|---|
| local | majority | 23.915 ± 0.000 | 30.351 ± 0.000 | 35.576 ± 0.000 | — | 0s |
| local | random_forest | 24.793 ± 0.039 | 32.533 ± 0.025 | 37.815 ± 0.044 | — | 40s |
| local | mlp | 23.943 ± 0.040 | 30.530 ± 0.155 | 35.511 ± 0.149 | 1.6M | 17s |
| local | gru | 23.934 ± 0.024 | 30.485 ± 0.049 | 35.542 ± 0.021 | 52k | 33s |
| local | cnn | 23.986 ± 0.110 | 30.710 ± 0.117 | 35.630 ± 0.120 | 49k | 30s |
| local | resnet | 23.930 ± 0.071 | 30.969 ± 0.623 | 36.158 ± 1.156 | 506k | 1.5m |
| line | majority | 23.915 ± 0.000 | 30.351 ± 0.000 | 35.576 ± 0.000 | — | 0s |
| line | random_forest | 24.443 ± 0.038 | 31.076 ± 0.051 | 36.320 ± 0.069 | — | 1.2m |
| line | mlp | 23.915 ± 0.057 | 30.470 ± 0.095 | 35.470 ± 0.083 | 3.1M | 21s |
| line | gru | 23.887 ± 0.041 | 30.485 ± 0.062 | 35.469 ± 0.130 | 55k | 39s |
| line | cnn | 23.005 ± 0.552 | 24.522 ± 4.373 | 30.117 ± 3.913 | 51k | 1.1m |
| line | resnet | 23.890 ± 0.162 | 29.672 ± 2.190 | 34.721 ± 1.658 | 509k | 1.9m |
| global | majority | 23.915 ± 0.000 | 30.351 ± 0.000 | 35.576 ± 0.000 | — | 0s |
| global | random_forest | 24.131 ± 0.047 | 30.481 ± 0.083 | 35.294 ± 0.058 | — | 12.6m |
| global | mlp | 22.196 ± 0.031 | 18.868 ± 0.943 | 25.599 ± 0.692 | 42.9M | 9.6m |
| global | gru | 22.648 ± 0.381 | 20.388 ± 2.245 | 26.257 ± 1.817 | 117k | 3.0m |
| global | cnn | 22.345 ± 0.302 | 19.521 ± 0.995 | 25.709 ± 0.835 | 124k | 3.4m |
| global | resnet | 22.857 ± 0.448 | 21.112 ± 2.293 | 26.827 ± 2.020 | 602k | 3.0m |

### TestGrid (110 kV HV)


#### Tier 1 — easy anchors


**ED — event detection** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| global | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.405 ± 0.000 | — | 0s |
| global | random_forest | 0.784 ± 0.001 | 0.949 ± 0.001 | 0.925 ± 0.002 | — | 3.1m |
| global | mlp | 0.790 ± 0.003 | 0.984 ± 0.000 | 0.975 ± 0.000 | 26.7M | 20.6m |
| global | gru | 0.881 ± 0.045 | 0.994 ± 0.004 | 0.992 ± 0.006 | 92k | 21.4m |
| global | cnn | 0.761 ± 0.009 | 0.982 ± 0.001 | 0.973 ± 0.001 | 94k | 21.1m |
| global | resnet | 0.873 ± 0.027 | 0.994 ± 0.003 | 0.993 ± 0.003 | 564k | 26.6m |

**FD — fault detection** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| local | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.407 ± 0.000 | — | 0s |
| local | threshold | 0.755 ± 0.000 | 0.886 ± 0.000 | 0.874 ± 0.000 | — | 2s |
| local | random_forest | 0.882 ± 0.001 | 0.672 ± 0.002 | 0.690 ± 0.003 | — | 15s |
| local | mlp | 0.847 ± 0.001 | 0.982 ± 0.000 | 0.971 ± 0.001 | 1.6M | 1.1m |
| local | gru | 0.846 ± 0.063 | 0.963 ± 0.035 | 0.952 ± 0.043 | 52k | 2.4m |
| local | cnn | 0.875 ± 0.014 | 0.987 ± 0.002 | 0.980 ± 0.002 | 49k | 1.7m |
| local | resnet | 0.949 ± 0.016 | 0.995 ± 0.004 | 0.995 ± 0.004 | 506k | 9.3m |
| line | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.407 ± 0.000 | — | 0s |
| line | random_forest | 0.878 ± 0.001 | 0.681 ± 0.028 | 0.702 ± 0.035 | — | 21s |
| line | mlp | 0.849 ± 0.002 | 0.985 ± 0.000 | 0.977 ± 0.001 | 3.1M | 1.3m |
| line | gru | 0.871 ± 0.024 | 0.967 ± 0.055 | 0.965 ± 0.041 | 55k | 2.2m |
| line | cnn | 0.875 ± 0.010 | 0.987 ± 0.002 | 0.980 ± 0.003 | 51k | 2.2m |
| line | resnet | 0.914 ± 0.026 | 0.990 ± 0.003 | 0.987 ± 0.008 | 509k | 9.3m |
| global | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.407 ± 0.000 | — | 0s |
| global | random_forest | 0.866 ± 0.003 | 0.500 ± 0.000 | 0.407 ± 0.000 | — | 1.1m |
| global | mlp | 0.848 ± 0.001 | 0.985 ± 0.000 | 0.976 ± 0.001 | 26.7M | 8.4m |
| global | gru | 0.857 ± 0.022 | 0.985 ± 0.003 | 0.977 ± 0.004 | 92k | 5.7m |
| global | cnn | 0.878 ± 0.010 | 0.987 ± 0.003 | 0.980 ± 0.004 | 94k | 8.4m |
| global | resnet | 0.864 ± 0.020 | 0.987 ± 0.004 | 0.980 ± 0.006 | 564k | 9.5m |

**grounded vs ungrounded** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| local | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.379 ± 0.000 | — | 0s |
| local | random_forest | 0.922 ± 0.002 | 0.817 ± 0.004 | 0.835 ± 0.004 | — | 8s |
| local | mlp | 0.956 ± 0.002 | 0.990 ± 0.001 | 0.991 ± 0.001 | 1.6M | 51s |
| local | gru | 0.938 ± 0.039 | 0.979 ± 0.019 | 0.980 ± 0.018 | 52k | 1.3m |
| local | cnn | 0.969 ± 0.001 | 0.998 ± 0.001 | 0.998 ± 0.001 | 49k | 2.2m |
| local | resnet | 0.967 ± 0.002 | 0.998 ± 0.002 | 0.999 ± 0.002 | 506k | 4.3m |
| line | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.379 ± 0.000 | — | 0s |
| line | random_forest | 0.938 ± 0.001 | 0.829 ± 0.009 | 0.848 ± 0.009 | — | 11s |
| line | mlp | 0.968 ± 0.001 | 0.996 ± 0.000 | 0.996 ± 0.000 | 3.1M | 1.0m |
| line | gru | 0.969 ± 0.001 | 0.998 ± 0.001 | 0.998 ± 0.001 | 55k | 2.3m |
| line | cnn | 0.969 ± 0.001 | 1.000 ± 0.000 | 1.000 ± 0.000 | 51k | 1.7m |
| line | resnet | 0.969 ± 0.002 | 0.998 ± 0.003 | 0.998 ± 0.003 | 509k | 5.2m |
| global | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.379 ± 0.000 | — | 0s |
| global | random_forest | 0.960 ± 0.001 | 0.680 ± 0.008 | 0.679 ± 0.010 | — | 36s |
| global | mlp | 0.969 ± 0.000 | 0.995 ± 0.000 | 0.994 ± 0.000 | 26.7M | 6.3m |
| global | gru | 0.970 ± 0.000 | 0.996 ± 0.002 | 0.996 ± 0.001 | 92k | 4.4m |
| global | cnn | 0.969 ± 0.001 | 0.999 ± 0.001 | 0.999 ± 0.001 | 94k | 5.0m |
| global | resnet | 0.969 ± 0.001 | 0.998 ± 0.002 | 0.998 ± 0.001 | 564k | 4.4m |

**SD — switch detection** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| global | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.498 ± 0.000 | — | 0s |
| global | random_forest | 0.682 ± 0.002 | 0.592 ± 0.006 | 0.636 ± 0.005 | — | 2.5m |
| global | mlp | 0.711 ± 0.001 | 0.734 ± 0.003 | 0.728 ± 0.003 | 26.7M | 19.6m |
| global | gru | 0.725 ± 0.057 | 0.805 ± 0.042 | 0.450 ± 0.024 | 92k | 17.5m |
| global | cnn | 0.694 ± 0.021 | 0.823 ± 0.002 | 0.422 ± 0.001 | 94k | 21.8m |
| global | resnet | 0.786 ± 0.006 | 0.785 ± 0.031 | 0.704 ± 0.214 | 564k | 31.5m |

#### Tier 2 — medium


**ES — event state (3-class)** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| global | majority | 0.333 ± 0.000 | 0.333 ± 0.000 | 0.161 ± 0.000 | — | 0s |
| global | random_forest | 0.705 ± 0.001 | 0.652 ± 0.006 | 0.628 ± 0.008 | — | 3.1m |
| global | mlp | 0.706 ± 0.001 | 0.805 ± 0.001 | 0.827 ± 0.007 | 26.7M | 19.1m |
| global | gru | 0.689 ± 0.060 | 0.801 ± 0.042 | 0.808 ± 0.053 | 92k | 17.1m |
| global | cnn | 0.662 ± 0.021 | 0.789 ± 0.030 | 0.803 ± 0.027 | 95k | 20.0m |
| global | resnet | 0.839 ± 0.013 | 0.868 ± 0.076 | 0.824 ± 0.167 | 565k | 28.2m |

**fault category (shc/incipient/hif)** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| local | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.329 ± 0.000 | — | 0s |
| local | random_forest | 0.772 ± 0.004 | 0.637 ± 0.010 | 0.423 ± 0.006 | — | 10s |
| local | mlp | 0.945 ± 0.001 | 0.985 ± 0.001 | 0.534 ± 0.005 | 1.6M | 52s |
| local | gru | 0.960 ± 0.002 | 0.991 ± 0.001 | 0.574 ± 0.005 | 53k | 1.6m |
| local | cnn | 0.958 ± 0.004 | 0.983 ± 0.005 | 0.525 ± 0.023 | 49k | 58s |
| local | resnet | 0.954 ± 0.007 | 0.984 ± 0.005 | 0.530 ± 0.030 | 506k | 4.3m |
| line | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.329 ± 0.000 | — | 0s |
| line | random_forest | 0.824 ± 0.005 | 0.628 ± 0.004 | 0.447 ± 0.004 | — | 14s |
| line | mlp | 0.955 ± 0.000 | 0.995 ± 0.000 | 0.607 ± 0.002 | 3.1M | 59s |
| line | gru | 0.963 ± 0.002 | 0.998 ± 0.000 | 0.646 ± 0.006 | 55k | 1.3m |
| line | cnn | 0.961 ± 0.003 | 0.997 ± 0.001 | 0.632 ± 0.017 | 52k | 1.4m |
| line | resnet | 0.960 ± 0.003 | 0.997 ± 0.001 | 0.634 ± 0.009 | 509k | 4.0m |
| global | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.329 ± 0.000 | — | 0s |
| global | random_forest | 0.829 ± 0.005 | 0.500 ± 0.000 | 0.329 ± 0.000 | — | 41s |
| global | mlp | 0.956 ± 0.007 | 0.993 ± 0.003 | 0.588 ± 0.021 | 26.7M | 5.0m |
| global | gru | 0.963 ± 0.004 | 0.997 ± 0.004 | 0.632 ± 0.035 | 92k | 3.9m |
| global | cnn | 0.957 ± 0.003 | 0.989 ± 0.005 | 0.565 ± 0.037 | 95k | 14.4m |
| global | resnet | 0.953 ± 0.005 | 0.990 ± 0.007 | 0.576 ± 0.061 | 565k | 11.2m |

**FC — fault classification** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| local | majority | 0.143 ± 0.000 | 0.143 ± 0.000 | 0.036 ± 0.000 | — | 0s |
| local | random_forest | 0.628 ± 0.003 | 0.586 ± 0.006 | 0.411 ± 0.006 | — | 9s |
| local | mlp | 0.656 ± 0.010 | 0.705 ± 0.001 | 0.522 ± 0.005 | 1.6M | 51s |
| local | gru | 0.667 ± 0.012 | 0.709 ± 0.002 | 0.532 ± 0.007 | 53k | 1.8m |
| local | cnn | 0.685 ± 0.005 | 0.727 ± 0.004 | 0.542 ± 0.008 | 50k | 1.8m |
| local | resnet | 0.825 ± 0.006 | 0.848 ± 0.007 | 0.642 ± 0.011 | 507k | 6.7m |
| line | majority | 0.143 ± 0.000 | 0.143 ± 0.000 | 0.036 ± 0.000 | — | 0s |
| line | random_forest | 0.631 ± 0.003 | 0.631 ± 0.009 | 0.455 ± 0.006 | — | 11s |
| line | mlp | 0.679 ± 0.005 | 0.715 ± 0.001 | 0.536 ± 0.003 | 3.1M | 1.0m |
| line | gru | 0.681 ± 0.005 | 0.718 ± 0.003 | 0.543 ± 0.014 | 56k | 1.8m |
| line | cnn | 0.706 ± 0.012 | 0.743 ± 0.021 | 0.558 ± 0.028 | 52k | 1.7m |
| line | resnet | 0.805 ± 0.036 | 0.834 ± 0.015 | 0.638 ± 0.010 | 510k | 6.2m |
| global | majority | 0.143 ± 0.000 | 0.143 ± 0.000 | 0.036 ± 0.000 | — | 0s |
| global | random_forest | 0.660 ± 0.004 | 0.600 ± 0.005 | 0.378 ± 0.006 | — | 34s |
| global | mlp | 0.676 ± 0.004 | 0.713 ± 0.001 | 0.539 ± 0.016 | 26.7M | 6.1m |
| global | gru | 0.680 ± 0.007 | 0.704 ± 0.007 | 0.523 ± 0.026 | 93k | 2.6m |
| global | cnn | 0.670 ± 0.008 | 0.707 ± 0.007 | 0.503 ± 0.009 | 95k | 2.2m |
| global | resnet | 0.773 ± 0.107 | 0.772 ± 0.070 | 0.577 ± 0.073 | 565k | 5.9m |

**fault origin (line vs bus)** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| global | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.467 ± 0.000 | — | 0s |
| global | random_forest | 0.759 ± 0.006 | 0.799 ± 0.017 | 0.710 ± 0.005 | — | 34s |
| global | mlp | 0.854 ± 0.007 | 0.984 ± 0.016 | 0.962 ± 0.007 | 26.7M | 8.5m |
| global | gru | 0.875 ± 0.016 | 0.972 ± 0.014 | 0.971 ± 0.006 | 92k | 5.7m |
| global | cnn | 0.886 ± 0.010 | 0.974 ± 0.016 | 0.968 ± 0.007 | 94k | 5.1m |
| global | resnet | 0.866 ± 0.026 | 0.983 ± 0.019 | 0.964 ± 0.008 | 564k | 4.7m |

**faulted phase (7-class)** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| local | majority | 0.143 ± 0.000 | 0.143 ± 0.000 | 0.035 ± 0.000 | — | 0s |
| local | random_forest | 0.839 ± 0.001 | 0.949 ± 0.001 | 0.948 ± 0.001 | — | 9s |
| local | mlp | 0.843 ± 0.004 | 0.979 ± 0.000 | 0.978 ± 0.001 | 1.6M | 1.2m |
| local | gru | 0.838 ± 0.032 | 0.970 ± 0.010 | 0.969 ± 0.010 | 53k | 2.2m |
| local | cnn | 0.883 ± 0.023 | 0.981 ± 0.006 | 0.981 ± 0.006 | 49k | 1.7m |
| local | resnet | 0.898 ± 0.026 | 0.989 ± 0.008 | 0.988 ± 0.008 | 506k | 6.1m |
| line | majority | 0.143 ± 0.000 | 0.143 ± 0.000 | 0.035 ± 0.000 | — | 0s |
| line | random_forest | 0.839 ± 0.002 | 0.966 ± 0.002 | 0.966 ± 0.002 | — | 11s |
| line | mlp | 0.856 ± 0.004 | 0.983 ± 0.000 | 0.983 ± 0.000 | 3.1M | 1.1m |
| line | gru | 0.876 ± 0.010 | 0.982 ± 0.001 | 0.982 ± 0.001 | 55k | 2.3m |
| line | cnn | 0.886 ± 0.003 | 0.984 ± 0.002 | 0.984 ± 0.002 | 52k | 1.9m |
| line | resnet | 0.895 ± 0.020 | 0.987 ± 0.005 | 0.986 ± 0.005 | 510k | 6.4m |
| global | majority | 0.143 ± 0.000 | 0.143 ± 0.000 | 0.035 ± 0.000 | — | 0s |
| global | random_forest | 0.833 ± 0.006 | 0.973 ± 0.004 | 0.973 ± 0.004 | — | 35s |
| global | mlp | 0.849 ± 0.004 | 0.982 ± 0.001 | 0.981 ± 0.002 | 26.7M | 6.2m |
| global | gru | 0.866 ± 0.024 | 0.982 ± 0.003 | 0.982 ± 0.003 | 92k | 4.9m |
| global | cnn | 0.874 ± 0.017 | 0.980 ± 0.002 | 0.980 ± 0.002 | 95k | 5.6m |
| global | resnet | 0.867 ± 0.022 | 0.977 ± 0.006 | 0.976 ± 0.007 | 565k | 4.8m |

**ST — switch type** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| global | majority | 0.200 ± 0.000 | 0.200 ± 0.000 | 0.048 ± 0.000 | — | 0s |
| global | random_forest | 0.658 ± 0.005 | 0.247 ± 0.001 | 0.098 ± 0.001 | — | 37s |
| global | mlp | 0.694 ± 0.002 | 0.692 ± 0.026 | 0.371 ± 0.023 | 26.7M | 6.2m |
| global | gru | 0.747 ± 0.006 | 0.766 ± 0.029 | 0.423 ± 0.036 | 92k | 5.7m |
| global | cnn | 0.729 ± 0.003 | 0.739 ± 0.007 | 0.411 ± 0.010 | 95k | 5.6m |
| global | resnet | 0.835 ± 0.056 | 0.865 ± 0.084 | 0.529 ± 0.069 | 565k | 7.3m |

#### Tier 3 — hard frontier


**EC — event classification (23-class)** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| global | majority | 0.059 ± 0.000 | 0.059 ± 0.000 | 0.021 ± 0.000 | — | 0s |
| global | random_forest | 0.523 ± 0.002 | 0.227 ± 0.006 | 0.150 ± 0.006 | — | 2.5m |
| global | mlp | 0.556 ± 0.002 | 0.584 ± 0.020 | 0.344 ± 0.014 | 26.7M | 20.4m |
| global | gru | 0.675 ± 0.013 | 0.730 ± 0.030 | 0.423 ± 0.022 | 94k | 19.6m |
| global | cnn | 0.654 ± 0.011 | 0.706 ± 0.015 | 0.412 ± 0.007 | 97k | 20.3m |
| global | resnet | 0.782 ± 0.006 | 0.831 ± 0.025 | 0.537 ± 0.022 | 567k | 20.3m |

**FL — fault location (regression)** (mae ↓)

| observability | baseline | mae (test) | mae (bench) | rmse (bench) | params | train |
|---|---|---|---|---|---|---|
| local | majority | 24.378 ± 0.000 | 31.168 ± 0.000 | 36.040 ± 0.000 | — | 0s |
| local | random_forest | 23.653 ± 0.042 | 27.029 ± 0.048 | 31.951 ± 0.058 | — | 31s |
| local | mlp | 20.435 ± 0.046 | 17.456 ± 0.118 | 21.818 ± 0.105 | 1.6M | 53s |
| local | gru | 18.768 ± 0.171 | 10.855 ± 0.572 | 15.262 ± 0.761 | 52k | 1.5m |
| local | cnn | 19.727 ± 0.084 | 14.990 ± 0.643 | 19.748 ± 0.708 | 49k | 1.6m |
| local | resnet | 19.904 ± 0.229 | 15.943 ± 0.645 | 20.797 ± 0.536 | 506k | 2.9m |
| line | majority | 24.378 ± 0.000 | 31.168 ± 0.000 | 36.040 ± 0.000 | — | 0s |
| line | random_forest | 21.576 ± 0.051 | 19.552 ± 0.089 | 24.106 ± 0.076 | — | 59s |
| line | mlp | 17.141 ± 0.065 | 6.545 ± 0.540 | 8.846 ± 0.579 | 3.1M | 59s |
| line | gru | 17.256 ± 0.294 | 6.415 ± 0.903 | 8.617 ± 1.130 | 55k | 1.0m |
| line | cnn | 17.693 ± 0.366 | 8.842 ± 1.200 | 12.243 ± 1.766 | 51k | 1.6m |
| line | resnet | 17.864 ± 0.278 | 8.733 ± 0.828 | 11.630 ± 1.055 | 509k | 3.1m |
| global | majority | 24.378 ± 0.000 | 31.168 ± 0.000 | 36.040 ± 0.000 | — | 0s |
| global | random_forest | 21.830 ± 0.072 | 23.947 ± 0.088 | 29.280 ± 0.078 | — | 12.6m |
| global | mlp | 16.994 ± 0.043 | 6.538 ± 0.115 | 9.471 ± 0.182 | 26.7M | 6.5m |
| global | gru | 17.359 ± 0.181 | 7.804 ± 0.638 | 10.579 ± 0.851 | 92k | 2.3m |
| global | cnn | 18.590 ± 0.215 | 11.585 ± 0.441 | 16.314 ± 0.512 | 94k | 3.2m |
| global | resnet | 18.794 ± 0.250 | 12.354 ± 0.641 | 17.127 ± 0.703 | 564k | 2.6m |

### IEEE-39 (345 kV EHV)


#### Tier 1 — easy anchors


**ED — event detection** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| global | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.405 ± 0.000 | — | 0s |
| global | random_forest | 0.930 ± 0.000 | 0.506 ± 0.000 | 0.419 ± 0.000 | — | 18.3m |
| global | mlp | 0.843 ± 0.001 | 0.981 ± 0.001 | 0.971 ± 0.001 | 103.4M | 1.6h |
| global | gru | 0.770 ± 0.051 | 0.867 ± 0.128 | 0.876 ± 0.114 | 211k | 1.5h |
| global | cnn | 0.786 ± 0.005 | 0.706 ± 0.313 | 0.659 ± 0.362 | 234k | 1.7h |
| global | resnet | 0.832 ± 0.058 | 0.775 ± 0.314 | 0.739 ± 0.375 | 744k | 1.6h |

**FD — fault detection** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| local | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.407 ± 0.000 | — | 0s |
| local | threshold | 0.793 ± 0.000 | 0.914 ± 0.000 | 0.894 ± 0.000 | — | 2s |
| local | random_forest | 0.931 ± 0.001 | 0.886 ± 0.000 | 0.910 ± 0.000 | — | 15s |
| local | mlp | 0.863 ± 0.001 | 0.984 ± 0.000 | 0.976 ± 0.000 | 1.6M | 1.2m |
| local | gru | 0.879 ± 0.049 | 0.979 ± 0.016 | 0.970 ± 0.025 | 52k | 2.7m |
| local | cnn | 0.899 ± 0.005 | 0.987 ± 0.001 | 0.980 ± 0.001 | 49k | 2.6m |
| local | resnet | 0.902 ± 0.006 | 0.989 ± 0.001 | 0.982 ± 0.002 | 506k | 10.2m |
| line | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.407 ± 0.000 | — | 0s |
| line | random_forest | 0.935 ± 0.000 | 0.868 ± 0.004 | 0.896 ± 0.003 | — | 21s |
| line | mlp | 0.866 ± 0.001 | 0.987 ± 0.000 | 0.980 ± 0.000 | 3.1M | 1.4m |
| line | gru | 0.887 ± 0.031 | 0.989 ± 0.004 | 0.983 ± 0.006 | 55k | 2.6m |
| line | cnn | 0.895 ± 0.010 | 0.987 ± 0.000 | 0.979 ± 0.001 | 51k | 2.5m |
| line | resnet | 0.897 ± 0.017 | 0.988 ± 0.003 | 0.981 ± 0.005 | 509k | 8.4m |
| global | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.407 ± 0.000 | — | 0s |
| global | random_forest | 0.934 ± 0.001 | 0.500 ± 0.000 | 0.407 ± 0.000 | — | 6.9m |
| global | mlp | 0.854 ± 0.000 | 0.714 ± 0.072 | 0.735 ± 0.082 | 103.4M | 47.3m |
| global | gru | 0.854 ± 0.004 | 0.848 ± 0.222 | 0.836 ± 0.224 | 211k | 35.8m |
| global | cnn | 0.865 ± 0.010 | 0.594 ± 0.267 | 0.518 ± 0.311 | 234k | 40.9m |
| global | resnet | 0.870 ± 0.001 | 0.500 ± 0.000 | 0.407 ± 0.000 | 744k | 48.6m |

**grounded vs ungrounded** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| local | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.379 ± 0.000 | — | 0s |
| local | random_forest | 0.943 ± 0.002 | 0.884 ± 0.002 | 0.895 ± 0.002 | — | 12s |
| local | mlp | 0.968 ± 0.000 | 0.995 ± 0.000 | 0.996 ± 0.000 | 1.6M | 35s |
| local | gru | 0.967 ± 0.001 | 0.992 ± 0.001 | 0.993 ± 0.001 | 52k | 1.2m |
| local | cnn | 0.969 ± 0.001 | 0.997 ± 0.001 | 0.998 ± 0.001 | 49k | 1.3m |
| local | resnet | 0.969 ± 0.001 | 0.997 ± 0.001 | 0.998 ± 0.001 | 506k | 6.9m |
| line | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.379 ± 0.000 | — | 0s |
| line | random_forest | 0.952 ± 0.001 | 0.910 ± 0.001 | 0.919 ± 0.001 | — | 16s |
| line | mlp | 0.970 ± 0.000 | 0.998 ± 0.000 | 0.998 ± 0.000 | 3.1M | 45s |
| line | gru | 0.969 ± 0.001 | 0.998 ± 0.001 | 0.998 ± 0.001 | 55k | 1.7m |
| line | cnn | 0.970 ± 0.000 | 0.999 ± 0.000 | 0.999 ± 0.000 | 51k | 1.7m |
| line | resnet | 0.970 ± 0.000 | 0.998 ± 0.001 | 0.999 ± 0.001 | 509k | 6.4m |
| global | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.379 ± 0.000 | — | 0s |
| global | random_forest | 0.955 ± 0.001 | 0.870 ± 0.009 | 0.886 ± 0.008 | — | 2.8m |
| global | mlp | 0.950 ± 0.003 | 0.926 ± 0.006 | 0.917 ± 0.005 | 103.4M | 33.0m |
| global | gru | 0.936 ± 0.021 | 0.902 ± 0.057 | 0.895 ± 0.067 | 211k | 22.9m |
| global | cnn | 0.967 ± 0.001 | 0.961 ± 0.027 | 0.952 ± 0.034 | 234k | 30.0m |
| global | resnet | 0.967 ± 0.003 | 0.967 ± 0.006 | 0.959 ± 0.008 | 744k | 27.2m |

**SD — switch detection** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| global | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.498 ± 0.000 | — | 0s |
| global | random_forest | 0.922 ± 0.000 | 0.778 ± 0.055 | 0.727 ± 0.209 | — | 11.5m |
| global | mlp | 0.754 ± 0.015 | 0.760 ± 0.016 | 0.442 ± 0.034 | 103.4M | 1.6h |
| global | gru | 0.720 ± 0.041 | 0.789 ± 0.021 | 0.411 ± 0.016 | 211k | 1.7h |
| global | cnn | 0.783 ± 0.011 | 0.769 ± 0.044 | 0.418 ± 0.033 | 234k | 1.7h |
| global | resnet | 0.888 ± 0.013 | 0.654 ± 0.092 | 0.627 ± 0.098 | 744k | 1.6h |

#### Tier 2 — medium


**ES — event state (3-class)** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| global | majority | 0.333 ± 0.000 | 0.333 ± 0.000 | 0.162 ± 0.000 | — | 0s |
| global | random_forest | 0.911 ± 0.001 | 0.643 ± 0.021 | 0.383 ± 0.028 | — | 16.2m |
| global | mlp | 0.787 ± 0.001 | 0.749 ± 0.026 | 0.770 ± 0.028 | 103.4M | 2.0h |
| global | gru | 0.637 ± 0.037 | 0.636 ± 0.009 | 0.338 ± 0.005 | 212k | 1.5h |
| global | cnn | 0.707 ± 0.016 | 0.681 ± 0.144 | 0.453 ± 0.209 | 234k | 1.7h |
| global | resnet | 0.729 ± 0.034 | 0.656 ± 0.224 | 0.613 ± 0.229 | 744k | 1.3h |

**fault category (shc/incipient/hif)** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| local | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.329 ± 0.000 | — | 0s |
| local | random_forest | 0.951 ± 0.000 | 0.997 ± 0.000 | 0.631 ± 0.001 | — | 12s |
| local | mlp | 0.959 ± 0.001 | 0.993 ± 0.001 | 0.593 ± 0.007 | 1.6M | 29s |
| local | gru | 0.960 ± 0.001 | 0.993 ± 0.001 | 0.592 ± 0.007 | 53k | 57s |
| local | cnn | 0.955 ± 0.005 | 0.984 ± 0.005 | 0.537 ± 0.024 | 49k | 28s |
| local | resnet | 0.955 ± 0.004 | 0.984 ± 0.006 | 0.538 ± 0.032 | 506k | 2.3m |
| line | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.329 ± 0.000 | — | 0s |
| line | random_forest | 0.951 ± 0.001 | 0.999 ± 0.000 | 0.658 ± 0.001 | — | 17s |
| line | mlp | 0.963 ± 0.003 | 0.998 ± 0.002 | 0.639 ± 0.020 | 3.1M | 32s |
| line | gru | 0.964 ± 0.001 | 0.998 ± 0.001 | 0.648 ± 0.008 | 55k | 1.1m |
| line | cnn | 0.964 ± 0.003 | 0.997 ± 0.002 | 0.634 ± 0.021 | 52k | 48s |
| line | resnet | 0.962 ± 0.003 | 0.997 ± 0.002 | 0.631 ± 0.021 | 509k | 2.4m |
| global | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.329 ± 0.000 | — | 0s |
| global | random_forest | 0.941 ± 0.000 | 0.667 ± 0.016 | 0.496 ± 0.011 | — | 2.8m |
| global | mlp | 0.962 ± 0.001 | 0.536 ± 0.088 | 0.351 ± 0.044 | 103.4M | 31.4m |
| global | gru | 0.962 ± 0.002 | 0.910 ± 0.055 | 0.530 ± 0.079 | 212k | 18.4m |
| global | cnn | 0.955 ± 0.002 | 0.693 ± 0.332 | 0.416 ± 0.150 | 234k | 19.0m |
| global | resnet | 0.957 ± 0.004 | 0.794 ± 0.334 | 0.478 ± 0.171 | 744k | 14.4m |

**FC — fault classification** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| local | majority | 0.143 ± 0.000 | 0.143 ± 0.000 | 0.036 ± 0.000 | — | 0s |
| local | random_forest | 0.665 ± 0.002 | 0.667 ± 0.002 | 0.496 ± 0.001 | — | 10s |
| local | mlp | 0.675 ± 0.003 | 0.711 ± 0.000 | 0.526 ± 0.012 | 1.6M | 1.5m |
| local | gru | 0.678 ± 0.010 | 0.703 ± 0.003 | 0.509 ± 0.005 | 53k | 2.2m |
| local | cnn | 0.679 ± 0.011 | 0.709 ± 0.002 | 0.519 ± 0.009 | 50k | 1.8m |
| local | resnet | 0.681 ± 0.013 | 0.704 ± 0.008 | 0.510 ± 0.024 | 507k | 3.3m |
| line | majority | 0.143 ± 0.000 | 0.143 ± 0.000 | 0.036 ± 0.000 | — | 0s |
| line | random_forest | 0.661 ± 0.002 | 0.679 ± 0.001 | 0.515 ± 0.001 | — | 13s |
| line | mlp | 0.674 ± 0.005 | 0.715 ± 0.000 | 0.527 ± 0.010 | 3.1M | 2.1m |
| line | gru | 0.671 ± 0.007 | 0.715 ± 0.002 | 0.543 ± 0.014 | 56k | 2.1m |
| line | cnn | 0.666 ± 0.013 | 0.713 ± 0.002 | 0.534 ± 0.014 | 52k | 1.2m |
| line | resnet | 0.682 ± 0.006 | 0.716 ± 0.003 | 0.524 ± 0.015 | 510k | 3.2m |
| global | majority | 0.143 ± 0.000 | 0.143 ± 0.000 | 0.036 ± 0.000 | — | 0s |
| global | random_forest | 0.684 ± 0.008 | 0.683 ± 0.002 | 0.494 ± 0.015 | — | 4.3m |
| global | mlp | 0.645 ± 0.004 | 0.606 ± 0.041 | 0.439 ± 0.033 | 103.4M | 33.1m |
| global | gru | 0.620 ± 0.008 | 0.446 ± 0.089 | 0.325 ± 0.058 | 212k | 9.9m |
| global | cnn | 0.634 ± 0.019 | 0.385 ± 0.052 | 0.277 ± 0.061 | 235k | 10.0m |
| global | resnet | 0.655 ± 0.015 | 0.432 ± 0.081 | 0.316 ± 0.068 | 745k | 8.9m |

**fault origin (line vs bus)** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| global | majority | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.461 ± 0.000 | — | 0s |
| global | random_forest | 0.548 ± 0.005 | 0.534 ± 0.006 | 0.533 ± 0.003 | — | 3.1m |
| global | mlp | 0.869 ± 0.003 | 0.970 ± 0.003 | 0.914 ± 0.009 | 103.4M | 32.4m |
| global | gru | 0.858 ± 0.007 | 0.973 ± 0.003 | 0.939 ± 0.015 | 211k | 21.4m |
| global | cnn | 0.860 ± 0.006 | 0.951 ± 0.024 | 0.919 ± 0.039 | 234k | 21.5m |
| global | resnet | 0.859 ± 0.010 | 0.961 ± 0.025 | 0.895 ± 0.066 | 744k | 16.3m |

**faulted phase (7-class)** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| local | majority | 0.143 ± 0.000 | 0.143 ± 0.000 | 0.035 ± 0.000 | — | 0s |
| local | random_forest | 0.914 ± 0.002 | 0.955 ± 0.001 | 0.955 ± 0.001 | — | 13s |
| local | mlp | 0.856 ± 0.001 | 0.980 ± 0.000 | 0.980 ± 0.000 | 1.6M | 34s |
| local | gru | 0.865 ± 0.007 | 0.969 ± 0.004 | 0.969 ± 0.004 | 53k | 2.0m |
| local | cnn | 0.897 ± 0.001 | 0.983 ± 0.003 | 0.983 ± 0.003 | 49k | 1.6m |
| local | resnet | 0.904 ± 0.029 | 0.990 ± 0.009 | 0.990 ± 0.008 | 506k | 6.2m |
| line | majority | 0.143 ± 0.000 | 0.143 ± 0.000 | 0.035 ± 0.000 | — | 0s |
| line | random_forest | 0.926 ± 0.001 | 0.972 ± 0.001 | 0.973 ± 0.001 | — | 18s |
| line | mlp | 0.854 ± 0.002 | 0.982 ± 0.000 | 0.982 ± 0.000 | 3.1M | 44s |
| line | gru | 0.874 ± 0.011 | 0.981 ± 0.003 | 0.981 ± 0.003 | 55k | 1.9m |
| line | cnn | 0.893 ± 0.001 | 0.984 ± 0.001 | 0.984 ± 0.001 | 52k | 1.7m |
| line | resnet | 0.916 ± 0.003 | 0.993 ± 0.001 | 0.993 ± 0.001 | 510k | 7.2m |
| global | majority | 0.143 ± 0.000 | 0.143 ± 0.000 | 0.035 ± 0.000 | — | 0s |
| global | random_forest | 0.923 ± 0.002 | 0.972 ± 0.005 | 0.973 ± 0.005 | — | 2.7m |
| global | mlp | 0.848 ± 0.001 | 0.939 ± 0.006 | 0.941 ± 0.007 | 103.4M | 31.9m |
| global | gru | 0.824 ± 0.006 | 0.848 ± 0.021 | 0.852 ± 0.018 | 212k | 20.6m |
| global | cnn | 0.837 ± 0.003 | 0.880 ± 0.065 | 0.889 ± 0.057 | 235k | 19.1m |
| global | resnet | 0.842 ± 0.006 | 0.768 ± 0.077 | 0.783 ± 0.085 | 745k | 16.3m |

**ST — switch type** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| global | majority | 0.200 ± 0.000 | 0.200 ± 0.000 | 0.054 ± 0.000 | — | 0s |
| global | random_forest | 0.872 ± 0.002 | 0.664 ± 0.023 | 0.420 ± 0.016 | — | 2.4m |
| global | mlp | 0.719 ± 0.005 | 0.270 ± 0.005 | 0.088 ± 0.002 | 103.4M | 20.0m |
| global | gru | 0.669 ± 0.016 | 0.251 ± 0.017 | 0.069 ± 0.029 | 212k | 13.3m |
| global | cnn | 0.682 ± 0.011 | 0.231 ± 0.030 | 0.072 ± 0.015 | 235k | 12.9m |
| global | resnet | 0.689 ± 0.022 | 0.234 ± 0.021 | 0.083 ± 0.013 | 745k | 8.2m |

#### Tier 3 — hard frontier


**EC — event classification (23-class)** (balanced_accuracy ↑)

| observability | baseline | balanced_accuracy (test) | balanced_accuracy (bench) | macro_f1 (bench) | params | train |
|---|---|---|---|---|---|---|
| global | majority | 0.062 ± 0.000 | 0.062 ± 0.000 | 0.021 ± 0.000 | — | 0s |
| global | random_forest | 0.762 ± 0.002 | 0.553 ± 0.018 | 0.327 ± 0.009 | — | 9.5m |
| global | mlp | 0.616 ± 0.003 | 0.262 ± 0.007 | 0.131 ± 0.005 | 103.4M | 1.8h |
| global | gru | 0.592 ± 0.037 | 0.219 ± 0.016 | 0.108 ± 0.012 | 214k | 1.3h |
| global | cnn | 0.582 ± 0.017 | 0.226 ± 0.011 | 0.121 ± 0.007 | 237k | 46.2m |
| global | resnet | 0.640 ± 0.024 | 0.283 ± 0.045 | 0.151 ± 0.026 | 747k | 39.2m |

**FL — fault location (regression)** (mae ↓)

| observability | baseline | mae (test) | mae (bench) | rmse (bench) | params | train |
|---|---|---|---|---|---|---|
| local | majority | 25.224 ± 0.000 | 31.129 ± 0.000 | 36.039 ± 0.000 | — | 0s |
| local | random_forest | 24.497 ± 0.022 | 28.371 ± 0.035 | 33.417 ± 0.040 | — | 37s |
| local | mlp | 22.484 ± 0.027 | 21.403 ± 0.013 | 26.036 ± 0.057 | 1.6M | 60s |
| local | gru | 22.716 ± 0.097 | 21.981 ± 0.495 | 26.585 ± 0.519 | 52k | 1.1m |
| local | cnn | 22.981 ± 0.326 | 22.972 ± 0.660 | 28.608 ± 0.747 | 49k | 58s |
| local | resnet | 23.077 ± 0.317 | 23.973 ± 0.637 | 30.227 ± 1.104 | 506k | 2.2m |
| line | majority | 25.224 ± 0.000 | 31.129 ± 0.000 | 36.039 ± 0.000 | — | 0s |
| line | random_forest | 23.590 ± 0.020 | 23.631 ± 0.039 | 28.441 ± 0.050 | — | 1.2m |
| line | mlp | 18.681 ± 0.137 | 9.396 ± 0.257 | 12.472 ± 0.328 | 3.1M | 1.0m |
| line | gru | 18.887 ± 0.218 | 10.259 ± 0.662 | 13.373 ± 0.605 | 55k | 53s |
| line | cnn | 20.081 ± 0.231 | 13.893 ± 0.551 | 19.018 ± 0.993 | 51k | 40s |
| line | resnet | 20.500 ± 0.473 | 15.229 ± 0.959 | 19.945 ± 0.847 | 509k | 1.8m |
| global | majority | 25.224 ± 0.000 | 31.129 ± 0.000 | 36.039 ± 0.000 | — | 0s |
| global | random_forest | 24.659 ± 0.057 | 29.816 ± 0.039 | 34.613 ± 0.015 | — | 1.3h |
| global | mlp | 20.697 ± 0.115 | 20.389 ± 0.724 | 25.943 ± 1.027 | 103.4M | 33.6m |
| global | gru | 23.203 ± 0.385 | 24.926 ± 1.094 | 29.673 ± 0.985 | 211k | 9.4m |
| global | cnn | 21.447 ± 0.374 | 27.133 ± 4.908 | 36.281 ± 6.039 | 234k | 12.9m |
| global | resnet | 21.816 ± 0.698 | 21.762 ± 3.878 | 28.369 ± 6.023 | 744k | 9.1m |
