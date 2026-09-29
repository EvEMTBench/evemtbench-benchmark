#!/usr/bin/env python3
"""Standalone builder → paper/manuscript/figures/fig_supp_ed.{pdf,svg,png}
(S2: the full per-baseline event detection matrix, as a figure).

Replaces the two 13-column S2 tables for this function. Those tables were wrapped in a
scale-to-fit box, which shrank 7 pt type to ~3.8 pt on the page -- below the ~5 pt legibility
floor, and in violation of the journal's "same typeface, same size" rule for display items. The
exact values remain machine-readable in ``results/<version>/aggregated_results.csv``.

Requires matplotlib (NOT in the sweep's py_dl env) and is compute -- run via srun, never the
login node:

    srun --partition=a100mig --gres=gpu:a100small:1 --time=10:00 bash -c \
      'CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
       "$PAPERFIGS_PY" hpc/paper_artifacts/build_fig_supp_ed.py'
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from supp_figure_lib import build_per_baseline_figure

if __name__ == "__main__":
    build_per_baseline_figure("ed")
