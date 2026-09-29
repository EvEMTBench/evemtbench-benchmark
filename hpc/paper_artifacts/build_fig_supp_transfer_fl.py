#!/usr/bin/env python3
"""Standalone builder → paper/manuscript/figures/fig_supp_transfer_fl.{pdf,svg,png}
(S3: cross-grid transfer for one fault-triad function, as a figure).

Replaces the 8-column transfer table, which the scale-to-fit box rendered below the legibility
floor. Exact values remain in ``results/<version>/aggregated_results.csv``.

Requires matplotlib and is compute -- run via srun, never the login node.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from supp_figure_lib import build_transfer_figure

if __name__ == "__main__":
    build_transfer_figure("fl")
