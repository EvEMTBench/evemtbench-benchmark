#!/usr/bin/env python3
"""Standalone builder → paper/manuscript/figures/fig_supp_protection.{pdf,svg,png}
(S8: protection-grade dependability and security, as a figure).

Replaces the 10-column protection table. The always-positive collapse -- balanced accuracy ~0.5
with MDR ~0 and FAR ~1 -- becomes a visual signature rather than three numbers on a 6.97 pt line.

Requires matplotlib and is compute -- run via srun, never the login node.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from supp_figure_lib import build_protection_figure  # noqa: E402

if __name__ == "__main__":
    build_protection_figure()
