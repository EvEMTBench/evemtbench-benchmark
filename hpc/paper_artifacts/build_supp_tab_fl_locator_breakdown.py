#!/usr/bin/env python3
"""Standalone builder → paper/manuscript/generated/supp_tab_fl_locator_breakdown.tex
(fault-location error of the impedance locator and the selected learned reference by window
position, plus the one-ended CIGRE MV line (needs the mounted dataset)).

Reads the released dataset (``$EVEMT_ROOT``: ``overlap_ratio`` and ``y_fault_line`` of each window)
in addition to the committed prediction files, so it must run where the windowed data is mounted —
on the cluster that means an ``srun`` allocation, never the login node.

    PYTHONPATH=src python hpc/paper_artifacts/build_supp_tab_fl_locator_breakdown.py            # write
    PYTHONPATH=src python hpc/paper_artifacts/build_supp_tab_fl_locator_breakdown.py --check    # verify, write nothing
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_lib import run
from supplement_lib import fl_locator_breakdown_table

if __name__ == "__main__":
    run("supp_tab_fl_locator_breakdown.tex", fl_locator_breakdown_table())
