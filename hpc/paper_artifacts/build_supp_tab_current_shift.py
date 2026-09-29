#!/usr/bin/env python3
"""Standalone builder → paper/manuscript/generated/supp_tab_current_shift.tex
(S7: raw steady-state current scale, adapt_grid train vs eval_grid (needs the mounted dataset)).

Reads the released dataset (``$EVEMT_ROOT``) in addition to the committed
repository artefacts, so it must run where the windowed data is mounted — on the
cluster that means an ``srun`` allocation, never the login node.

    PYTHONPATH=src python hpc/paper_artifacts/build_supp_tab_current_shift.py            # write
    PYTHONPATH=src python hpc/paper_artifacts/build_supp_tab_current_shift.py --check    # verify, write nothing
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_lib import run
from supplement_lib import current_shift_table

if __name__ == "__main__":
    run("supp_tab_current_shift.tex", current_shift_table())
