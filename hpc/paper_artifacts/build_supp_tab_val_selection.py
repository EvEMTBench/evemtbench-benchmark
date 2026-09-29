#!/usr/bin/env python3
"""Standalone builder → paper/manuscript/generated/supp_tab_val_selection.tex (S10: the
validation-split selection audit — the winner's validation score and identity per primary cell).

    PYTHONPATH=src python hpc/paper_artifacts/build_supp_tab_val_selection.py            # write
    PYTHONPATH=src python hpc/paper_artifacts/build_supp_tab_val_selection.py --check    # verify
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_lib import run, val_selection_table

if __name__ == "__main__":
    run("supp_tab_val_selection.tex", val_selection_table())
