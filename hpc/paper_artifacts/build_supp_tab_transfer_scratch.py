#!/usr/bin/env python3
"""Standalone builder → paper/manuscript/generated/supp_tab_transfer_scratch.tex
(S3 companion: the source-independent from-scratch target-grid reference for cross-grid transfer).

    PYTHONPATH=src python hpc/paper_artifacts/build_supp_tab_transfer_scratch.py            # write
    PYTHONPATH=src python hpc/paper_artifacts/build_supp_tab_transfer_scratch.py --check    # verify, write nothing
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_lib import run
from supplement_lib import transfer_scratch_table

if __name__ == "__main__":
    run("supp_tab_transfer_scratch.tex", transfer_scratch_table())
