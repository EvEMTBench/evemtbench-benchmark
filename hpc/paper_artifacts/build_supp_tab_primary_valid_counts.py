#!/usr/bin/env python3
"""Standalone builder → paper/manuscript/generated/supp_tab_primary_valid_counts.tex
(S9: evaluated window counts (the reported n) for the five primary functions).

    PYTHONPATH=src python hpc/paper_artifacts/build_supp_tab_primary_valid_counts.py            # write
    PYTHONPATH=src python hpc/paper_artifacts/build_supp_tab_primary_valid_counts.py --check    # verify, write nothing
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_lib import run
from supplement_lib import primary_valid_counts_table

if __name__ == "__main__":
    run("supp_tab_primary_valid_counts.tex", primary_valid_counts_table())
