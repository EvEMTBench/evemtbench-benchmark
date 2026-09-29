#!/usr/bin/env python3
"""Standalone builder → paper/manuscript/generated/supp_tab_splits.tex (F13: held-out split
episode counts per grid)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_lib import dataset_split_table, run

if __name__ == "__main__":
    run("supp_tab_splits.tex", dataset_split_table())
