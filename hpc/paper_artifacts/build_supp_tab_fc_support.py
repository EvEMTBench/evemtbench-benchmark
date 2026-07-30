#!/usr/bin/env python3
"""Standalone builder → paper/manuscript/generated/supp_tab_fc_support.tex (F13: per-class FC
support counts per grid)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_lib import fc_support_table, run  # noqa: E402

if __name__ == "__main__":
    run("supp_tab_fc_support.tex", fc_support_table())
