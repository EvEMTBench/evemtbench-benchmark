#!/usr/bin/env python3
"""Standalone builder → paper/manuscript/generated/supp_tab_diagnostics.tex (F04: per-baseline
balanced accuracy for the seven diagnostic functions)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_lib import diagnostic_overview_table, run  # noqa: E402

if __name__ == "__main__":
    run("supp_tab_diagnostics.tex", diagnostic_overview_table())
