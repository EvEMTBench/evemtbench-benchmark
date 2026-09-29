#!/usr/bin/env python3
"""Standalone builder → paper/manuscript/generated/supp_tab_ec_benchmark.tex (EC, benchmark)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_lib import run, supp_table

if __name__ == "__main__":
    run("supp_tab_ec_benchmark.tex", supp_table("ec", "benchmark"))
