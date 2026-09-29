#!/usr/bin/env python3
"""Standalone builder → paper/manuscript/generated/tab_baselines.tex (static specification table)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_lib import run, spec_table

if __name__ == "__main__":
    run("tab_baselines.tex", spec_table("baselines"))
