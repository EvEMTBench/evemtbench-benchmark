#!/usr/bin/env python3
"""Standalone builder → paper/manuscript/generated/tab_transfer_fc.tex (Fault classification, transfer)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_lib import run, transfer_table

if __name__ == "__main__":
    run("tab_transfer_fc.tex", transfer_table("fc"))
