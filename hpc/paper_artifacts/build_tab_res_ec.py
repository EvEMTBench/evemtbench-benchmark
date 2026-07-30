#!/usr/bin/env python3
"""Standalone builder → paper/manuscript/generated/tab_res_ec.tex (Event classification, held_out)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_lib import result_table, run  # noqa: E402

if __name__ == "__main__":
    run("tab_res_ec.tex", result_table("ec"))
